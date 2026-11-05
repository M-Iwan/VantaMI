from typing import Dict, Optional, Tuple, Any, Iterable, List

import numpy as np

import torch
from torch import Tensor, nn
from torch_geometric.nn import global_mean_pool

from vantami.dl.unit import MMUnit, MMRegressionUnit, MMClassificationUnit
from vantami.dl.modules import build_graph_layers, build_linear_layers


LossSummary = Dict[str, Any]
LossValue = Tuple[Tensor, Tensor]


class IonMeleon(MMRegressionUnit):
    """
    CheMeleon-like model implemented using our code.
    
    Parameters
    ----------
    device: str
        Device to use. Default is cpu.
    num_task: int, optional.
        Number of regresison outputs. Default is 1613 (Mordred descriptors size)
    model_name: str, optional
        Default is IonMeleon.
    gnn_params: dict
        Dictionary for build_graph_layers
    lin_params: dict
        Dictionary for build_linear_layers
    max_grad_norm: float
        Gradient clipping max norm in fit_epoch. Default is 1.0.
    verbose: int, optional
        Set verbosity level. 0 = no logs, 1 = standard progress, 2 = debugging
    """

    def __init__(self, device: str = 'cpu', num_task: int = 1, model_name: str = 'IonMeleon',
                 gnn_params: Optional[Dict] = None, lin_params: Optional[Dict] = None,
                 max_grad_norm: Optional[float] = 1.0, keep_fraction: Optional[float] = 0.15,
                 verbose: Optional[int] = 1):
        super().__init__(device=device, num_task=num_task, model_name=model_name, max_grad_norm=max_grad_norm)
        self.gnn_params = gnn_params
        self.lin_params = lin_params
        self.gnn_input_name = self.gnn_params.get('input_name', "Graph")
        self.gnn_layer, self.gnn_out_size = build_graph_layers(self.gnn_params)
        self.lin_params["sizes"] = [self.gnn_out_size] + self.lin_params.get("sizes") + [self.num_task]
        self.lin_layers, _ = build_linear_layers(**self.lin_params)
        self.keep_fraction = keep_fraction

    def forward(self, batch_data: dict):
        """
        Message passing on the graph batch, then global mean pool and lin_layers.

        Parameters
        ----------
        batch_data : dict

        Returns
        -------
        x_out : torch.Tensor
            Shape (batch, num_task).
        """
        graph_batch = batch_data[self.gnn_input_name].to(self.device)

        for layer in self.gnn_layer:
            x = layer(graph_batch)
            graph_batch.x = x

        x_out = global_mean_pool(graph_batch.x, graph_batch.batch)
        x_out = self.lin_layers(x_out)

        return x_out

    def fit_epoch(self, dataloader: Iterable[dict]) -> Tuple[LossSummary, Dict[str, Any], List[float]]:
        """
        Overwrite the base method to sample fraction of target values.

        Parameters
        ----------
        dataloader : iterable
            Iterable yielding batches.

        Returns
        -------
        tuple
            Normalised epoch loss, epoch metrics, and per-batch gradient norms.
        """
        if self.optimizer is None:
            raise RuntimeError("An optimizer must be configured before training.")

        self.train()

        batch_losses: List[LossSummary] = []
        epoch_data = {"y_pred": [], "y_true": [], "y_wgts": []}
        gradient_norms = []

        for batch_data in dataloader:
            self.optimizer.zero_grad()

            y_pred = self(batch_data)
            y_true = batch_data["y_true"].to(self.device)
            y_wgts = batch_data["y_wgts"].to(self.device)

            target_mask = torch.rand_like(y_true, dtype=torch.float16) < self.keep_fraction
            masked_y_wgts = y_wgts * target_mask.to(dtype=y_wgts.dtype)
            if masked_y_wgts.sum() == 0:
                continue

            epoch_data["y_pred"].append(y_pred.detach())
            epoch_data["y_true"].append(y_true.detach())
            epoch_data["y_wgts"].append(y_wgts.detach())

            batch_loss = self.loss(
                y_pred=y_pred,
                y_true=y_true,
                y_wgts=y_wgts,
                group=self._get_batch_group(batch_data),
            )

            total_loss, total_weight = batch_loss["Total"]
            total_loss.backward()

            gradient_norms.append(self.grad_norm() / (total_weight.item() + 1e-6))

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    max_norm=self.max_grad_norm,
                )

            self.optimizer.step()
            batch_losses.append(batch_loss)

        if self.scheduler is not None:
            self.scheduler.step()

        epoch_loss, epoch_metrics = self._summarise_epoch(batch_losses, epoch_data)
        return epoch_loss, epoch_metrics, gradient_norms
