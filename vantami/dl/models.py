from copy import copy
from typing import Dict, Optional, Tuple, Any, Iterable, List

import numpy as np

import torch
from torch import Tensor, nn
from torch_geometric.nn import global_mean_pool, global_add_pool

from vantami.dl.unit import MMUnit, MMRegressionUnit, MMClassificationUnit
from vantami.dl.modules import build_graph_layers, build_linear_layers


LossSummary = Dict[str, Any]
LossValue = Tuple[Tensor, Tensor]


class SpatialGINEConv(MMRegressionUnit):
    """
    Dual channel covalent/spatial graph neural network.
    
    Parameters
    ----------
    device: str
        Device to use. Default is cpu.
    node_input_size: int
        Node encoding size.
    cov_gnn_params: dict
        Dictionary of GNN parameters for covalent channel
    spatial_gnn_params: dict
        Dictionary of GNN parameters for spatial channel
    lin_params: dict
        Dictionary for build_linear_layers
    num_task: int, optional
        Number of regression outputs
    keep_fraction: float, optional
        Fraction of target values to use for loss
    max_grad_norm: float
        Gradient clipping max norm in fit_epoch. Default is 1.0.
    verbose: int, optional
        Set verbosity level. 0 = no logs, 1 = standard progress, 2 = debugging
    model_name: str, optional
        Name of the model
    """

    def __init__(self, device: str, node_input_size: int, cov_gnn_params: dict, spatial_gnn_params: dict,
                 lin_params: dict, num_task: int, keep_fraction: float = 0.15, max_grad_norm: float = 1.0,
                 verbose: int = 1, model_name: str = 'IonFlow_I'):

        super().__init__(device=device, num_task=num_task, model_name=model_name,
                         max_grad_norm=max_grad_norm, verbose=verbose)

        self.node_input_size = node_input_size
        self.conv_input_size = cov_gnn_params.get("input_dim")
        self.keep_fraction = keep_fraction

        self.cov_gnn_params = cov_gnn_params
        self.spatial_gnn_params = spatial_gnn_params
        self.lin_params = dict(lin_params)

        self.adaptation_layer = nn.Sequential(
            nn.Linear(self.node_input_size, self.conv_input_size),
            nn.SiLU()
        )

        self.cov_gnn_layer, self.cov_gnn_out_size = build_graph_layers(self.cov_gnn_params)
        self.spatial_gnn_layer, self.spatial_gnn_out_size = build_graph_layers(self.spatial_gnn_params)
        self.gnn_out_size = self.spatial_gnn_out_size + self.cov_gnn_out_size

        self.lin_params["sizes"] = [self.gnn_out_size] + self.lin_params.get("sizes")
        self.lin_layers, self.lin_out_size = build_linear_layers(**self.lin_params)

        self.prediction_heads = nn.Linear(self.lin_out_size, self.num_task)

    def forward(self, batch_data: dict):
        """
        Parameters
        ----------
        batch_data: dict

        Returns
        -------
        x_out: torch.Tensor
            Shape (batch, num_task).
        """
        graph_batch = batch_data["graph"].to(self.device)

        graph_batch.x = self.adaptation_layer(graph_batch.x)

        cov_graph = copy(graph_batch)
        spatial_graph = copy(graph_batch)

        for layer in self.cov_gnn_layer:
            cov_graph.x = layer(cov_graph)

        for layer in self.spatial_gnn_layer:
            spatial_graph.x = layer(spatial_graph)

        cov_pool = global_add_pool(cov_graph.x, cov_graph.batch)
        spatial_pool = global_add_pool(spatial_graph.x, spatial_graph.batch)

        x = torch.cat((cov_pool, spatial_pool), dim=-1)

        for layer in self.lin_layers:
            x = layer(x)

        x_out = self.prediction_heads(x)

        return x_out

    def fit_epoch(self, dataloader: Iterable[dict]) -> Tuple[LossSummary, Dict[str, Any], List[float]]:
        """
        Overwrite the base method to sample fraction of target values.

        Parameters
        ----------
        dataloader: iterable
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
                y_wgts=masked_y_wgts,
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
