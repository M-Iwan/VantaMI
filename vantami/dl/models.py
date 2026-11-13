from copy import copy
from typing import Dict, Tuple, Any, Iterable, List

import torch
from torch import Tensor, nn
from torch_geometric.nn import global_mean_pool

from vantami.dl.unit import MMRegressionUnit
from vantami.dl.modules import build_graph_layers, build_linear_layers


LossSummary = Dict[str, Any]
LossValue = Tuple[Tensor, Tensor]


class SingleGNN(MMRegressionUnit):
    """
    Single channel covalent graph neural network.
    """

    def __init__(self, device: str, node_input_size: int, cov_gnn_params: dict, lin_params: dict, num_task: int,
                 autocast: bool = False, autocast_dtype: torch.dtype = torch.bfloat16,
                 max_grad_norm: float = 1.0, verbose: int = 1, seed: int | None = 42,
                 log_config: Dict[str, bool] | None = None, model_name: str = "SingleGNN"):

        """
        Initialize SingleGNN class

        Parameters
        ----------
        device: str
            Device to use. Default is cpu.
        node_input_size: int
            Node encoding size.
        cov_gnn_params: dict
            Dictionary of GNN parameters for covalent channel
        lin_params: dict
            Dictionary for build_linear_layers
        num_task: int, optional
            Number of regression outputs
        autocast: bool, default=False
            Flag to enable autocasting of dtypes
        autocast_dtype: torch.dtype, default=torch.bfloat16
            DataType to use when autocasting
        max_grad_norm: float
            Gradient clipping max norm in fit_epoch. Default is 1.0.
        verbose: int, optional
            Set verbosity level. 0 = no logs, 1 = standard progress, 2 = debugging
        model_name: str, optional
            Name of the model
        log_config: Dict[str, bool]
            Logging configuration, with the following (option | default):
                * `per_task_loss`: False
                * `gradient_norm`: True
        """

        super().__init__(device=device, num_task=num_task, model_name=model_name, autocast=autocast,
                         autocast_dtype=autocast_dtype, max_grad_norm=max_grad_norm, verbose=verbose,
                         seed=seed, log_config=log_config)

        self.node_input_size = node_input_size
        self.cov_input_size = cov_gnn_params["input_dim"]

        self.cov_gnn_params = cov_gnn_params
        self.lin_params = dict(lin_params)

        self.adaptation_layer = nn.Sequential(
            nn.Linear(self.node_input_size, self.cov_input_size),
            nn.SiLU()
        )

        self.cov_gnn_layer, self.cov_gnn_out_size = build_graph_layers(self.cov_gnn_params)

        self.lin_params["sizes"] = [self.cov_gnn_out_size] + self.lin_params["sizes"]
        self.lin_layers, self.lin_out_size = build_linear_layers(**self.lin_params)

        self.prediction_heads = nn.Linear(self.lin_out_size, self.num_task)
        self.to(self.device)

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
        graph_batch = batch_data["graph"]

        graph_batch.x = self.adaptation_layer(graph_batch.x)

        for layer in self.cov_gnn_layer:
            graph_batch.x = layer(graph_batch)

        x = global_mean_pool(x=graph_batch.x, batch=graph_batch.batch)

        for layer in self.lin_layers:
            x = layer(x)

        x_out = self.prediction_heads(x)

        return x_out


class DualGNN(MMRegressionUnit):
    """
    Dual channel covalent/spatial graph neural network.
    """

    def __init__(self, device: str, node_input_size: int, cov_gnn_params: dict, spatial_gnn_params: dict,
                 lin_params: dict, num_task: int, autocast: bool = False, autocast_dtype: torch.dtype = torch.bfloat16,
                 max_grad_norm: float = 1.0, verbose: int = 1, seed: int | None = 42,
                 log_config: Dict[str, bool] | None = None, model_name: str = "DualGNN"):

        """
        Initialize DualGNN class

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
        autocast: bool, default=False
            Flag to enable autocasting of dtypes
        autocast_dtype: torch.dtype, default=torch.bfloat16
            DataType to use when autocasting
        max_grad_norm: float
            Gradient clipping max norm in fit_epoch. Default is 1.0.
        verbose: int, optional
            Set verbosity level. 0 = no logs, 1 = standard progress, 2 = debugging
        model_name: str, optional
            Name of the model
        log_config: Dict[str, bool]
            Logging configuration, with the following (option | default):
                * `per_task_loss`: False
                * `gradient_norm`: True
        """

        super().__init__(device=device, num_task=num_task, model_name=model_name, autocast=autocast,
                         autocast_dtype=autocast_dtype, max_grad_norm=max_grad_norm, verbose=verbose,
                         seed=seed, log_config=log_config)

        self.node_input_size = node_input_size
        self.cov_input_size = cov_gnn_params["input_dim"]

        self.cov_gnn_params = cov_gnn_params
        self.spatial_gnn_params = spatial_gnn_params
        self.lin_params = dict(lin_params)

        self.adaptation_layer = nn.Sequential(
            nn.Linear(self.node_input_size, self.cov_input_size),
            nn.SiLU()
        )

        self.cov_gnn_layer, self.cov_gnn_out_size = build_graph_layers(self.cov_gnn_params)
        self.spatial_gnn_layer, self.spatial_gnn_out_size = build_graph_layers(self.spatial_gnn_params)
        self.gnn_out_size = self.spatial_gnn_out_size + self.cov_gnn_out_size

        self.lin_params["sizes"] = [self.gnn_out_size] + self.lin_params.get("sizes", [])
        self.lin_layers, self.lin_out_size = build_linear_layers(**self.lin_params)

        self.prediction_heads = nn.Linear(self.lin_out_size, self.num_task)
        self.to(self.device)

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
        graph_batch = batch_data["graph"]

        adapted_x = self.adaptation_layer(graph_batch.x)

        cov_graph = copy(graph_batch)
        spatial_graph = copy(graph_batch)

        cov_graph.x = adapted_x
        spatial_graph.x = adapted_x

        for layer in self.cov_gnn_layer:
            cov_graph.x = layer(cov_graph)

        for layer in self.spatial_gnn_layer:
            spatial_graph.x = layer(spatial_graph)

        cov_pool = global_mean_pool(x=cov_graph.x, batch=cov_graph.batch)
        spatial_pool = global_mean_pool(x=spatial_graph.x, batch=spatial_graph.batch)

        x = torch.cat(tensors=(cov_pool, spatial_pool), dim=-1)

        for layer in self.lin_layers:
            x = layer(x)

        x_out = self.prediction_heads(x)

        return x_out


class AngularGNN(MMRegressionUnit):
    """
    Dual channel covalent/spatial graph neural network with added Angle ACSF.
    """

    def __init__(self, device: str, node_input_size: int, abf_input_size: int, cov_gnn_params: dict, spatial_gnn_params: dict,
                 lin_params: dict, num_task: int, autocast: bool = False, autocast_dtype: torch.dtype = torch.bfloat16,
                 max_grad_norm: float = 1.0, verbose: int = 1, seed: int | None = 42,
                 log_config: Dict[str, bool] | None = None, model_name: str = "AngularGNN"):

        """
        Initialize AngularGNN class

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
        autocast: bool, default=False
            Flag to enable autocasting of dtypes
        autocast_dtype: torch.dtype, default=torch.bfloat16
            DataType to use when autocasting
        max_grad_norm: float
            Gradient clipping max norm in fit_epoch. Default is 1.0.
        verbose: int, optional
            Set verbosity level. 0 = no logs, 1 = standard progress, 2 = debugging
        model_name: str, optional
            Name of the model
        log_config: Dict[str, bool]
            Logging configuration, with the following (option | default):
                * `per_task_loss`: False
                * `gradient_norm`: True
        """

        super().__init__(device=device, num_task=num_task, model_name=model_name, autocast=autocast,
                         autocast_dtype=autocast_dtype, max_grad_norm=max_grad_norm, verbose=verbose,
                         seed=seed, log_config=log_config)

        self.node_input_size = node_input_size
        self.abf_input_size = abf_input_size
        self.spatial_node_input_size = self.node_input_size + self.abf_input_size
        self.cov_input_size = cov_gnn_params["input_dim"]
        self.spatial_input_size = spatial_gnn_params["input_dim"]

        self.cov_gnn_params = cov_gnn_params
        self.spatial_gnn_params = spatial_gnn_params
        self.lin_params = dict(lin_params)

        self.cov_adaptation_layer = nn.Sequential(
            nn.Linear(self.node_input_size, self.cov_input_size),
            nn.SiLU()
        )
        self.spatial_adaptation_layer = nn.Sequential(
            nn.Linear(self.spatial_node_input_size, self.spatial_input_size),
            nn.SiLU()
        )

        self.cov_gnn_layer, self.cov_gnn_out_size = build_graph_layers(self.cov_gnn_params)
        self.spatial_gnn_layer, self.spatial_gnn_out_size = build_graph_layers(self.spatial_gnn_params)
        self.gnn_out_size = self.spatial_gnn_out_size + self.cov_gnn_out_size

        self.lin_params["sizes"] = [self.gnn_out_size] + self.lin_params.get("sizes", [])
        self.lin_layers, self.lin_out_size = build_linear_layers(**self.lin_params)

        self.prediction_heads = nn.Linear(self.lin_out_size, self.num_task)
        self.to(self.device)

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
        graph_batch = batch_data["graph"]

        spatial_x = torch.cat(tensors=(graph_batch["x"], graph_batch["atom_angular_basis"]), dim=-1)

        cov_graph = copy(graph_batch)
        spatial_graph = copy(graph_batch)

        cov_graph.x = self.cov_adaptation_layer(graph_batch.x)
        spatial_graph.x = self.spatial_adaptation_layer(spatial_x)

        for layer in self.cov_gnn_layer:
            cov_graph.x = layer(cov_graph)

        for layer in self.spatial_gnn_layer:
            spatial_graph.x = layer(spatial_graph)

        cov_pool = global_mean_pool(x=cov_graph.x, batch=cov_graph.batch)
        spatial_pool = global_mean_pool(x=spatial_graph.x, batch=spatial_graph.batch)

        x = torch.cat(tensors=(cov_pool, spatial_pool), dim=-1)

        for layer in self.lin_layers:
            x = layer(x)

        x_out = self.prediction_heads(x)

        return x_out
