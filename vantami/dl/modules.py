"""
Reusable graph, convolution-on-sequence, RNN, and MLP blocks for vantami.dl.models
"""
import inspect
import ast
from typing import List, Optional

import torch
from torch import nn

from vantami.dl.utils import get_activation_fn


class GNNModule(nn.Module):
    """
    Apply one graph-convolution block to a PyTorch Geometric graph batch.

    Parameters
    ----------
    graph_layer: nn.Module
        PyTorch Geometric graph-convolution layer to apply.
    batch_norm: Optional[nn.Module]
        Node-feature normalization layer.
    activation: Optional[nn.Module]
        Activation layer applied after normalization.
    dropout: Optional[nn.Module]
        Dropout layer applied after the activation.
    """

    def __init__(self, graph_layer: nn.Module, batch_norm: Optional[nn.Module] = None,
                 activation: Optional[nn.Module] = None, dropout: Optional[nn.Module] = None):
        super().__init__()
        self.graph_layer = graph_layer
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout = dropout
        self.accepts_edge_attr = "edge_attr" in inspect.signature(self.graph_layer.forward).parameters

    def forward(self, graph_batch):
        """
        Transform node features in a graph batch.

        Parameters
        ----------
        graph_batch: torch_geometric.data.Data or torch_geometric.data.Batch
            Graph object containing `x` and `edge_index` attributes and, optionally, an `edge_attr` attribute.

        Returns
        -------
        torch.Tensor
        """
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_attr = getattr(graph_batch, "edge_attr", None)

        if self.accepts_edge_attr and edge_attr is not None:
            x = self.graph_layer(x, edge_index, edge_attr)
        else:
            x = self.graph_layer(x, edge_index)

        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x


def build_graph_layers(gnn_params):
    """Build a sequence of graph-convolution blocks from a parameter mapping.

    Parameters
    ----------
    gnn_params: dict
        Graph-network configuration. Required keys are:
         * `layer` (a graph-layer class),
         * `layer_type` ("convolutional" | "attention" | "edge"),
        * `sizes` (output dimensions)
        * `input_dim`
        Optional keys are:
        * `args` (per-layer keyword mappings; default `{}` for every layer),
        * `activation` (default "relu"),
        * `dropout` (default 0.0),
        * `batch_norm` (default True)
        * `heads` for attention layers (default 1).

    Returns
    -------
    tuple[nn.ModuleList, int]
        Ordered :class:`GNNModule` blocks and the final node-feature width.
    """
    layer_class = gnn_params["layer"]
    layer_type = gnn_params.get("layer_type")  # default
    sizes = gnn_params["sizes"]
    input_dim = gnn_params["input_dim"]
    layer_args = gnn_params.get("args", [{} for _ in sizes])
    activation = gnn_params.get("activation", "relu")
    dropout = gnn_params.get("dropout", 0.0)
    batch_norm = gnn_params.get("batch_norm", True)

    gnn_blocks = []
    in_dim = input_dim

    for i, (out_dim, args) in enumerate(zip(sizes, layer_args)):

        if layer_type == "attention":
            heads = gnn_params.get("heads", 1)
            next_in = out_dim * heads
            graph_layer = layer_class(in_dim, out_dim, heads=heads, **args)
        elif layer_type == "convolutional":
            next_in = out_dim
            graph_layer = layer_class(in_dim, out_dim, **args)
        elif layer_type == "edge":
            nn_layer = nn.Linear(in_dim * 2, out_dim)
            next_in = out_dim
            graph_layer = layer_class(nn_layer, **args)
        else:
            raise ValueError(f"Unsupported layer_type: {layer_type}")

        batch_norm_layer = nn.BatchNorm1d(next_in) if batch_norm else None
        activation_layer = get_activation_fn(activation) if activation else None
        dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else None

        gnn_blocks.append(
            GNNModule(
                graph_layer=graph_layer,
                batch_norm=batch_norm_layer,
                activation=activation_layer,
                dropout=dropout_layer
            )
        )
        in_dim = next_in

    return nn.ModuleList(gnn_blocks), in_dim


def build_gnn_config():
    """
    Interactively collect a graph-network configuration.

    Returns
    -------
    dict
        Graph-network configuration compatible with :func:`build_graph_layers`.
    """
    try:
        from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, EdgeConv
    except ImportError as exc:
        raise ImportError(f"< build_gnn_config > requires < pytorch_geometric > library") from exc

    layer_types = ["convolutional", "attention", "edge"]
    available_activations = ["relu", "leaky_relu", "gelu", "tanh"]

    def suggest_gnn_layers():
        return {
            "convolutional": [GCNConv, SAGEConv],
            "attention": [GATConv, GATv2Conv],
            "edge": [EdgeConv]
        }

    print("=== GNN Layer Configuration ===")

    while True:
        print(f"Available layer types: {layer_types}")
        layer_type = input("Select layer type: ").strip().lower()
        if layer_type in layer_types:
            break
        print("Invalid layer type. Please try again.")

    suggestions = suggest_gnn_layers()[layer_type]
    layer_class_names = [cls.__name__ for cls in suggestions]
    while True:
        print(f"Suggested layers for < {layer_type} >: {layer_class_names}")
        layer_name = input("Enter the layer class name: ").strip()
        layer_class = next((cls for cls in suggestions if cls.__name__ == layer_name), None)
        if layer_class:
            break
        print("Invalid layer name. Please choose from the suggestions.")

    sizes = input("Enter output sizes (comma-separated, e.g. 64,64,32): ")
    sizes = list(map(int, sizes.split(",")))
    assert all(size > 0 for size in sizes)

    input_dim = int(input("Enter input dimension: "))
    assert input_dim > 0

    layer_args = [{} for _ in sizes]
    edge_dim = None
    heads = 1

    if layer_type == "attention":
        heads = int(input("Enter number of attention heads [1]: ") or 1)
        edge_dim = int(input("Enter edge feature dimension (edge_dim): "))
        assert heads >= 1
        assert edge_dim > 0

    if input("Do you want to enter custom args for each layer? (y/n): ").strip().lower() == "y":
        for i in range(len(sizes)):
            arg_str = input(f"Layer {i} args (as Python dict): ")
            layer_args[i] = ast.literal_eval(arg_str)

    if layer_type == "attention" and edge_dim is not None:
        for args in layer_args:
            args["edge_dim"] = edge_dim

    while True:
        print(f"Available activations: {available_activations}")
        activation = input("Select activation function [relu]: ").strip().lower() or "relu"
        if activation in available_activations:
            break
        print("Unsupported activation function. Try again.")

    while True:
        try:
            dropout = input("Enter dropout rate [0.1]: ").strip()
            dropout = float(dropout) if dropout else 0.1
            if 0.0 <= dropout < 1.0:
                break
            else:
                print("Dropout rate must be between 0 and 1.")
        except ValueError:
            print("Invalid float. Try again.")

    while True:
        try:
            batch_norm = input("Enter batch_norm [y/n]: ").strip()
            if batch_norm in ["y", "n"]:
                batch_norm = {"y": True, "n": False}.get(batch_norm)
                break
            else:
                print(f"Batch norm must be y/n. Try again.")
        except ValueError:
            "Invalid answer. Try again."

    gnn_params = {
        "layer": layer_class,
        "layer_type": layer_type,
        "sizes": sizes,
        "input_dim": input_dim,
        "args": layer_args,
        "activation": activation,
        "dropout": dropout,
        "batch_norm": batch_norm
    }

    if layer_type == "attention":
        gnn_params["heads"] = heads

    return gnn_params


class CNNModule(nn.Module):
    """
    Apply one one-dimensional convolutional sequence block.

    Parameters
    ----------
    conv_layer: nn.Module
        One-dimensional convolution layer.
    batch_norm: Optional[nn.Module]
        Channel normalization layer.
    activation: Optional[nn.Module]
        Activation layer.
    max_pool: Optional[nn.Module]
        One-dimensional max-pooling layer.
    dropout: Optional[nn.Module]
        Dropout layer.
    kernel_size: int
        Convolution kernel size.
    stride: int
        Convolution stride.
    pool_kernel_size: int
        Max-pooling kernel size. A value of `1` disables pooling.
    """

    def __init__(self, conv_layer: nn.Module, batch_norm: Optional[nn.Module] = None,
                 activation: Optional[nn.Module] = None, max_pool: Optional[nn.Module] = None,
                 dropout: Optional[nn.Module] = None, kernel_size: int = 5, stride: int = 1,
                 pool_kernel_size: int = 2):
        super().__init__()
        self.conv_layer = conv_layer
        self.batch_norm = batch_norm
        self.activation = activation
        self.max_pool = max_pool
        self.dropout = dropout

        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x, lengths):
        """
        Transform padded sequences and update their effective lengths.

        Parameters
        ----------
        x: torch.Tensor
            Input sequence tensor with shape `(batch_size, channels, length)`.
        lengths: torch.Tensor
            Unpadded input lengths with shape `(batch_size,)`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
        """
        x = self.conv_layer(x)

        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.max_pool is not None:
            x = self.max_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)

        lengths = ((lengths - self.kernel_size) // self.stride) + 1
        lengths //= self.pool_kernel_size
        lengths = torch.clamp(lengths, min=1)

        return x, lengths


def build_conv_layers(cnn_params):
    """
    Build a token embedding and one-dimensional convolutional blocks.

    Parameters
    ----------
    cnn_params: dict
        Sequence-CNN configuration. Required keys are:
        * `alphabet_len`
        * `embedding_dim`
        Optional keys are:
        * `padding_idx` (default `0`),
        * `sizes` (default `[256]`)
        * `kernel_size` (default `5`),
        * `stride` (default `1`)
        * `pool_size` (default `2`)
        * `activation` (default "relu")
        * `dropout` (default `0.1`)
        * `batch_norm` (default `True`).

    Returns
    -------
    tuple[nn.ModuleList, nn.Embedding, int]
    """
    alphabet_len = cnn_params["alphabet_len"]
    embedding_dim = cnn_params["embedding_dim"]  # i.e. in_channels for the first layer
    padding_idx = cnn_params.get("padding_idx", 0)

    cnn_embedding = nn.Embedding(
        num_embeddings=alphabet_len,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx
    )

    sizes = cnn_params.get("sizes", [256])
    kernel_size = cnn_params.get("kernel_size", 5)
    stride = cnn_params.get("stride", 1)
    dropout = cnn_params.get("dropout", 0.1)
    activation_fn = get_activation_fn(cnn_params.get("activation", "relu"))
    pool_kernel_size = cnn_params.get("pool_size", 2)
    batch_norm = cnn_params.get("batch_norm", True)

    in_channels = embedding_dim
    cnn_blocks = []

    for out_channels in sizes:
        conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )
        batch_norm_layer = nn.BatchNorm1d(out_channels) if batch_norm else None
        activation_layer = activation_fn if activation_fn else None
        max_pool_layer = nn.MaxPool1d(kernel_size=pool_kernel_size) if pool_kernel_size > 1 else None
        dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else None

        cnn_blocks.append(
            CNNModule(
                conv_layer=conv_layer,
                batch_norm=batch_norm_layer,
                activation=activation_layer,
                dropout=dropout_layer,
                max_pool=max_pool_layer,
                kernel_size=kernel_size,
                stride=stride,
                pool_kernel_size=pool_kernel_size
            )
        )
        in_channels = out_channels

    return nn.ModuleList(cnn_blocks), cnn_embedding, in_channels


def build_cnn_config():
    """
    Interactively collect a one-dimensional sequence-CNN configuration.

    Returns
    -------
    dict
        Sequence-CNN configuration compatible with :func:`build_conv_layers`.
    """
    print("=== CNN Layer Configuration ===")
    available_activations = ["relu", "leaky_relu", "gelu", "tanh"]

    while True:
        try:
            alphabet_len = int(input("Enter alphabet length (num embeddings): "))
            if alphabet_len > 0:
                break
            else:
                print("Alphabet length must be a positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    while True:
        try:
            embedding_dim = int(input("Enter embedding dimension (input channels): "))
            if embedding_dim > 0:
                break
            else:
                print("Embedding dimensions must be a positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    padding_idx_input = input("Enter padding index [0]: ").strip()
    padding_idx = int(padding_idx_input) if padding_idx_input else 0

    while True:
        sizes_input = input("Enter conv layer output sizes (comma-separated, e.g. 256,128): ")
        try:
            sizes = list(map(int, sizes_input.split(",")))
            if all(s > 0 for s in sizes):
                break
            else:
                print("All convolutional layer sizes must be positive integers.")
        except Exception:
            print("Invalid input. Please enter comma-separated positive integers.")

    while True:
        try:
            kernel_size = int(input("Enter kernel size [5]: ") or 5)
            if kernel_size > 0:
                break
            else:
                print("Kernel size must be a positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    while True:
        try:
            stride = int(input("Enter stride [1]: ") or 1)
            if stride > 0:
                break
            else:
                print("Stride must be a be positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    while True:
        try:
            pool_size = int(input("Enter max pooling kernel size [2]: ") or 2)
            if pool_size >= 1:
                break
            else:
                print("Pooling kernel size must be a positive integer >= 1.")
        except ValueError:
            print("Invalid integer. Try again.")

    while True:
        print(f"Available activations: {available_activations}")
        activation = input("Select activation function [relu]: ").strip().lower() or "relu"
        if activation in available_activations:
            break
        print("Unsupported activation function. Try again.")

    while True:
        try:
            dropout = input("Enter dropout rate [0.1]: ").strip()
            dropout = float(dropout) if dropout else 0.1
            if 0.0 <= dropout < 1.0:
                break
            else:
                print("Dropout rate must be between 0 and 1.")
        except ValueError:
            print("Invalid float. Try again.")

    while True:
        try:
            batch_norm = input("Enter batch_norm [y/n]: ").strip()
            if batch_norm in ["y", "n"]:
                batch_norm = {"y": True, "n": False}.get(batch_norm)
                break
            else:
                print(f"Batch norm must be y/n. Try again.")
        except ValueError:
            "Invalid answer. Try again."

    print("\nCNN layer config complete.")
    return {
        "alphabet_len": alphabet_len,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "sizes": sizes,
        "kernel_size": kernel_size,
        "stride": stride,
        "pool_size": pool_size,
        "activation": activation,
        "dropout": dropout,
        "batch_norm": batch_norm
    }


class RNNModule(nn.Module):
    """
    Apply a packed recurrent layer and optional output transformations.

    Parameters
    ----------
    recurrent_layer: nn.Module
        LSTM, GRU, or RNN configured with `batch_first=True`.
    max_len: int
        Output sequence length passed to :func:`torch.nn.utils.rnn.pad_packed_sequence`.
    batch_norm: nn.Module or None
        Feature normalization layer applied to valid recurrent outputs. Default value is `None`.
    activation: nn.Module or None
        Activation layer applied after normalization. Default value is `None`.
    dropout: nn.Module or None
        Dropout layer applied after the activation. Default value is `None`.
    """

    def __init__(self, recurrent_layer: nn.Module, max_len: int, batch_norm: Optional[nn.Module] = None,
                 activation: Optional[nn.Module] = None, dropout: Optional[nn.Module] = None):
        super().__init__()
        self.recurrent_layer = recurrent_layer
        self.max_len = max_len
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, lengths):
        """
        Transform padded sequences while excluding padding from recurrence.

        Parameters
        ----------
        x: torch.Tensor
            Input embeddings with shape `(batch_size, sequence_length, features)`.
        lengths: torch.Tensor
            Unpadded sequence lengths with shape `(batch_size,)`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
        """

        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.recurrent_layer(packed_x)

        if self.batch_norm is not None:
            packed_out = packed_out._replace(data=self.batch_norm(packed_out.data))

        unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=self.max_len)

        if self.activation is not None:
            unpacked_out = self.activation(unpacked_out)
        if self.dropout is not None:
            unpacked_out = self.dropout(unpacked_out)

        return unpacked_out, lengths


def build_recurrent_layers(rnn_params: dict):
    """
    Build a token embedding and a single packed recurrent block.

    Parameters
    ----------
    rnn_params: dict
        Recurrent-network configuration. Required keys are:
        * `alphabet_len`
        * `embedding_dim`
        * `hidden_size`
        * `max_len`
        * `layer`: "lstm" | "gru" | "rnn"
        Optional keys are:
        * `padding_idx` defaults to 0
        * `batch_norm` defaults to True
        * `activation` defaults to "relu" 
        * `dropout` defaults to `0.0`

    Returns
    -------
    tuple[nn.ModuleList, nn.Embedding, int]
        A one-item list of recurrent blocks, token embedding, and hidden width.
    """

    alphabet_len = rnn_params["alphabet_len"]
    embedding_dim = rnn_params["embedding_dim"]
    padding_idx = rnn_params.get("padding_idx", 0)
    rnn_embedding = nn.Embedding(alphabet_len, embedding_dim, padding_idx=padding_idx)
    rnn_type = rnn_params.get("layer", "gru").lower()
    hidden_size = rnn_params.get("hidden_size", 256)
    activation = rnn_params.get("activation", "relu")
    dropout = rnn_params.get("dropout", 0.0)
    batch_norm = rnn_params.get("batch_norm", True)

    if rnn_type == "lstm":
        rnn_layer = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
    elif rnn_type == "gru":
        rnn_layer = nn.GRU(embedding_dim, hidden_size, batch_first=True)
    elif rnn_type == "rnn":
        rnn_layer = nn.RNN(embedding_dim, hidden_size, batch_first=True)
    else:
        raise ValueError(f"Unsupported RNN layer type: {rnn_type}")

    batch_norm_layer = nn.BatchNorm1d(hidden_size) if batch_norm else None
    activation_layer = get_activation_fn(activation) if activation else None
    dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else None
    rnn_blocks = nn.ModuleList([
        RNNModule(rnn_layer, rnn_params["max_len"], batch_norm_layer, activation_layer, dropout_layer)
    ])
    return rnn_blocks, rnn_embedding, hidden_size


def build_rnn_config():
    """
    Interactively collect a recurrent sequence-model configuration.

    Returns
    -------
    dict
        Recurrent-network configuration compatible with :func:`build_recurrent_layers`.
    """
    print("=== RNN Layer Configuration ===")
    while True:
        try:
            alphabet_len = int(input("Enter alphabet length (num embeddings): "))
            if alphabet_len > 0:
                break
            print("Alphabet length must be a positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")
    while True:
        try:
            embedding_dim = int(input("Enter embedding dimension (input channels): "))
            if embedding_dim > 0:
                break
            print("Embedding dimension must be positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")
    padding_idx_input = input("Enter padding index [0]: ").strip()
    padding_idx = int(padding_idx_input) if padding_idx_input else 0
    rnn_types = ["lstm", "gru", "rnn"]
    while True:
        print(f"Available RNN types: {rnn_types}")
        rnn_type = input("Select RNN type [gru]: ").strip().lower() or "gru"
        if rnn_type in rnn_types:
            break
        print("Invalid RNN type. Try again.")
    while True:
        try:
            hidden_size = int(input("Enter hidden size: "))
            if hidden_size > 0:
                break
            print("Hidden size must must be positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")
    while True:
        try:
            max_len = int(input("Enter max sequence length: "))
            if max_len > 0:
                break
            print("Maximum sequence length must be positive integer.")
        except ValueError:
            print("Invalid integer. Try again.")

    available_activations = ["relu", "leaky_relu", "gelu", "tanh", "none"]
    while True:
        print(f"Available activations: {available_activations}")
        activation = input("Select activation function [relu]: ").strip().lower() or "relu"
        if activation in available_activations:
            activation = None if activation == "none" else activation
            break
        print("Unsupported activation function. Try again.")

    while True:
        try:
            dropout = float(input("Enter dropout rate [0.0]: ").strip() or 0.0)
            if 0.0 <= dropout < 1.0:
                break
            print("Dropout rate must be between 0 and 1.")
        except ValueError:
            print("Invalid float. Try again.")

    while True:
        batch_norm = input("Enter batch_norm [y/n]: ").strip()
        if batch_norm in ["y", "n"]:
            batch_norm = {"y": True, "n": False}[batch_norm]
            break
        print("Batch norm must be y/n. Try again.")

    print("\nRNN layer config complete.")
    return {
        "alphabet_len": alphabet_len,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "layer": rnn_type,
        "hidden_size": hidden_size,
        "max_len": max_len,
        "activation": activation,
        "dropout": dropout,
        "batch_norm": batch_norm
    }


def build_hidden_layers(sizes: List[int], batch_norm: bool = True,
                        activation: Optional[str] = "relu", dropout: float = 0.0):
    """
    Build a fully connected network from successive feature dimensions.

    Each linear layer is Xavier-normal initialized and followed, when enabled,
    by batch normalization, an activation, and dropout.

    Parameters
    ----------
    sizes: list[int]
        Inclusive input, hidden, and output dimensions. Must contain at least two positive values.
    batch_norm: bool
        Whether to add batch normalization after each linear layer.
    activation: str or None
        Activation name accepted by :func:`get_activation_fn`; `None` omits
        activation layers. Default value is `"relu"`.
    dropout: float
        Dropout probability after each block. A value of `0.0` disables
        dropout. Default value is `0.0`.

    Returns
    -------
    tuple[nn.Sequential, int]
        Composed MLP and its final output dimension.
    """

    def init_linear(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.1, std=0.025)

    linear_layers = []

    linear_sizes = sizes.copy()
    lin_out_size = linear_sizes[-1]
    in_features = linear_sizes.pop(0)

    for out_features in linear_sizes:
        linear = nn.Linear(in_features, out_features)
        init_linear(linear)
        linear_layers.append(linear)

        if batch_norm:
            linear_layers.append(nn.BatchNorm1d(out_features))
        if activation is not None:
            linear_layers.append(get_activation_fn(activation))
        if dropout > 0:
            linear_layers.append(nn.Dropout(p=dropout))

        in_features = out_features

    return nn.Sequential(*linear_layers), lin_out_size


def build_hidden_config():
    """
    Interactively collect a fully connected network configuration.

    Returns
    -------
    dict
        MLP configuration compatible with :func:`build_hidden_layers`.
    """
    print("=== Linear Layer Configuration ===")

    while True:
        sizes_input = input("Enter linear layer sizes (comma-separated, with input and output sizes, e.g. 128,64,32): ")
        try:
            sizes = list(map(int, sizes_input.split(",")))
            if len(sizes) >= 2 and all(s > 0 for s in sizes):
                break
            else:
                print("Layer sizes must consist of at least two positive integers.")
        except Exception:
            print("Invalid input. Try again.")

    while True:
        try:
            batch_norm = input("Enter batch_norm [y/n]: ").strip()
            if batch_norm in ["y", "n"]:
                batch_norm = {"y": True, "n": False}.get(batch_norm)
                break
            else:
                print(f"Batch normalization must be y/n. Try again.")
        except ValueError:
            "Invalid answer. Try again."

    available_activations = ["relu", "leaky_relu", "gelu", "tanh", "none"]
    while True:
        print(f"Available activations: {available_activations}")
        activation = input("Select activation function [relu]: ").strip().lower() or "relu"
        if activation in available_activations:
            if activation == "none":
                activation = None
            break
        print("Unsupported activation function. Try again.")

    while True:
        try:
            dropout_input = input("Dropout rate [0.0]: ").strip()
            dropout = float(dropout_input) if dropout_input else 0.0
            if 0.0 <= dropout < 1.0:
                break
            else:
                print("Dropout rate must be between 0 and 1.")
        except ValueError:
            print("Invalid float. Try again.")

    print("\nLinear layer config complete.")
    return {
        "sizes": sizes,
        "batch_norm": batch_norm,
        "activation": activation,
        "dropout": dropout
    }
