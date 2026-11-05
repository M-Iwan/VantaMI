import os
import inspect
import datetime
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from sklearn.metrics import (confusion_matrix, roc_auc_score, r2_score, mean_absolute_error, root_mean_squared_error,
                             average_precision_score, brier_score_loss)

import torch
from torch import Tensor, nn

from vantami.ml.score import ece_score
from vantami.stats.utils import round_to_significant as rts


LossSummary = Dict[str, Any]
LossValue = Tuple[Tensor, Tensor]


class MMUnit(nn.Module, ABC):
    """
    Abstract base class for multitask deep-learning models.

    The class provides reusable training, evaluation, checkpointing, masked
    weighted-loss, and metric-aggregation functionality. Subclasses must
    implement :meth:`forward` and :meth:`score`.

    Parameters
    ----------
    device: str, default="cpu"
        Device on which model calculations are performed.
    num_task: int, default=1
        Number of prediction tasks.
    model_name: str, default="Unit"
        Model identifier used in checkpoint filenames.
    max_grad_norm: float, optional
        Maximum gradient norm used for gradient clipping. Gradient clipping is disabled when this is `None`.
    verbose: int, optional
        Set verbosity level. 0 = no logs, 1 = standard progress, 2 = debugging
    """

    def __init__(self, device: str = "cpu", num_task: int = 1, model_name: str = "Unit",
                 max_grad_norm: Optional[float] = None, verbose: int = 1) -> None:
        super().__init__()

        self.device = device
        self.num_task = num_task
        self.name = model_name
        self.model_name = model_name
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose

        self.loss_fn: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None

        self.logs = defaultdict(list)
        self.epoch = 1
        self.best_state_dict: Optional[Dict[str, Tensor]] = None
        self.best_epoch: Optional[int] = None
        self.hyperparameters: Dict[str, Any] = {}

    def log(self, message: str, level: int = 1):
        if self.verbose >= level:
            log_time = str(datetime.datetime.now()).split(".")[0]
            print(f"{log_time}: {message}")

    @abstractmethod
    def forward(self, batch_data: dict) -> Tensor:
        """
        Generate predictions for one batch.

        Parameters
        ----------
        batch_data : dict
            Mapping containing the model inputs.

        Returns
        -------
        torch.Tensor
            Model predictions with shape `(batch_size, num_task)`.
        """

    @staticmethod
    @abstractmethod
    def score(y_true: npt.NDArray, y_pred: npt.NDArray, y_wgts: npt.NDArray) -> Dict[str, float]:
        """
        Calculate task-specific metrics from target, prediction, and weight arrays.

        Parameters
        ----------
        y_true: numpy.ndarray
            Target values.
        y_pred: numpy.ndarray
            Predicted values.
        y_wgts: numpy.ndarray
            Observation weights.

        Returns
        -------
        dict
        """

    def grad_norm(self) -> float:
        """
        Calculate the L2 norm across all available parameter gradients.
        """
        squared_norm = 0.0

        for parameter in self.parameters():
            if parameter.grad is not None:
                parameter_norm = parameter.grad.detach().norm(2)
                squared_norm += parameter_norm.item() ** 2

        return squared_norm ** 0.5

    @staticmethod
    def _get_batch_group(batch_data: Any) -> Optional[Iterable[Any]]:
        """
        Retrieve optional group labels from a dict-like batch object.
        """
        if hasattr(batch_data, "get"):
            return batch_data.get("group")

        if isinstance(batch_data, dict):
            return batch_data.get("group")

        return None

    @staticmethod
    def _normalise_group_tags(group_values: Iterable[Any]) -> List[set[str]]:
        """
        Convert row-wise group annotations into cleaned sets of tag strings.
        """
        normalised_tags = []

        for row_value in group_values:
            row_tags = set()

            for value in np.asarray(row_value, dtype=object).ravel():
                if value is None:
                    continue

                tag = str(value).strip()
                if tag and tag.lower() != "nan":
                    row_tags.add(tag)

            normalised_tags.append(row_tags)

        return normalised_tags

    def _summarise_epoch(self, batch_losses: List[LossSummary], epoch_data: Dict[str, List[Tensor]]):
        """
        Aggregate losses and metrics accumulated over one epoch.
        """
        if not batch_losses:
            raise ValueError("The dataloader produced no batches.")

        cumulative_loss = reduce(self.combine_loss, batch_losses)
        normalised_loss = self.normalize_loss(cumulative_loss)
        epoch_metrics = self.mw_metrics(**self.pack_epoch_data(epoch_data))

        return normalised_loss, epoch_metrics

    def fit_epoch(self, dataloader: Iterable[dict]) -> Tuple[LossSummary, Dict[str, Any], List[float]]:
        """
        Train the model for one epoch.

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

        for batch_index, batch_data in enumerate(dataloader, start=1):
            self.log(f"Iterating over batch: {batch_index}", level=2)
            self.optimizer.zero_grad()

            y_pred = self(batch_data)
            y_true = batch_data["y_true"].to(self.device)
            y_wgts = batch_data["y_wgts"].to(self.device)

            self.log(f"Training batch {batch_index}: y_pred shape={y_pred.shape}, "
                     f"y_true shape={y_true.shape}, y_wgts shape={y_wgts.shape}.", level=2)

            if any([y_pred.shape == 0, y_true.shape==0, y_wgts.shape==0]):
                self.log(f"Zero size batch encountered", level=1)
                continue

            epoch_data["y_pred"].append(y_pred)
            epoch_data["y_true"].append(y_true)
            epoch_data["y_wgts"].append(y_wgts)

            batch_loss = self.loss(
                y_pred=y_pred,
                y_true=y_true,
                y_wgts=y_wgts,
                group=self._get_batch_group(batch_data),
            )

            total_loss, total_weight = batch_loss["Total"]
            total_loss.backward()

            batch_gradient_norm = self.grad_norm() / (total_weight.item() + 1e-6)
            gradient_norms.append(batch_gradient_norm)

            self.log(f"Training batch {batch_index}: loss sum={total_loss.item():.5f}, "
                     f"weight sum={total_weight.item():.5f}, gradient norm={batch_gradient_norm:.5f}.", level=2)

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    max_norm=self.max_grad_norm,
                )

            self.optimizer.step()
            batch_losses.append(batch_loss)

        if self.scheduler is not None:
            self.scheduler.step()
            self.log("Learning-rate scheduler step completed.", level=2)

        epoch_loss, epoch_metrics = self._summarise_epoch(batch_losses, epoch_data)
        return epoch_loss, epoch_metrics, gradient_norms

    def eval_epoch(self, dataloader: Iterable[dict]) -> Tuple[LossSummary, Dict[str, Any]]:
        """
        Evaluate the model.
        """
        self.eval()

        batch_losses: List[LossSummary] = []
        epoch_data = {"y_pred": [], "y_true": [], "y_wgts": []}

        with torch.no_grad():
            for batch_index, batch_data in enumerate(dataloader, start=1):
                y_pred = self(batch_data)
                y_true = batch_data["y_true"].to(self.device)
                y_wgts = batch_data["y_wgts"].to(self.device)

                self.log(f"Evaluation batch {batch_index}: y_pred shape={y_pred.shape}, "
                         f"y_true shape={y_true.shape}, y_wgts shape={y_wgts.shape}.", level=2)

                epoch_data["y_pred"].append(y_pred)
                epoch_data["y_true"].append(y_true)
                epoch_data["y_wgts"].append(y_wgts)

                batch_loss = self.loss(
                    y_pred=y_pred,
                    y_true=y_true,
                    y_wgts=y_wgts,
                    group=self._get_batch_group(batch_data),
                )

                total_loss, total_weight = batch_loss["Total"]
                self.log(f"Evaluation batch {batch_index}: loss sum={total_loss.item():.5f}, "
                         f"weight sum={total_weight.item():.5f}.", level=2)

                batch_losses.append(batch_loss)

        return self._summarise_epoch(batch_losses, epoch_data)

    def fit(self, dataloader: Iterable[dict], n_epochs: int = 4, early_stop: Optional[int] = 8,
            save_freq: Optional[int] = 8, save_dir: Optional[str] = None):
        """
        Train the model using training loss for early stopping.
        """
        if self.loss_fn is None or self.optimizer is None:
            raise ValueError("Set the loss function and optimizer before fitting the model.")

        self.log(f"Starting training for up to {n_epochs} epoch(s).", level=1)

        best_loss = float("inf")
        patience = deepcopy(early_stop) if early_stop is not None else None

        for epoch_number in range(1, n_epochs + 1):
            self.log(f"Starting training epoch {epoch_number}/{n_epochs}.", level=1)

            train_loss, train_metrics, train_grad_norms = self.fit_epoch(dataloader, )

            self.logs["train_loss"].append((epoch_number, train_loss))
            self.logs["train_metrics"].append((epoch_number, train_metrics))
            self.logs["train_grad_norms"].append((epoch_number, train_grad_norms))

            current_loss = train_loss["Total"]

            self.log(f"Epoch {epoch_number} train loss: {current_loss:.5f}", level=1)

            if early_stop is not None:
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience = deepcopy(early_stop)
                    self.set_best_model(epoch_number)

                    self.log(f"Epoch {epoch_number}: new best training loss: {best_loss:.5f}.", level=1)

                    if save_dir is not None:
                        save_path = os.path.join(save_dir, f"{self.model_name}_min.pth")
                        self.save(save_path)
                        self.log(f"Saved best checkpoint to {save_path}.", level=1)
                else:
                    patience -= 1
                    self.log(f"Epoch {epoch_number}: early-stopping patience remaining: {patience}.", level=1)

                if patience == 0:
                    self.log(f"Early stopping at epoch {epoch_number} with minimum loss {best_loss:.5f}", level=1)
                    self.get_best_model()
                    self.epoch = self.best_epoch
                    break

            if save_freq is not None and save_dir is not None and epoch_number % save_freq == 0:
                save_path = os.path.join(save_dir, f"{self.model_name}_{epoch_number}.pth")
                self.save(save_path)
                self.log(f"Saved checkpoint to {save_path}.", level=1)

            self.epoch += 1

        self.log("Training completed.", level=1)

    def fit_eval(self, train_dataloader: Iterable[dict], eval_dataloader: Iterable[dict], n_epochs: int = 4,
                 early_stop: Optional[int] = 4, save_freq: Optional[int] = 4, save_dir: Optional[str] = None):
        """
        Train and evaluate the model, using evaluation loss for early stopping.
        """
        if self.loss_fn is None or self.optimizer is None:
            raise ValueError("Set the loss function and optimizer before fitting the model.")

        self.log(f"Starting training and evaluation for up to {n_epochs} epoch(s).", level=1)

        best_loss = float("inf")
        patience = deepcopy(early_stop) if early_stop is not None else None

        for epoch_number in range(1, n_epochs + 1):
            self.log(f"Starting training and evaluation epoch {epoch_number}/{n_epochs}.", level=1)

            train_loss, train_metrics, train_grad_norms = self.fit_epoch(train_dataloader, )
            eval_loss, eval_metrics = self.eval_epoch(eval_dataloader, )

            self.logs["train_loss"].append((epoch_number, train_loss))
            self.logs["train_metrics"].append((epoch_number, train_metrics))
            self.logs["train_grad_norms"].append((epoch_number, train_grad_norms))
            self.logs["eval_loss"].append((epoch_number, eval_loss))
            self.logs["eval_metrics"].append((epoch_number, eval_metrics))

            current_loss = eval_loss["Total"]

            self.log(f"Epoch {epoch_number} train loss: {train_loss['Total']:.5f}" , level=1)
            self.log(f"Epoch {epoch_number} eval loss: {current_loss:.5f}", level=1)

            if early_stop is not None:
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience = deepcopy(early_stop)
                    self.set_best_model(epoch_number)

                    self.log(f"Epoch {epoch_number}: new best evaluation loss: {best_loss:.5f}.", level=1)

                    if save_dir is not None:
                        save_path = os.path.join(save_dir, f"{self.model_name}_min.pth")
                        self.save(save_path)
                        self.log(f"Saved best checkpoint to {save_path}.", level=1)
                else:
                    patience -= 1
                    self.log(f"Epoch {epoch_number}: early-stopping patience remaining: {patience}.", level=1)

                if patience == 0:
                    self.log(f"Early stopping at epoch {epoch_number} with minimum loss {best_loss:.5f}", level=1)
                    self.get_best_model()
                    self.epoch = self.best_epoch
                    break

            if save_freq is not None and save_dir is not None and epoch_number % save_freq == 0:
                save_path = os.path.join(save_dir, f"{self.model_name}_{epoch_number}.pth")
                self.save(save_path)
                self.log(f"Saved checkpoint to {save_path}.", level=1)

            self.epoch += 1

        self.log("Training and evaluation completed.", level=1)

    def predict(self, dataloader: Iterable[dict]) -> npt.NDArray:
        """
        Generate predictions for all batches in a dataloader.
        """
        self.eval()
        predictions = []

        with torch.no_grad():
            for batch_data in dataloader:
                predictions.append(self(batch_data))

        if not predictions:
            raise ValueError("The dataloader produced no batches.")

        return torch.cat(predictions, dim=0).detach().cpu().numpy()

    def loss(self, y_pred: Tensor, y_true: Tensor, y_wgts: Tensor, group: Optional[Iterable[Any]] = None) -> LossSummary:
        """
        Calculate a masked and weighted element-wise loss.

        Missing targets are identified by `NaN` values and excluded from all
        loss calculations. Group losses are optionally calculated for every
        group tag assigned to one or more rows in the batch.

        Parameters
        ----------
        y_pred: torch.Tensor
            Model predictions.
        y_true: torch.Tensor
            Target values, where missing observations are represented by NaN.
        y_wgts: torch.Tensor
            Per-observation loss weights.
        group: iterable, optional
            Row-aligned group annotations. Each row may contain one or more tags.

        Returns
        -------
        dict
            `Total` and `Task` entries contain `(loss_sum, weight_sum)`.
            When group annotations are supplied, `Group` maps each tag to the
            same tuple structure.
        """
        valid_target_mask = ~torch.isnan(y_true)
        safe_y_true = torch.nan_to_num(y_true, nan=0.0)

        elementwise_loss = self.loss_fn(y_pred, safe_y_true)
        masked_weights = y_wgts * valid_target_mask
        weighted_loss = elementwise_loss * masked_weights

        total_loss = weighted_loss.sum()
        total_weight = masked_weights.sum()

        task_loss = weighted_loss.sum(dim=0)
        task_weight = masked_weights.sum(dim=0)

        batch_loss: LossSummary = {
            "Total": (total_loss, total_weight),
            "Task": (task_loss, task_weight),
        }

        if group is None:
            return batch_loss

        group_tags = self._normalise_group_tags(group)
        if len(group_tags) != y_pred.shape[0]:
            return batch_loss

        unique_tags = set().union(*group_tags)
        if not unique_tags:
            return batch_loss

        group_losses: Dict[str, LossValue] = {}
        row_axes = (1,) * max(0, weighted_loss.ndim - 1)

        for tag in unique_tags:
            row_mask = torch.tensor(
                [tag in row_tags for row_tags in group_tags],
                device=y_pred.device, dtype=weighted_loss.dtype,
            ).view(y_pred.shape[0], *row_axes)

            group_weight = (masked_weights * row_mask).sum()
            if group_weight.item() > 0.0:
                group_losses[tag] = (
                    (weighted_loss * row_mask).sum(), group_weight,
                )

        if group_losses:
            batch_loss["Group"] = group_losses

        return batch_loss

    @staticmethod
    def combine_loss(accumulated_loss: LossSummary, batch_loss: LossSummary) -> LossSummary:
        """
        Combine loss totals from two batches or partial epochs.
        """
        total_loss = accumulated_loss["Total"][0] + batch_loss["Total"][0]
        total_weight = accumulated_loss["Total"][1] + batch_loss["Total"][1]

        task_loss = accumulated_loss["Task"][0] + batch_loss["Task"][0]
        task_weight = accumulated_loss["Task"][1] + batch_loss["Task"][1]

        combined_loss: LossSummary = {
            "Total": (total_loss, total_weight),
            "Task": (task_loss, task_weight),
        }

        accumulated_groups = accumulated_loss.get("Group") or {}
        batch_groups = batch_loss.get("Group") or {}

        if accumulated_groups or batch_groups:
            group_losses: Dict[str, LossValue] = {}

            for tag in set(accumulated_groups) | set(batch_groups):
                loss_sum = torch.zeros((), device=total_loss.device, dtype=total_loss.dtype)
                weight_sum = torch.zeros((), device=total_weight.device, dtype=total_weight.dtype)

                if tag in accumulated_groups:
                    loss_sum += accumulated_groups[tag][0]
                    weight_sum += accumulated_groups[tag][1]

                if tag in batch_groups:
                    loss_sum += batch_groups[tag][0]
                    weight_sum += batch_groups[tag][1]

                group_losses[tag] = (loss_sum, weight_sum)

            combined_loss["Group"] = group_losses

        return combined_loss

    @staticmethod
    def normalize_loss(weighted_loss: LossSummary) -> LossSummary:
        """
        Convert weighted loss totals into normalized loss values.
        """
        total_loss, total_weight = weighted_loss["Total"]
        task_loss, task_weight = weighted_loss["Task"]

        normalized_total = (
            total_loss / total_weight if total_weight.item() > 0.0
            else torch.zeros((), device=total_loss.device, dtype=total_loss.dtype)
        )

        normalized_task = torch.where(
            task_weight > 0,
            task_loss / task_weight.clamp_min(torch.finfo(task_weight.dtype).eps),
            torch.zeros_like(task_loss),
        )

        normalized_loss: LossSummary = {
            "Total": round(normalized_total.item(), 5),
            "Task": np.round(normalized_task.detach().cpu().numpy(), 5),
        }

        group_losses = weighted_loss.get("Group")
        if group_losses:
            normalized_loss["Group"] = {
                tag: round(loss_sum.div(weight_sum).item(), 5) if weight_sum.item() > 0.0 else 0.0
                for tag, (loss_sum, weight_sum) in group_losses.items()
            }

        return normalized_loss

    def mw_metrics(self, y_true: Tensor, y_pred: Tensor, y_wgts: Tensor) -> Dict[str, Any]:
        """
        Calculate overall and task-level weighted metrics for an epoch.
        """
        y_true_array = y_true.detach().cpu().numpy()
        y_pred_array = y_pred.detach().cpu().numpy()
        y_weight_array = y_wgts.detach().cpu().numpy()

        total_metrics = self.score(
            y_true=y_true_array.flatten(),
            y_pred=y_pred_array.flatten(),
            y_wgts=y_weight_array.flatten(),
        )

        task_metrics = {}
        for task_index in range(self.num_task):
            task_metrics[f"Task_{task_index}"] = self.score(
                y_true=y_true_array[:, task_index].flatten(),
                y_pred=y_pred_array[:, task_index].flatten(),
                y_wgts=y_weight_array[:, task_index].flatten(),
            )

        return {"Total": total_metrics, "Task": task_metrics}

    @staticmethod
    def pack_epoch_data(epoch_data: Dict[str, List[Tensor]]) -> Dict[str, Tensor]:
        """
        Concatenate batch-level tensors collected during one epoch.
        """
        return {
            key: torch.cat(values, dim=0)
            for key, values in epoch_data.items()
        }

    def set_loss_function(self, loss_fn: Any = torch.nn.BCEWithLogitsLoss, loss_params: Optional[dict] = None) -> None:
        """
        Instantiate an element-wise loss function.

        The loss function must use `reduction="none"`
        """
        if not callable(loss_fn):
            raise ValueError("Loss function must be callable.")

        self.loss_fn = loss_fn(**loss_params) if loss_params else loss_fn()

        if getattr(self.loss_fn, "reduction", None) != "none":
            raise ValueError('Loss function must be initialized with reduction="none".')

    def set_optimizer(self, optimizer_class: Any = torch.optim.AdamW, optimizer_params: Optional[dict] = None) -> None:
        """
        Instantiate the model optimizer.
        """
        if not callable(optimizer_class):
            raise ValueError("Optimizer must be callable.")

        self.optimizer = optimizer_class(self.parameters(), **(optimizer_params or {}))

    def set_scheduler(self, scheduler_class: Any, scheduler_params: Optional[dict] = None) -> None:
        """
        Instantiate a learning-rate scheduler for the configured optimizer.
        """
        if self.optimizer is None:
            raise ValueError("Set the optimizer before configuring a scheduler.")

        if not callable(scheduler_class):
            raise ValueError("Scheduler must be callable.")

        self.scheduler = scheduler_class(
            self.optimizer,
            **(scheduler_params or {}),
        )

    @staticmethod
    def get_hyperparameters() -> Dict[str, Any]:
        """
        Capture constructor-local values when called from a subclass initializer.
        """
        frame = inspect.currentframe()
        try:
            _, _, _, local_variables = inspect.getargvalues(frame.f_back)
            return {
                name: value for name, value in local_variables.items()
                if name != "self"
            }
        finally:
            del frame

    def set_best_model(self, epoch_number: int) -> None:
        """
        Store the current model state as the best observed state.
        """
        self.best_state_dict = deepcopy(self.state_dict())
        self.best_epoch = epoch_number

    def get_best_model(self) -> None:
        """
        Restore the previously stored best model state.
        """
        if self.best_state_dict is None:
            raise RuntimeError("Best model has not been saved yet.")

        self.load_state_dict(self.best_state_dict)

    def save(self, path: str) -> None:
        """
        Save model, optimizer, scheduler, and hyperparameter states.
        """
        extension = os.path.splitext(path)[1].lower()
        if extension not in {".pt", ".pth"}:
            raise ValueError(f"Unsupported file extension: {extension}")

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "hyperparameters": self.hyperparameters,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        Load available model, optimizer, and scheduler state from a checkpoint.
        """
        extension = os.path.splitext(path)[1].lower()
        if extension not in {".pt", ".pth"}:
            raise ValueError(f"Unsupported file extension: {extension}")

        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if self.optimizer is not None and optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        self.hyperparameters = checkpoint.get("hyperparameters", {})

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """
        Set NumPy and PyTorch random seeds.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)


class MMClassificationUnit(MMUnit):
    """
    Abstract base class for binary classification multitask deep-learning models.

    The class provides reusable training, evaluation, checkpointing, masked weighted-loss,
    and metric-aggregation functionality. Subclasses must implement :meth:`forward`.

    Parameters
    ----------
    device: str, default="cpu"
        Device on which model calculations are performed.
    num_task: int, default=1
        Number of prediction tasks.
    model_name: str, default="Unit"
        Model identifier used in checkpoint filenames.
    max_grad_norm: float, optional
        Maximum gradient norm used for gradient clipping. Gradient clipping is disabled when this is `None`.
    verbose: int, optional
        Set verbosity level. 0 = no logs, 1 = standard progress, 2 = debugging
    """

    def __init__(self, device: str = "cpu", num_task: int = 1, model_name: str = "ClassificationUnit",
                 max_grad_norm: Optional[float] = None, verbose: int = 1) -> None:
        super().__init__(device=device, num_task=num_task, model_name=model_name, max_grad_norm=max_grad_norm,
                         verbose=verbose)

    @abstractmethod
    def forward(self, batch_data: dict) -> Tensor:
        """
        Generate predictions for one batch.

        Parameters
        ----------
        batch_data : dict
            Mapping containing the model inputs.

        Returns
        -------
        torch.Tensor
            Model predictions with shape `(batch_size, num_task)`.
        """

    @staticmethod
    def score(y_true: npt.NDArray, y_pred: npt.NDArray, y_wgts: Optional[npt.NDArray] = None,
              y_score: Optional[npt.NDArray] = None):
        """
        Calculate classification performance metrics from known labels and predictions.

        Parameters
        ----------
        y_true: npt.NDArray
            One-dimensional array containing true binary target labels.
        y_pred: npt.NDArray
            One-dimensional array containing predicted binary target labels.
        y_wgts: Optional[npt.NDArray]
            One-dimensional array of non-negative sample weights.
        y_score: npt.NDArray or None
            One-dimensional array containing positive-class probabilities or scores.

        Returns
        -------
        dict
        """

        def safe_div(numerator, denominator, default_=0.0):
            return numerator / denominator if denominator != 0 else default_

        conf_mat = confusion_matrix(y_true, y_pred, sample_weight=y_wgts)

        TN, FP, FN, TP = conf_mat.ravel()

        rec = safe_div(TP, TP + FN)
        spec = safe_div(TN, TN + FP)

        metrics = {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'Accuracy': rts(safe_div(TP + TN, TP + TN + FP + FN), 5),
            'Recall': rts(rec, 5),
            'Specificity': rts(spec, 5),
            'Precision': rts(safe_div(TP, TP + FP), 5),
            'Balanced Accuracy': rts((rec + spec) / 2, 5),
            'GeomRS': rts(np.sqrt(rec * spec), 5),
            'HarmRS': rts(safe_div(2 * rec * spec, rec + spec), 5),
            'F1 Score': rts(safe_div(2 * TP, 2 * TP + FP + FN), 5),
            'MCC': rts(safe_div((TP * TN) - (FP * FN), np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))), 5),
        }

        if y_score is not None:
            metrics.update({
                "ROC AUC": rts(roc_auc_score(y_true, y_score, sample_weight=y_wgts), 5),
                'PRC AUC': rts(average_precision_score(y_true=y_true, y_score=y_score, sample_weight=y_wgts), 5),
                'Brier': rts(brier_score_loss(y_true=y_true, y_proba=y_score, sample_weight=y_wgts), 5),
                'ECE': rts(ece_score(y_true=y_true, y_score=y_score, sample_weight=y_wgts), 5)
            })

        return metrics


class MMRegressionUnit(MMUnit):
    """
    Abstract base class for regression multitask deep-learning models.

    The class provides reusable training, evaluation, checkpointing, masked weighted-loss,
    and metric-aggregation functionality. Subclasses must implement :meth:`forward`.

    Parameters
    ----------
    device: str, default="cpu"
        Device on which model calculations are performed.
    num_task: int, default=1
        Number of prediction tasks.
    model_name: str, default="Unit"
        Model identifier used in checkpoint filenames.
    max_grad_norm: float, optional
        Maximum gradient norm used for gradient clipping. Gradient clipping is disabled when this is `None`.
    verbose: int, optional
        Set verbosity level. 0 = no logs, 1 = standard progress, 2 = debugging
    """

    def __init__(self, device: str = "cpu", num_task: int = 1, model_name: str = "RegressionUnit",
                 max_grad_norm: Optional[float] = None, verbose: int = 1) -> None:
        super().__init__(device=device, num_task=num_task, model_name=model_name, max_grad_norm=max_grad_norm,
                         verbose=verbose)

    @abstractmethod
    def forward(self, batch_data: dict) -> Tensor:
        """
        Generate predictions for one batch.

        Parameters
        ----------
        batch_data : dict
            Mapping containing the model inputs.

        Returns
        -------
        torch.Tensor
            Model predictions with shape `(batch_size, num_task)`.
        """

    @staticmethod
    def score(y_true: npt.NDArray, y_pred: npt.NDArray, y_wgts: npt.NDArray):
        """
        Calculate regression performance metrics from known and predicted target values.

        Parameters
        ----------
        y_true: npt.NDArray
            One-dimensional array containing true target values.
        y_pred: npt.NDArray
            One-dimensional array containing predicted target values.
        y_wgts: npt.NDArray
            One-dimensional array of non-negative sample weights.

        Returns
        -------
        dict
        """

        metrics = {
            'R2': rts(r2_score(y_true=y_true, y_pred=y_pred, sample_weight=y_wgts), 5),
            'MAE': rts(mean_absolute_error(y_true=y_true, y_pred=y_pred, sample_weight=y_wgts), 5),
            'RMSE': rts(root_mean_squared_error(y_true=y_true, y_pred=y_pred, sample_weight=y_wgts), 5)
        }

        return metrics