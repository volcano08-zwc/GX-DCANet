"""Leakage-controlled training utilities for the 26BCI V121 experiment."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from model.registry import apply_model_constraints
from utils import build_seed_generator


class Dataset26BCI(Dataset):
    def __init__(self, main, aligned_main, power, labels, weights=None, source_indices=None):
        n = len(labels)
        if main.shape != (n, 1, 16, 500) or aligned_main.shape != main.shape:
            raise ValueError(f"Unexpected 26BCI EEG shapes {main.shape}/{aligned_main.shape}.")
        if power.shape != (n, 96) or np.asarray(labels).shape != (n,):
            raise ValueError(f"Unexpected 26BCI power/label shapes {power.shape}/{np.asarray(labels).shape}.")
        if weights is None:
            weights = np.ones(n, dtype=np.float32)
        if source_indices is None:
            source_indices = np.arange(n)
        self.main = torch.from_numpy(np.ascontiguousarray(main, dtype=np.float32))
        self.aligned = torch.from_numpy(np.ascontiguousarray(aligned_main, dtype=np.float32))
        self.power = torch.from_numpy(np.ascontiguousarray(power, dtype=np.float32))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))
        self.weights = torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float32))
        self.source_indices = torch.from_numpy(np.ascontiguousarray(source_indices, dtype=np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.main[index], self.aligned[index], self.power[index],
            self.labels[index], self.weights[index], self.source_indices[index],
        )


class MergedLastBatchSampler(Sampler[list[int]]):
    """Shuffle batches and merge a final singleton to keep BatchNorm valid."""

    def __init__(self, length: int, batch_size: int, seed: int, shuffle: bool):
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def __iter__(self):
        indices = np.arange(self.length)
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        self.epoch += 1
        batches = [indices[i:i + self.batch_size].tolist() for i in range(0, self.length, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        count = (self.length + self.batch_size - 1) // self.batch_size
        return count - int(count > 1 and self.length % self.batch_size == 1)


def _metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "confusion_matrix": matrix.tolist(),
    }
    result["auc"] = (
        float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")
    )
    return result


class Protocol26BCI:
    """Protected-residual training with validation-only checkpoint selection."""

    def __init__(self, model, config: dict, device: torch.device, output_dir: Path, *, lr: float, weight_decay: float):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def _loader(self, dataset: Dataset26BCI, training: bool) -> DataLoader:
        sampler = MergedLastBatchSampler(
            len(dataset), int(self.config["batch_size"]),
            int(self.config["random_seed"]), training,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=int(self.config.get("num_workers", 0)),
            generator=build_seed_generator(int(self.config["random_seed"])),
        )

    def _run(self, dataset: Dataset26BCI, training: bool) -> dict:
        self.model.train(training)
        loader = self._loader(dataset, training)
        total_loss = total_objective = 0.0
        labels_all, predictions_all, probabilities_all, indices_all = [], [], [], []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for main, aligned, power, labels, weights, indices in loader:
                main = main.to(self.device)
                aligned = aligned.to(self.device)
                power = power.to(self.device)
                labels = labels.to(self.device)
                weights = weights.to(self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits, aux = self.model(
                    main, {"aligned_main": aligned, "power": power}, return_aux=True
                )
                main_losses = self.loss(aux["main_logits"], labels)
                side_losses = self.loss(aux["main_logits"].detach() + aux["side_logits"], labels)
                objective = ((main_losses + side_losses) * weights).sum() / weights.sum()
                combined = self.loss(logits, labels).mean()
                if not torch.isfinite(objective):
                    raise FloatingPointError("Non-finite 26BCI training objective.")
                if training:
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.optimizer.step()
                    apply_model_constraints(self.model)
                probability = torch.softmax(logits, dim=1)[:, 1]
                predicted = logits.argmax(dim=1)
                total_loss += float(combined.item()) * len(labels)
                total_objective += float(objective.item()) * len(labels)
                labels_all.extend(labels.detach().cpu().tolist())
                predictions_all.extend(predicted.detach().cpu().tolist())
                probabilities_all.extend(probability.detach().cpu().tolist())
                indices_all.extend(indices.tolist())
        labels_np = np.asarray(labels_all, dtype=np.int64)
        predictions_np = np.asarray(predictions_all, dtype=np.int64)
        probabilities_np = np.asarray(probabilities_all, dtype=np.float64)
        result = _metrics(labels_np, predictions_np, probabilities_np)
        result.update({
            "loss": total_loss / len(dataset),
            "objective_loss": total_objective / len(dataset),
            "labels": labels_np,
            "predictions": predictions_np,
            "probabilities": probabilities_np,
            "source_indices": np.asarray(indices_all, dtype=np.int64),
        })
        return result

    @staticmethod
    def _selection_key(metrics: dict) -> tuple[float, float, float, float]:
        return (
            metrics["balanced_accuracy"], metrics["macro_f1"],
            metrics["accuracy"], -metrics["loss"],
        )

    @staticmethod
    def _serializable(metrics: dict) -> dict:
        return {
            key: value for key, value in metrics.items()
            if key not in {"labels", "predictions", "probabilities", "source_indices"}
        }

    def fit(
        self,
        train_set: Dataset26BCI,
        validation_set: Dataset26BCI,
        *,
        max_epochs: int,
        checkpoint_name: str = "best.pth",
    ) -> dict:
        best_key = (-np.inf, -np.inf, -np.inf, -np.inf)
        best_state = None
        best_metrics = None
        best_epoch = 0
        history = []
        for epoch in range(1, int(max_epochs) + 1):
            self.model.augmentation.probability = 0.5 * (epoch - 1) / max(1, max_epochs - 1)
            train_metrics = self._run(train_set, True)
            validation_metrics = self._run(validation_set, False)
            key = self._selection_key(validation_metrics)
            history.append({
                "epoch": epoch,
                "train": self._serializable(train_metrics),
                "validation": self._serializable(validation_metrics),
            })
            if key > best_key:
                best_key = key
                best_epoch = epoch
                best_metrics = self._serializable(validation_metrics)
                best_state = copy.deepcopy(self.model.state_dict())
            print(
                f"[26BCI] epoch={epoch:03d}/{max_epochs} "
                f"train_BA={train_metrics['balanced_accuracy']:.4f} "
                f"val_BA={validation_metrics['balanced_accuracy']:.4f} "
                f"best={best_key[0]:.4f}", flush=True,
            )
        if best_state is None:
            raise RuntimeError("Training produced no checkpoint.")
        self.model.load_state_dict(best_state)
        torch.save(best_state, self.output_dir / checkpoint_name)
        (self.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {"best_epoch": best_epoch, "validation": best_metrics, "epochs_ran": len(history)}

    def fit_fixed(self, train_set: Dataset26BCI, *, epochs: int, checkpoint_name: str) -> dict:
        history = []
        for epoch in range(1, int(epochs) + 1):
            self.model.augmentation.probability = 0.5 * (epoch - 1) / max(1, epochs - 1)
            metrics = self._run(train_set, True)
            history.append({"epoch": epoch, "train": self._serializable(metrics)})
            print(
                f"[26BCI/refit] epoch={epoch:03d}/{epochs} "
                f"train_BA={metrics['balanced_accuracy']:.4f}", flush=True,
            )
        torch.save(self.model.state_dict(), self.output_dir / checkpoint_name)
        return {"epochs": int(epochs), "last_train": history[-1]["train"]}

    def evaluate(self, dataset: Dataset26BCI) -> dict:
        return self._run(dataset, False)


__all__ = ["Dataset26BCI", "Protocol26BCI"]
