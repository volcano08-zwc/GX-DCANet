"""Training protocol for LYNet's protected RUN-EA residual."""

from __future__ import annotations

import csv
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from model.registry import apply_model_constraints
from utils import build_seed_generator, seed_worker


class LYNetDataset(Dataset):
    """Tensor dataset for the main EEG view and 528-D side view."""

    def __init__(
        self,
        main: np.ndarray,
        aligned_main: np.ndarray,
        power: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        expected_main = (1, 22, 1000)
        if main.ndim != 4 or tuple(main.shape[1:]) != expected_main:
            raise ValueError(f"Unexpected LYNet raw main shape {main.shape}.")
        if aligned_main.shape != main.shape:
            raise ValueError(
                f"Unexpected LYNet aligned main shape {aligned_main.shape}."
            )
        if power.shape != (len(main), 528) or labels.shape != (len(main),):
            raise ValueError(f"Unexpected LYNet power/label shape {power.shape}/{labels.shape}.")
        self.main = torch.from_numpy(np.ascontiguousarray(main, dtype=np.float32))
        self.aligned_main = torch.from_numpy(
            np.ascontiguousarray(aligned_main, dtype=np.float32)
        )
        self.power = torch.from_numpy(np.ascontiguousarray(power, dtype=np.float32))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.main[index],
            self.aligned_main[index],
            self.power[index],
            self.labels[index],
        )


def _safe_kappa(labels: list[int], predictions: list[int]) -> float:
    value = float(cohen_kappa_score(labels, predictions))
    return value if np.isfinite(value) else float("nan")


class LYNetProtocol:
    """AxxT training, AxxE every epoch, exploratory Best-E checkpointing.

    The side objective uses ``main_logits.detach() + side_logits``.  This is a
    functional part of LYNet: gradients from the residual objective cannot
    update the main CX classifier.
    """

    def __init__(self, net, config, optimizer, loss_func, result_savepath, device):
        self.net = net
        self.config = config
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.result_savepath = Path(result_savepath)
        self.device = device

    def _build_dataloaders(self, train_dataset, eval_dataset):
        worker_fn = partial(seed_worker, base_seed=int(self.config["random_seed"]))
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=int(self.config["batch_size"]),
            num_workers=int(self.config["num_workers"]),
            worker_init_fn=worker_fn,
            generator=build_seed_generator(int(self.config["random_seed"])),
        )
        eval_loader = DataLoader(
            eval_dataset,
            shuffle=False,
            batch_size=int(self.config["batch_size"]),
            num_workers=int(self.config["num_workers"]),
            worker_init_fn=worker_fn,
        )
        return train_loader, eval_loader

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, object]:
        self.net.train(training)
        totals = {"combined": 0.0, "objective": 0.0, "main": 0.0, "side": 0.0}
        main_square = side_square = 0.0
        main_count = side_count = 0
        labels_all: list[int] = []
        predictions: list[int] = []
        main_predictions: list[int] = []
        correct = main_correct = corrected = harmed = batches = 0
        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for main, aligned_main, power, labels in loader:
                main = main.to(self.device, dtype=torch.float32)
                aligned_main = aligned_main.to(self.device, dtype=torch.float32)
                power = power.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                logits, aux = self.net(
                    main,
                    {"aligned_main": aligned_main, "power": power},
                    return_aux=True,
                )
                main_logits = aux["main_logits"]
                side_logits = aux["side_logits"]
                main_loss = self.loss_func(main_logits, labels)
                side_loss = self.loss_func(main_logits.detach() + side_logits, labels)
                objective = main_loss + side_loss
                combined_loss = self.loss_func(logits, labels)
                if not torch.isfinite(objective):
                    raise FloatingPointError("Non-finite LYNet objective.")

                if training:
                    objective.backward()
                    bad_gradients = [
                        name
                        for name, parameter in self.net.named_parameters()
                        if parameter.grad is not None
                        and not torch.isfinite(parameter.grad).all()
                    ]
                    if bad_gradients:
                        raise FloatingPointError(f"Non-finite LYNet gradients: {bad_gradients}.")
                    self.optimizer.step()
                    apply_model_constraints(self.net)

                predicted = logits.argmax(dim=1)
                main_predicted = main_logits.argmax(dim=1)
                totals["combined"] += float(combined_loss.item())
                totals["objective"] += float(objective.item())
                totals["main"] += float(main_loss.item())
                totals["side"] += float(side_loss.item())
                main_square += float(main_logits.square().sum().item())
                side_square += float(side_logits.square().sum().item())
                main_count += main_logits.numel()
                side_count += side_logits.numel()
                correct += int((predicted == labels).sum().item())
                main_correct += int((main_predicted == labels).sum().item())
                corrected += int(((main_predicted != labels) & (predicted == labels)).sum().item())
                harmed += int(((main_predicted == labels) & (predicted != labels)).sum().item())
                labels_all.extend(int(value) for value in labels.detach().cpu().tolist())
                predictions.extend(int(value) for value in predicted.detach().cpu().tolist())
                main_predictions.extend(
                    int(value) for value in main_predicted.detach().cpu().tolist()
                )
                batches += 1

        if batches == 0 or not labels_all:
            raise RuntimeError("LYNet protocol processed no batches.")
        main_rms = float(np.sqrt(main_square / main_count))
        side_rms = float(np.sqrt(side_square / side_count))
        return {
            "loss": totals["combined"] / batches,
            "objective_loss": totals["objective"] / batches,
            "main_loss": totals["main"] / batches,
            "side_loss": totals["side"] / batches,
            "accuracy": correct / len(labels_all),
            "main_accuracy": main_correct / len(labels_all),
            "kappa": _safe_kappa(labels_all, predictions),
            "macro_f1": float(f1_score(labels_all, predictions, average="macro")),
            "side_corrected_count": corrected,
            "side_harmed_count": harmed,
            "main_logit_rms": main_rms,
            "side_logit_rms": side_rms,
            "side_main_logit_rms_ratio": side_rms / (main_rms + 1e-12),
            "confusion_matrix": confusion_matrix(
                labels_all,
                predictions,
                labels=[0, 1, 2, 3],
            ),
        }

    def _plot_history(self, rows: list[dict[str, object]]) -> None:
        epochs = [int(row["epoch"]) for row in rows]
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, [row["train_loss"] for row in rows], label="Training loss")
        plt.plot(epochs, [row["eval_loss"] for row in rows], label="Evaluation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy")
        plt.title("LYNet training and evaluation loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(self.result_savepath / "loss_epoch.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, [row["train_accuracy"] for row in rows], label="Training accuracy")
        plt.plot(epochs, [row["eval_accuracy"] for row in rows], label="Evaluation accuracy")
        plt.plot(
            epochs,
            [row["eval_main_accuracy"] for row in rows],
            label="Evaluation main-only accuracy",
            linestyle="--",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("LYNet protected residual diagnostics")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(self.result_savepath / "accuracy_epoch.png")
        plt.close()

    def train_test(self, train_dataset, eval_dataset, subject_id: str) -> dict[str, object]:
        self.result_savepath.mkdir(parents=True, exist_ok=True)
        train_loader, eval_loader = self._build_dataloaders(train_dataset, eval_dataset)
        diagnostics_path = self.result_savepath / "diagnostics.csv"
        fieldnames = (
            "epoch",
            "train_seconds",
            "eval_seconds",
            "train_loss",
            "train_objective_loss",
            "train_accuracy",
            "train_main_accuracy",
            "eval_loss",
            "eval_accuracy",
            "eval_main_accuracy",
            "eval_kappa",
            "eval_macro_f1",
            "main_logit_rms",
            "side_logit_rms",
            "side_main_logit_rms_ratio",
            "side_corrected_count",
            "side_harmed_count",
        )
        rows: list[dict[str, object]] = []
        best_train_accuracy = -1.0
        best_train_kappa = float("nan")
        best_eval_accuracy = -1.0
        best_eval_metrics: dict[str, object] | None = None
        best_epoch = 0

        with diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for epoch in range(1, int(self.config["epochs"]) + 1):
                progress = (epoch - 1) / max(1, int(self.config["epochs"]) - 1)
                self.net.augmentation.probability = 0.50 * progress
                started = perf_counter()
                train_metrics = self._run_epoch(train_loader, training=True)
                train_seconds = perf_counter() - started
                started = perf_counter()
                eval_metrics = self._run_epoch(eval_loader, training=False)
                eval_seconds = perf_counter() - started
                row = {
                    "epoch": epoch,
                    "train_seconds": train_seconds,
                    "eval_seconds": eval_seconds,
                    "train_loss": train_metrics["loss"],
                    "train_objective_loss": train_metrics["objective_loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_main_accuracy": train_metrics["main_accuracy"],
                    "eval_loss": eval_metrics["loss"],
                    "eval_accuracy": eval_metrics["accuracy"],
                    "eval_main_accuracy": eval_metrics["main_accuracy"],
                    "eval_kappa": eval_metrics["kappa"],
                    "eval_macro_f1": eval_metrics["macro_f1"],
                    "main_logit_rms": eval_metrics["main_logit_rms"],
                    "side_logit_rms": eval_metrics["side_logit_rms"],
                    "side_main_logit_rms_ratio": eval_metrics[
                        "side_main_logit_rms_ratio"
                    ],
                    "side_corrected_count": eval_metrics["side_corrected_count"],
                    "side_harmed_count": eval_metrics["side_harmed_count"],
                }
                writer.writerow(row)
                handle.flush()
                rows.append(row)

                if float(train_metrics["accuracy"]) > best_train_accuracy:
                    best_train_accuracy = float(train_metrics["accuracy"])
                    best_train_kappa = float(train_metrics["kappa"])
                if float(eval_metrics["accuracy"]) > best_eval_accuracy:
                    best_eval_accuracy = float(eval_metrics["accuracy"])
                    best_eval_metrics = eval_metrics
                    best_epoch = epoch
                    torch.save(
                        self.net.state_dict(),
                        self.result_savepath / f"model_{subject_id}_best.pth",
                    )
                    np.savetxt(
                        self.result_savepath / "best_confusion_matrix.csv",
                        eval_metrics["confusion_matrix"],
                        delimiter=",",
                        fmt="%d",
                    )
                print(
                    f"[LYNet/RUN-EA][{subject_id}] epoch={epoch:03d}/"
                    f"{int(self.config['epochs'])} "
                    f"train={100 * float(train_metrics['accuracy']):.2f}% "
                    f"eval={100 * float(eval_metrics['accuracy']):.2f}% "
                    f"best={100 * best_eval_accuracy:.2f}%",
                    flush=True,
                )

        if best_eval_metrics is None:
            raise RuntimeError("LYNet did not produce evaluation metrics.")
        torch.save(
            self.net.state_dict(),
            self.result_savepath / f"model_{subject_id}_final.pth",
        )
        metrics: dict[str, object] = {
            # Common names retain compatibility with train_2a.write_summary.
            "best_train_acc": best_train_accuracy,
            "best_train_kappa": best_train_kappa,
            "best_test_acc": best_eval_accuracy,
            "best_test_kappa": float(best_eval_metrics["kappa"]),
            "best_eval_macro_f1": float(best_eval_metrics["macro_f1"]),
            "best_epoch": best_epoch,
            "best_side_corrected_count": int(best_eval_metrics["side_corrected_count"]),
            "best_side_harmed_count": int(best_eval_metrics["side_harmed_count"]),
            "best_side_main_logit_rms_ratio": float(
                best_eval_metrics["side_main_logit_rms_ratio"]
            ),
            "protocol": (
                "AxxT train; AxxE evaluated every epoch; exploratory Best-E; "
                "offline-transductive run-baseline reference"
            ),
        }
        with (self.result_savepath / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        with (self.result_savepath / "test_accuracy_results.txt").open(
            "w", encoding="utf-8"
        ) as handle:
            handle.write(f"Best Training Accuracy: {100 * best_train_accuracy:.9f}%\n")
            handle.write(f"Best Training Kappa Score: {best_train_kappa:.9f}\n")
            handle.write(f"Best Evaluation Accuracy: {100 * best_eval_accuracy:.9f}%\n")
            handle.write(
                f"Best Evaluation Kappa Score: {float(best_eval_metrics['kappa']):.9f}\n"
            )
            handle.write(f"Best Epoch: {best_epoch}\n")
            handle.write(f"Seed: {int(self.config['random_seed'])}\n")
            handle.write("Protocol: offline-transductive RUN-EA / exploratory Best-E\n")
        self._plot_history(rows)
        return metrics


__all__ = ["LYNetDataset", "LYNetProtocol"]
