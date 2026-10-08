"""FP32 execution optimization; training/evaluation semantics remain V121."""

import json
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from model.registry import apply_model_constraints
from protocol.lynet_protocol import LYNetProtocol, _safe_kappa


class FastLYNetProtocol(LYNetProtocol):
    def _build_dataloaders(self, train_dataset, eval_dataset):
        if int(self.config["num_workers"]) != 0:
            raise ValueError("Resident GPU datasets require num_workers=0.")
        started = perf_counter()
        # Same DataLoader sampler/generator and iterator RNG consumption as V121.
        loaders = super()._build_dataloaders(
            list(range(len(train_dataset))), list(range(len(eval_dataset)))
        )
        self.resident = {}
        for loader, dataset, training in zip(
            loaders, (train_dataset, eval_dataset), (True, False)
        ):
            tensors = tuple(value.to(self.device) for value in (
                dataset.main, dataset.aligned_main, dataset.power, dataset.labels
            ))
            cache = None
            if training:
                chunks = []
                size = int(self.config["batch_size"])
                for start in range(0, len(dataset), size):
                    chunks.append(self.net.intrinsic_masking.prepare_cache(
                        tensors[0][start:start + size], tensors[1][start:start + size]
                    ))
                cache = tuple(torch.cat(parts, dim=0) for parts in zip(*chunks))
                if not all(torch.isfinite(value).all() for value in cache):
                    raise FloatingPointError("Non-finite intrinsic cache.")
            self.resident[id(loader)] = tensors, cache
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        with (self.result_savepath / "cache_timing.json").open("w") as handle:
            json.dump({"preparation_seconds": perf_counter() - started}, handle, indent=2)
        return loaders

    def _run_epoch(self, loader, training):
        self.net.train(training)
        tensors, cache = self.resident[id(loader)]
        totals = torch.zeros(10, device=self.device, dtype=torch.float64)
        main_count = side_count = batches = 0
        records = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for indices in loader:
                indices = indices.to(self.device)
                main, aligned, power, labels = (value[indices] for value in tensors)
                payload = {"aligned_main": aligned, "power": power}
                if cache is not None:
                    payload["intrinsic_cache"] = tuple(value[indices] for value in cache)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits, aux = self.net(main, payload, return_aux=True)
                main_logits, side_logits = aux["main_logits"], aux["side_logits"]
                main_loss = self.loss_func(main_logits, labels)
                side_loss = self.loss_func(main_logits.detach() + side_logits, labels)
                objective = main_loss + side_loss
                combined_loss = self.loss_func(logits, labels)
                if not torch.isfinite(objective):
                    raise FloatingPointError("Non-finite LYNet objective.")
                if training:
                    objective.backward()
                    gradients = [(name, parameter.grad) for name, parameter in
                                 self.net.named_parameters() if parameter.grad is not None]
                    # Every gradient is still checked at every batch, one CPU sync.
                    if not torch.stack([torch.isfinite(grad).all() for _, grad in gradients]).all():
                        bad = [name for name, grad in gradients if not torch.isfinite(grad).all()]
                        raise FloatingPointError(f"Non-finite LYNet gradients: {bad}.")
                    self.optimizer.step()
                    apply_model_constraints(self.net)
                predicted = logits.argmax(dim=1)
                main_predicted = main_logits.argmax(dim=1)
                values = (
                    combined_loss, objective, main_loss, side_loss,
                    main_logits.square().sum(), side_logits.square().sum(),
                    (predicted == labels).sum(), (main_predicted == labels).sum(),
                    ((main_predicted != labels) & (predicted == labels)).sum(),
                    ((main_predicted == labels) & (predicted != labels)).sum(),
                )
                # Convert each batch scalar to FP64, matching Python float sums.
                totals += torch.stack([value.detach().to(torch.float64) for value in values])
                records.append(torch.stack((labels, predicted), dim=1).detach())
                main_count += main_logits.numel()
                side_count += side_logits.numel()
                batches += 1
        if not batches:
            raise RuntimeError("LYNet protocol processed no batches.")
        combined, objective, main, side, main_square, side_square, correct, main_correct, corrected, harmed = totals.cpu().tolist()
        records = torch.cat(records).cpu().numpy()
        labels, predictions = records[:, 0], records[:, 1]
        main_rms, side_rms = float(np.sqrt(main_square / main_count)), float(np.sqrt(side_square / side_count))
        return {
            "loss": combined / batches, "objective_loss": objective / batches,
            "main_loss": main / batches, "side_loss": side / batches,
            "accuracy": correct / len(labels), "main_accuracy": main_correct / len(labels),
            "kappa": _safe_kappa(labels, predictions),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "side_corrected_count": int(corrected), "side_harmed_count": int(harmed),
            "main_logit_rms": main_rms, "side_logit_rms": side_rms,
            "side_main_logit_rms_ratio": side_rms / (main_rms + 1e-12),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2, 3]),
        }
