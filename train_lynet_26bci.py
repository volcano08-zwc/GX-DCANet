"""Train V121 on 26BCI with session_id-defined RUN-EA.

Workflow per patient:
1. pretrain on the original standard-data train/validation trial split;
2. for each treatment fold, choose epochs using grouped inner validation from
   the other two treatment folds;
3. restart from the same standard checkpoint, refit for the selected epoch
   count on all non-test data, and evaluate the untouched treatment fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from data.lynet_26bci import (
    Prepared26BCI,
    base_split_indices,
    fit_feature_scalers,
    hierarchical_sample_weights,
    prepare_case,
    resolve_data_root,
    select_inner_validation_groups,
    transform_features,
)
from model.registry import build_registered_model, validate_model_data_contract
from protocol.lynet_26bci_protocol import Dataset26BCI, Protocol26BCI
from utils import get_device, kaiming_init_weights, load_yaml, set_random_seed


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V121 / 26BCI session-RUN-EA experiment")
    parser.add_argument("--config", default=str(ROOT / "config" / "lynet_26bci.yaml"))
    parser.add_argument("--cases", default=None, help="Comma-separated subset, e.g. G or G,X")
    parser.add_argument("--epochs", type=int, default=None, help="Override both 500-epoch maxima")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size 64")
    parser.add_argument("--device", choices=("cpu", "gpu", "cuda", "npu", "ascend"), default=None)
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    return parser.parse_args()


def _model(config: dict, device: torch.device):
    return build_registered_model(
        config["network"], config["network_args"], device,
        dcanet_initializer=kaiming_init_weights,
    )


def _dataset(
    views: tuple[np.ndarray, np.ndarray, np.ndarray],
    data: Prepared26BCI,
    indices: np.ndarray,
    *,
    weighted: bool,
) -> Dataset26BCI:
    main, aligned, power = views
    weights = hierarchical_sample_weights(data, indices) if weighted else None
    return Dataset26BCI(
        main[indices], aligned[indices], power[indices], data.labels[indices],
        weights=weights, source_indices=indices,
    )


def _state_dict(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _split_audit(data: Prepared26BCI, indices: np.ndarray) -> dict:
    groups = np.unique(data.groups[indices])
    return {
        "window_count": int(len(indices)),
        "group_count": int(len(groups)),
        "group_keys": [data.group_keys[int(group)] for group in groups],
        "session_ids": sorted(set(str(value) for value in data.run_ids[indices])),
        "label_windows": {
            str(label): int(np.sum(data.labels[indices] == label)) for label in (0, 1)
        },
        "domain_windows": {
            "standard": int(np.sum(data.domains[indices] == 0)),
            "treatment": int(np.sum(data.domains[indices] == 1)),
        },
    }


def _assert_group_disjoint(data: Prepared26BCI, named_indices: dict[str, np.ndarray]) -> None:
    names = list(named_indices)
    for i, left in enumerate(names):
        left_groups = np.unique(data.groups[named_indices[left]])
        for right in names[i + 1:]:
            overlap = np.intersect1d(left_groups, np.unique(data.groups[named_indices[right]]))
            if overlap.size:
                raise RuntimeError(f"Group leakage between {left} and {right}: {overlap.tolist()}")


def _save_scalers(path: Path, scalers) -> None:
    np.savez(
        path,
        raw_mean=scalers.raw_mean, raw_scale=scalers.raw_scale,
        aligned_mean=scalers.aligned_mean, aligned_scale=scalers.aligned_scale,
        power_mean=scalers.power_mean, power_scale=scalers.power_scale,
    )


def run_patient(
    patient: str,
    data: Prepared26BCI,
    case_spec: dict,
    data_root: Path,
    config: dict,
    device: torch.device,
    output_root: Path,
) -> dict:
    patient_dir = output_root / patient
    patient_dir.mkdir(parents=True, exist_ok=True)
    split_path = (
        data_root / config["runtime_models_dir"] / data.subject_id / "models"
        / str(case_spec["base_model_version"]) / "split.json"
    )
    base_train, base_validation = base_split_indices(data, split_path)
    _assert_group_disjoint(data, {"base_train": base_train, "base_validation": base_validation})

    scalers = fit_feature_scalers(data, base_train)
    views = transform_features(data, scalers)
    _save_scalers(patient_dir / "feature_scalers.npz", scalers)
    (patient_dir / "alignment_audit.json").write_text(
        json.dumps(data.alignment_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    base_dir = patient_dir / "standard_pretrain"
    base_model = _model(config, device)
    base_protocol = Protocol26BCI(
        base_model, config, device, base_dir,
        lr=config["lr"], weight_decay=config["weight_decay"],
    )
    base_result = base_protocol.fit(
        _dataset(views, data, base_train, weighted=True),
        _dataset(views, data, base_validation, weighted=False),
        max_epochs=int(config["epochs"]),
        checkpoint_name="standard_best.pth",
    )
    base_checkpoint = base_dir / "standard_best.pth"

    standard_all = np.flatnonzero(data.domains == 0)
    fold_results = []
    oof_labels, oof_predictions, oof_probabilities, oof_indices = [], [], [], []
    for test_fold in (0, 1, 2):
        fold_dir = patient_dir / f"treatment_fold_{test_fold}"
        selection_dir = fold_dir / "selection"
        refit_dir = fold_dir / "refit"
        treatment_test = np.flatnonzero((data.domains == 1) & (data.folds == test_fold))
        treatment_outer_train = np.flatnonzero((data.domains == 1) & (data.folds != test_fold))
        treatment_inner_train, treatment_inner_validation = select_inner_validation_groups(
            data, treatment_outer_train,
            seed=int(config["random_seed"]) + test_fold,
        )
        selection_train = np.concatenate((standard_all, treatment_inner_train))
        _assert_group_disjoint(data, {
            "selection_train": selection_train,
            "selection_validation": treatment_inner_validation,
            "outer_test": treatment_test,
        })

        selection_model = _model(config, device)
        selection_model.load_state_dict(_state_dict(base_checkpoint, device))
        selection_protocol = Protocol26BCI(
            selection_model, config, device, selection_dir,
            lr=config["adaptation_lr"], weight_decay=config["adaptation_weight_decay"],
        )
        selection_result = selection_protocol.fit(
            _dataset(views, data, selection_train, weighted=True),
            _dataset(views, data, treatment_inner_validation, weighted=False),
            max_epochs=int(config["adaptation_epochs"]),
            checkpoint_name="adaptation_selection_best.pth",
        )

        # Inner validation chooses only the epoch count.  Refit from the same
        # standard checkpoint on every non-test group before outer evaluation.
        outer_train = np.concatenate((standard_all, treatment_outer_train))
        _assert_group_disjoint(data, {"outer_train": outer_train, "outer_test": treatment_test})
        refit_model = _model(config, device)
        refit_model.load_state_dict(_state_dict(base_checkpoint, device))
        refit_protocol = Protocol26BCI(
            refit_model, config, device, refit_dir,
            lr=config["adaptation_lr"], weight_decay=config["adaptation_weight_decay"],
        )
        refit_result = refit_protocol.fit_fixed(
            _dataset(views, data, outer_train, weighted=True),
            epochs=int(selection_result["best_epoch"]),
            checkpoint_name="adaptation_refit.pth",
        )
        test_metrics = refit_protocol.evaluate(
            _dataset(views, data, treatment_test, weighted=False)
        )
        np.savez(
            fold_dir / "test_predictions.npz",
            source_indices=test_metrics["source_indices"],
            labels=test_metrics["labels"],
            predictions=test_metrics["predictions"],
            probabilities=test_metrics["probabilities"],
        )
        serial_test = Protocol26BCI._serializable(test_metrics)
        split_audit = {
            "selection_train": _split_audit(data, selection_train),
            "selection_validation": _split_audit(data, treatment_inner_validation),
            "outer_refit_train": _split_audit(data, outer_train),
            "outer_test": _split_audit(data, treatment_test),
        }
        fold_record = {
            "test_fold": test_fold,
            "selection": selection_result,
            "refit": refit_result,
            "test": serial_test,
            "split_audit": split_audit,
        }
        (fold_dir / "result.json").write_text(
            json.dumps(fold_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fold_results.append(fold_record)
        oof_labels.extend(test_metrics["labels"].tolist())
        oof_predictions.extend(test_metrics["predictions"].tolist())
        oof_probabilities.extend(test_metrics["probabilities"].tolist())
        oof_indices.extend(test_metrics["source_indices"].tolist())

    labels = np.asarray(oof_labels)
    predictions = np.asarray(oof_predictions)
    probabilities = np.asarray(oof_probabilities)
    aggregate = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "window_count": int(len(labels)),
    }
    np.savez(
        patient_dir / "treatment_oof_predictions.npz",
        source_indices=np.asarray(oof_indices), labels=labels,
        predictions=predictions, probabilities=probabilities,
    )
    result = {
        "patient": patient,
        "subject_id": data.subject_id,
        "base": base_result,
        "base_split": {
            "train": _split_audit(data, base_train),
            "validation": _split_audit(data, base_validation),
        },
        "treatment_oof": aggregate,
        "folds": fold_results,
    }
    (patient_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    if args.epochs is not None:
        config["epochs"] = args.epochs
        config["adaptation_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.device is not None:
        config["preferred_device"] = args.device
    cache_dir = Path(config["cache_dir"])
    if not cache_dir.is_absolute():
        config["cache_dir"] = str((ROOT / cache_dir).resolve())
    validate_model_data_contract(
        config["network"], config["network_args"],
        channels=16, samples=500, num_classes=2,
    )
    data_root = resolve_data_root(config, config_path)
    cases = [value.strip() for value in (args.cases.split(",") if args.cases else config["cases"])]
    invalid = [case for case in cases if case not in config["case_specs"]]
    if invalid:
        raise ValueError(f"Unknown patient cases: {invalid}")

    prepared = {}
    for patient in cases:
        prepared[patient] = prepare_case(
            data_root, patient, config["case_specs"][patient], config,
            force=args.force_preprocess,
        )
        item = prepared[patient]
        print(
            f"[26BCI/{patient}] windows={len(item.labels)} "
            f"sessions={len(np.unique(item.run_ids))} "
            f"standard={np.sum(item.domains == 0)} treatment={np.sum(item.domains == 1)}",
            flush=True,
        )
    if args.preprocess_only:
        return

    set_random_seed(config)
    device = get_device(config)
    output_root = Path(config["out_folder"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results = []
    for patient in cases:
        set_random_seed(config)
        results.append(run_patient(
            patient, prepared[patient], config["case_specs"][patient],
            data_root, config, device, output_root,
        ))
    summary = {
        "protocol": config["protocol"],
        "batch_size": int(config["batch_size"]),
        "max_epochs": int(config["epochs"]),
        "patients": [
            {"patient": result["patient"], **result["treatment_oof"]} for result in results
        ],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
