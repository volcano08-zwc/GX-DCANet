"""Train standalone LYNet on BCI Competition IV 2a."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

# Required before the first CUDA/cuBLAS operation when deterministic
# algorithms are enabled on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn, optim

from data.lynet_2a import LYNetPreprocessing, prepare_lynet_views
from model.registry import (
    build_registered_model,
    model_args_from_config,
    validate_model_data_contract,
)
from protocol.lynet_protocol import LYNetDataset
from protocol.fast_protocol import FastLYNetProtocol as LYNetProtocol
from utils import (
    count_parameters,
    dict_to_yaml,
    get_device,
    kaiming_init_weights,
    load_yaml,
    set_random_seed,
)


PROTOCOL_NAME = "offline_transductive_run_baseline_best_e"
LYNET_NETWORKS = {"LYNet"}


def parse_subjects(specification: str) -> list[int]:
    subjects: set[int] = set()
    for part in specification.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = (int(value) for value in part.split("-", maxsplit=1))
            subjects.update(range(left, right + 1))
        else:
            subjects.add(int(part))
    result = sorted(subjects)
    if not result or any(subject not in range(1, 10) for subject in result):
        raise argparse.ArgumentTypeError("subjects must be within 1-9")
    return result


def validate_lynet_config(config: dict) -> LYNetPreprocessing:
    network = str(config.get("network"))
    if network not in LYNET_NETWORKS:
        raise ValueError(
            f"train_lynet_2a.py requires one of {sorted(LYNET_NETWORKS)}, got {network!r}."
        )
    model_args = model_args_from_config(config)
    validate_model_data_contract(
        network,
        model_args,
        channels=22,
        samples=1000,
        num_classes=4,
    )
    protocol = config.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("name") != PROTOCOL_NAME:
        raise ValueError(f"LYNet YAML requires protocol.name: {PROTOCOL_NAME}.")
    if protocol.get("reference_uses_evaluation_labels") is not False:
        raise ValueError("LYNet alignment protocol must explicitly forbid evaluation labels.")
    if protocol.get("checkpoint_selection") != "best_evaluation_accuracy":
        raise ValueError("LYNet source protocol selects the best evaluation accuracy.")
    subjects = list(config.get("subjects", []))
    if not subjects or any(int(subject) not in range(1, 10) for subject in subjects):
        raise ValueError("subjects must be a non-empty subset of 1..9.")
    if int(config.get("epochs", 0)) <= 0 or int(config.get("batch_size", 0)) <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if int(config.get("num_workers", -1)) < 0:
        raise ValueError("num_workers must be non-negative.")
    if float(config.get("lr", 0.0)) <= 0.0:
        raise ValueError("lr must be positive.")

    preprocessing = LYNetPreprocessing.from_mapping(config.get("preprocessing"))
    preprocessing.validate()
    return preprocessing


def build_network(config: dict, device: torch.device) -> nn.Module:
    return build_registered_model(
        str(config["network"]),
        model_args_from_config(config),
        device,
        dcanet_initializer=kaiming_init_weights,
    )


def write_summary(config: dict, results: list[tuple[str, dict[str, object]]]) -> None:
    output_dir = Path(config["out_folder"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "subject_results.csv"
    fields = (
        "subject",
        "best_epoch",
        "best_test_acc",
        "best_test_kappa",
        "best_eval_macro_f1",
        "best_side_corrected_count",
        "best_side_harmed_count",
        "best_side_main_logit_rms_ratio",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for subject_id, metrics in results:
            writer.writerow(
                {"subject": subject_id, **{field: metrics[field] for field in fields[1:]}}
            )

    accuracies = np.asarray([float(metrics["best_test_acc"]) for _, metrics in results])
    summary = {
        "model": str(config["network"]),
        "source_variant": str(config.get("experiment_name", config["network"])),
        "subjects": [subject for subject, _ in results],
        "mean_best_e_accuracy": float(accuracies.mean()),
        "mean_best_e_accuracy_percent": float(100.0 * accuracies.mean()),
        "sample_std_accuracy_percent": (
            float(100.0 * accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0
        ),
        "mean_best_e_kappa": float(
            np.mean([float(metrics["best_test_kappa"]) for _, metrics in results])
        ),
        "parameter_count": int(config["parameter_count"]),
        "protocol": (
            "AxxT train; AxxE every epoch; exploratory Best-E; "
            "offline-transductive run-baseline alignment"
        ),
        "reporting_warning": (
            "Evaluation-session labels select the best epoch. Results are exploratory "
            "and must not be reported as inductive/causal-online performance."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main(config: dict, *, preprocess_only: bool = False, force_preprocess: bool = False) -> None:
    preprocessing = validate_lynet_config(config)
    output_dir = Path(config["out_folder"])

    if preprocess_only:
        for subject in config["subjects"]:
            views = prepare_lynet_views(
                config["data_root"],
                int(subject),
                preprocessing,
                config["cache_dir"],
                force=force_preprocess,
            )
            print(
                f"A{int(subject):02d} LYNet cache ready: "
                f"main={views.train_main.shape}/{views.eval_main.shape}, "
                f"power={views.train_power.shape}/{views.eval_power.shape}"
            )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(config)
    device = get_device(config)
    # Build once to record the exact parameter count, then reset per subject.
    parameter_probe = build_network(config, device)
    config = dict(config)
    config["parameter_count"] = count_parameters(parameter_probe)
    del parameter_probe
    dict_to_yaml(output_dir / "config_used.yaml", config)
    print(f"Selected network: {config['network']} ({config['parameter_count']} parameters)")
    print("Code will be running on device", device)

    results: list[tuple[str, dict[str, object]]] = []
    for subject in config["subjects"]:
        subject = int(subject)
        subject_id = f"A{subject:02d}"
        print(f"\nProcessing subject {subject_id}")
        views = prepare_lynet_views(
            config["data_root"],
            subject,
            preprocessing,
            config["cache_dir"],
            force=force_preprocess,
        )
        train_dataset = LYNetDataset(
            views.train_main,
            views.train_aligned_main,
            views.train_power,
            views.train_labels,
        )
        eval_dataset = LYNetDataset(
            views.eval_main,
            views.eval_aligned_main,
            views.eval_power,
            views.eval_labels,
        )

        set_random_seed(config)
        net = build_network(config, device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(
            net.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        result_path = output_dir / f"results_{subject_id}"
        result_path.mkdir(parents=True, exist_ok=True)
        dict_to_yaml(result_path / "config_used.yaml", config)
        with (result_path / "data_audit.json").open("w", encoding="utf-8") as handle:
            json.dump(views.metadata, handle, ensure_ascii=False, indent=2)

        trainer = LYNetProtocol(
            net,
            config,
            optimizer,
            loss_func,
            result_savepath=result_path,
            device=device,
        )
        metrics = trainer.train_test(train_dataset, eval_dataset, subject_id)
        results.append((subject_id, metrics))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary(config, results)
    print(f"\nLYNet experiment complete: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/lynet_2a.yaml",
    )
    parser.add_argument("--subjects", type=parse_subjects)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    args = parser.parse_args()

    loaded_config = load_yaml(args.config)
    if args.subjects is not None:
        loaded_config["subjects"] = args.subjects
    if args.epochs is not None:
        loaded_config["epochs"] = args.epochs
    main(
        loaded_config,
        preprocess_only=args.preprocess_only,
        force_preprocess=args.force_preprocess,
    )
