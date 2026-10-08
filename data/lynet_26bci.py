"""26BCI data adapter for V121 session-level RUN-EA experiments.

The exported ``dataset.npz`` supplies the labelled 1-second windows.  Raw
continuous recordings are read only to obtain label-free, non-task reference
windows for one Euclidean-alignment transform per ``session_id``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import butter, filtfilt


CHANNELS = (
    "FC5", "FC3", "CPz", "FC4", "FC6", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP4", "CP6",
)
EEG_CHANNELS = 16
SAMPLE_RATE = 500
WINDOW_SAMPLES = 500
POWER_BANDS_HZ = (
    (4.0, 8.0), (8.0, 12.0), (12.0, 16.0),
    (16.0, 20.0), (20.0, 24.0), (24.0, 32.0),
)
POWER_FEATURES = EEG_CHANNELS * len(POWER_BANDS_HZ)
CACHE_VERSION = 2


@dataclass(frozen=True)
class Preprocessing26BCI:
    low_hz: float = 4.0
    high_hz: float = 40.0
    filter_order: int = 4
    reflect_padding: int = 100
    ea_reference_seconds: float = 1.0
    standard_prepare_margin_seconds: float = 1.0
    treatment_gap_guard_seconds: float = 1.0
    treatment_reference_cap: int = 50
    shrinkage: float = 0.1
    covariance_epsilon: float = 1e-6
    power_bands_hz: tuple[tuple[float, float], ...] = POWER_BANDS_HZ

    @classmethod
    def from_mapping(cls, values: dict | None) -> "Preprocessing26BCI":
        values = dict(values or {})
        if "power_bands_hz" in values:
            values["power_bands_hz"] = tuple(
                (float(a), float(b)) for a, b in values["power_bands_hz"]
            )
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if not 0 < self.low_hz < self.high_hz < SAMPLE_RATE / 2:
            raise ValueError("Invalid 26BCI band-pass frequencies.")
        if int(round(self.ea_reference_seconds * SAMPLE_RATE)) != WINDOW_SAMPLES:
            raise ValueError("EA references must be exactly one second (500 samples).")
        if not 0 <= self.shrinkage <= 1 or self.covariance_epsilon <= 0:
            raise ValueError("Invalid EA shrinkage/epsilon.")
        if len(self.power_bands_hz) != 6:
            raise ValueError("Exactly six log-power bands are required.")


@dataclass(frozen=True)
class FeatureScalers:
    raw_mean: np.ndarray
    raw_scale: np.ndarray
    aligned_mean: np.ndarray
    aligned_scale: np.ndarray
    power_mean: np.ndarray
    power_scale: np.ndarray


@dataclass
class Prepared26BCI:
    patient: str
    subject_id: str
    raw: np.ndarray
    aligned: np.ndarray
    power: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    domains: np.ndarray
    folds: np.ndarray
    run_ids: np.ndarray
    group_keys: tuple[str, ...]
    group_metadata: tuple[dict, ...]
    alignment_audit: dict

    def __post_init__(self) -> None:
        n = len(self.labels)
        if self.raw.shape != (n, EEG_CHANNELS, WINDOW_SAMPLES):
            raise ValueError(f"Unexpected raw shape {self.raw.shape}.")
        if self.aligned.shape != self.raw.shape or self.power.shape != (n, POWER_FEATURES):
            raise ValueError("Aligned/power data do not satisfy the 26BCI contract.")
        for values in (self.groups, self.domains, self.folds, self.run_ids):
            if len(values) != n:
                raise ValueError("Per-window metadata length mismatch.")


def resolve_data_root(config: dict, config_path: str | Path) -> Path:
    override = os.environ.get("BCI26_DATA_ROOT")
    configured = Path(override or config["data_root"])
    if not configured.is_absolute():
        configured = Path(config_path).resolve().parent / configured
    root = configured.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"26BCI root not found: {root}. Set BCI26_DATA_ROOT on the server."
        )
    return root


def _json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _session_files(session_dir: Path) -> tuple[Path, Path]:
    manifest = session_dir / "manifest.json"
    raw = session_dir / "raw_eeg.npy"
    if manifest.is_file() and raw.is_file():
        return manifest, raw
    manifest = session_dir / "manifest.partial.json"
    raw = session_dir / "raw_eeg.partial.bin"
    if manifest.is_file() and raw.is_file():
        return manifest, raw
    raise FileNotFoundError(f"Missing raw/manifest pair in {session_dir}.")


def _load_physical_raw(raw_path: Path, manifest: dict) -> np.ndarray:
    if raw_path.suffix == ".npy":
        data = np.load(raw_path, mmap_mode="r")
    else:
        channels = len(manifest["channels"])
        values = np.memmap(raw_path, mode="r", dtype=manifest.get("recovery_dtype", "float32"))
        data = values[: (values.size // channels) * channels].reshape(-1, channels)
    if data.ndim != 2 or data.shape[1] != EEG_CHANNELS:
        raise ValueError(f"Unexpected continuous EEG shape {data.shape} in {raw_path}.")
    count = min(int(manifest.get("sample_count", len(data))), len(data))
    return data[:count]


def _canonical_window(data: np.ndarray, manifest: dict, start: int) -> np.ndarray:
    physical = tuple(manifest["channels"])
    missing = [name for name in CHANNELS if name not in physical]
    if missing:
        raise ValueError(f"Missing canonical channels: {missing}.")
    indices = [physical.index(name) for name in CHANNELS]
    stop = start + WINDOW_SAMPLES
    if start < 0 or stop > len(data):
        raise ValueError(f"Reference [{start}, {stop}) is outside raw recording.")
    return np.asarray(data[start:stop, indices].T, dtype=np.float64)


def _preprocess_reference(window: np.ndarray, cfg: Preprocessing26BCI) -> np.ndarray:
    # Match the exported dataset: canonical reorder -> CAR -> 4-40 Hz filtfilt.
    car = window - window.mean(axis=0, keepdims=True)
    padded = np.pad(car, ((0, 0), (cfg.reflect_padding, cfg.reflect_padding)), mode="reflect")
    numerator, denominator = butter(
        cfg.filter_order,
        (cfg.low_hz, cfg.high_hz),
        btype="bandpass",
        fs=SAMPLE_RATE,
    )
    filtered = filtfilt(numerator, denominator, padded, axis=1)
    return filtered[:, cfg.reflect_padding:-cfg.reflect_padding]


def _standard_reference_starts(manifest: dict, markers: list[dict], cfg: Preprocessing26BCI) -> list[int]:
    phase = next(
        (m for m in markers if m.get("event") == "phase" and m.get("phase") == "initial_prepare"),
        None,
    )
    if phase is None:
        raise ValueError(f"Session {manifest['session_id']} has no initial_prepare marker.")
    start = int(phase["sample_index"]) + int(round(cfg.standard_prepare_margin_seconds * SAMPLE_RATE))
    duration = float(phase.get("duration_s", manifest.get("initial_prepare_seconds", 5.0)))
    usable = duration - 2.0 * cfg.standard_prepare_margin_seconds
    count = int(usable // cfg.ea_reference_seconds)
    if count < 1:
        raise ValueError(f"Session {manifest['session_id']} has no usable initial_prepare reference.")
    return [start + i * WINDOW_SAMPLES for i in range(count)]


def _merge_intervals(intervals: Iterable[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    cleaned = sorted((max(0, a), min(limit, b)) for a, b in intervals if b > a)
    merged: list[list[int]] = []
    for start, stop in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(a, b) for a, b in merged]


def _treatment_reference_starts(manifest: dict, markers: list[dict], cfg: Preprocessing26BCI) -> list[int]:
    count = int(manifest["sample_count"])
    guard = int(round(cfg.treatment_gap_guard_seconds * SAMPLE_RATE))
    blocked: list[tuple[int, int]] = []
    for segment in manifest.get("segments", []):
        blocked.append((int(segment["start_sample"]) - guard, int(segment["end_sample"]) + guard))
    for marker in markers:
        if str(marker.get("event", "")).startswith("source_interruption"):
            sample = int(marker.get("sample_index", 0))
            blocked.append((sample - guard, sample + guard))
    merged = _merge_intervals(blocked, count)
    gaps: list[tuple[int, int]] = []
    cursor = guard
    for start, stop in merged:
        if start - cursor >= WINDOW_SAMPLES:
            gaps.append((cursor, start))
        cursor = max(cursor, stop)
    if count - guard - cursor >= WINDOW_SAMPLES:
        gaps.append((cursor, count - guard))
    starts = [a + ((b - a - WINDOW_SAMPLES) // 2) for a, b in gaps]
    if len(starts) > cfg.treatment_reference_cap:
        selected = np.linspace(0, len(starts) - 1, cfg.treatment_reference_cap).round().astype(int)
        starts = [starts[i] for i in selected]
    return starts


def _read_reference_windows(
    session_dir: Path,
    domain: int,
    cfg: Preprocessing26BCI,
) -> tuple[np.ndarray, list[int], str]:
    manifest_path, raw_path = _session_files(session_dir)
    manifest = _json(manifest_path)
    markers_path = session_dir / ("markers.json" if (session_dir / "markers.json").is_file() else "markers.partial.json")
    markers = _json(markers_path)
    starts = (
        _standard_reference_starts(manifest, markers, cfg)
        if domain == 0
        else _treatment_reference_starts(manifest, markers, cfg)
    )
    raw = _load_physical_raw(raw_path, manifest)
    valid = [start for start in starts if 0 <= start and start + WINDOW_SAMPLES <= len(raw)]
    windows = [
        _preprocess_reference(_canonical_window(raw, manifest, start), cfg)
        for start in valid
    ]
    result = np.stack(windows) if windows else np.empty((0, EEG_CHANNELS, WINDOW_SAMPLES))
    source = "initial_prepare" if domain == 0 else "unlabelled_inter_segment_gap"
    return result, valid, source


def _whitening(references: np.ndarray, cfg: Preprocessing26BCI) -> np.ndarray:
    if references.ndim != 3 or references.shape[1:] != (EEG_CHANNELS, WINDOW_SAMPLES):
        raise ValueError(f"Unexpected EA reference shape {references.shape}.")
    centered = references - references.mean(axis=2, keepdims=True)
    covariance = np.einsum("nct,ndt->ncd", centered, centered) / (WINDOW_SAMPLES - 1)
    mean_cov = covariance.mean(axis=0)
    target = np.trace(mean_cov) / EEG_CHANNELS * np.eye(EEG_CHANNELS)
    regularized = (1.0 - cfg.shrinkage) * mean_cov + cfg.shrinkage * target
    values, vectors = np.linalg.eigh(regularized)
    values = np.maximum(values, cfg.covariance_epsilon)
    return (vectors * values[None, :] ** -0.5) @ vectors.T


def log_band_power(waveforms: np.ndarray, bands=POWER_BANDS_HZ) -> np.ndarray:
    centered = np.asarray(waveforms, dtype=np.float64) - waveforms.mean(axis=2, keepdims=True)
    taper = np.hanning(WINDOW_SAMPLES)
    spectrum = np.fft.rfft(centered * taper, axis=2)
    periodogram = np.abs(spectrum) ** 2 / np.square(taper).sum()
    frequencies = np.fft.rfftfreq(WINDOW_SAMPLES, 1.0 / SAMPLE_RATE)
    features = []
    for index, (low, high) in enumerate(bands):
        mask = (frequencies >= low) & (
            frequencies <= high if index == len(bands) - 1 else frequencies < high
        )
        features.append(np.log(np.maximum(periodogram[:, :, mask].mean(axis=2), 1e-12)))
    return np.asarray(np.stack(features, axis=1).reshape(len(waveforms), -1), dtype=np.float32)


def _session_dir(runtime_root: Path, subject_id: str, domain: int, session_id: str) -> Path:
    folder = "sessions" if domain == 0 else "treatment_sessions"
    path = runtime_root / subject_id / folder / session_id
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def prepare_case(
    data_root: Path,
    patient: str,
    case_spec: dict,
    config: dict,
    *,
    force: bool = False,
) -> Prepared26BCI:
    derived = data_root / config["derived_data_dir"] / patient
    metadata_path = derived / "metadata.json"
    dataset_path = derived / "dataset.npz"
    metadata = _json(metadata_path)
    cfg = Preprocessing26BCI.from_mapping(config.get("preprocessing"))
    subject_id = str(metadata["subject_id"])
    runtime_root = data_root / config["runtime_data_dir"]
    cache_root = Path(config.get("cache_dir", "cache/26bci_v121"))
    if not cache_root.is_absolute():
        cache_root = Path.cwd() / cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{patient}_session_ea_v{CACHE_VERSION}.npz"

    with np.load(dataset_path) as source:
        raw = np.asarray(source["x"], dtype=np.float32)
        labels = np.asarray(source["y"], dtype=np.int64)
        groups = np.asarray(source["group_index"], dtype=np.int32)
        domains = np.asarray(source["domain"], dtype=np.int8)
        folds = np.asarray(source["fold"], dtype=np.int8)
    group_meta = tuple(metadata["groups"])
    group_keys = tuple(str(group["key"]) for group in group_meta)
    group_sessions = np.asarray([str(group["session"]) for group in group_meta], dtype=object)
    run_ids = group_sessions[groups]

    if cache_path.is_file() and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            if (
                int(cached["cache_version"]) == CACHE_VERSION
                and tuple(cached["dataset_shape"]) == raw.shape
            ):
                return Prepared26BCI(
                    patient, subject_id, raw,
                    np.asarray(cached["aligned"], dtype=np.float32),
                    np.asarray(cached["power"], dtype=np.float32),
                    labels, groups, domains, folds, run_ids, group_keys, group_meta,
                    json.loads(str(cached["audit_json"])),
                )

    session_domains: dict[str, int] = {}
    for group in group_meta:
        session_domains.setdefault(str(group["session"]), int(group["domain"]))
    references: dict[str, np.ndarray] = {}
    audit: dict[str, dict] = {}
    pooled_standard: list[np.ndarray] = []
    for session, domain in session_domains.items():
        if domain != 0:
            continue
        windows, starts, source = _read_reference_windows(
            _session_dir(runtime_root, subject_id, domain, session), domain, cfg
        )
        references[session] = windows
        pooled_standard.append(windows)
        audit[session] = {"domain": "standard", "source": source, "starts": starts, "fallback": False}
    standard_pool = np.concatenate(pooled_standard, axis=0)
    for session, domain in session_domains.items():
        if domain != 1:
            continue
        windows, starts, source = _read_reference_windows(
            _session_dir(runtime_root, subject_id, domain, session), domain, cfg
        )
        fallback = len(windows) == 0
        references[session] = standard_pool if fallback else windows
        audit[session] = {
            "domain": "treatment", "source": source,
            "starts": starts, "fallback": fallback,
        }

    aligned = np.empty_like(raw)
    matrices: dict[str, list[list[float]]] = {}
    for session in sorted(session_domains):
        transform = _whitening(references[session], cfg)
        selection = np.flatnonzero(run_ids == session)
        aligned[selection] = np.einsum(
            "cd,ndt->nct", transform, raw[selection], optimize=True
        ).astype(np.float32)
        matrices[session] = transform.tolist()
        audit[session]["reference_count"] = int(len(references[session]))
    power = log_band_power(aligned, cfg.power_bands_hz)
    alignment_audit = {
        "run_definition": "session_id",
        "label_free_reference": True,
        "preprocessing": asdict(cfg),
        "sessions": audit,
        "whitening_matrices": matrices,
    }
    np.savez(
        cache_path,
        cache_version=np.asarray(CACHE_VERSION),
        dataset_shape=np.asarray(raw.shape),
        aligned=aligned,
        power=power,
        audit_json=np.asarray(json.dumps(alignment_audit, ensure_ascii=False)),
    )
    return Prepared26BCI(
        patient, subject_id, raw, aligned, power, labels, groups, domains, folds,
        run_ids, group_keys, group_meta, alignment_audit,
    )


def base_split_indices(data: Prepared26BCI, split_path: Path) -> tuple[np.ndarray, np.ndarray]:
    split = _json(split_path)
    key_to_group = {key: i for i, key in enumerate(data.group_keys)}
    result = []
    for name in ("train", "validation"):
        selected_groups = []
        for trial in split["assignments"][name]:
            key = f"standard:{trial['session_id']}:{trial['original_trial_index']}"
            if key not in key_to_group:
                raise KeyError(f"Base split trial not found in exported data: {key}")
            selected_groups.append(key_to_group[key])
        result.append(np.flatnonzero(np.isin(data.groups, selected_groups)))
    train, validation = result
    if np.intersect1d(data.groups[train], data.groups[validation]).size:
        raise RuntimeError("Base split leaks a trial group between train/validation.")
    return train, validation


def fit_feature_scalers(data: Prepared26BCI, indices: np.ndarray) -> FeatureScalers:
    def channel_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values[indices].mean(axis=(0, 2), keepdims=True)
        scale = values[indices].std(axis=(0, 2), keepdims=True)
        return mean, np.maximum(scale, 1e-6)
    raw_mean, raw_scale = channel_stats(data.raw)
    aligned_mean, aligned_scale = channel_stats(data.aligned)
    power_mean = data.power[indices].mean(axis=0, keepdims=True)
    power_scale = np.maximum(data.power[indices].std(axis=0, keepdims=True), 1e-6)
    return FeatureScalers(raw_mean, raw_scale, aligned_mean, aligned_scale, power_mean, power_scale)


def transform_features(
    data: Prepared26BCI,
    scalers: FeatureScalers,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = ((data.raw - scalers.raw_mean) / scalers.raw_scale)[:, None]
    aligned = ((data.aligned - scalers.aligned_mean) / scalers.aligned_scale)[:, None]
    power = (data.power - scalers.power_mean) / scalers.power_scale
    return raw.astype(np.float32), aligned.astype(np.float32), power.astype(np.float32)


def select_inner_validation_groups(
    data: Prepared26BCI,
    candidate_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole treatment segments, with both labels when available."""
    candidate_groups = np.unique(data.groups[candidate_indices])
    rng = np.random.default_rng(seed)
    validation_groups: list[int] = []
    for label in (0, 1):
        labelled = [
            int(group) for group in candidate_groups
            if np.any((data.groups == group) & (data.labels == label))
        ]
        if labelled:
            rng.shuffle(labelled)
            validation_groups.append(labelled[0])
    validation = candidate_indices[np.isin(data.groups[candidate_indices], validation_groups)]
    training = candidate_indices[~np.isin(data.groups[candidate_indices], validation_groups)]
    if not len(validation) or not len(training):
        raise RuntimeError("Unable to form grouped inner treatment validation split.")
    return training, validation


def hierarchical_sample_weights(data: Prepared26BCI, indices: np.ndarray) -> np.ndarray:
    """Equalize domain, then label, then complete group within a training set."""
    weights = np.zeros(len(indices), dtype=np.float32)
    local_domain = data.domains[indices]
    local_label = data.labels[indices]
    local_group = data.groups[indices]
    present_domains = np.unique(local_domain)
    for domain in present_domains:
        domain_mask = local_domain == domain
        labels = np.unique(local_label[domain_mask])
        for label in labels:
            pair_mask = domain_mask & (local_label == label)
            groups = np.unique(local_group[pair_mask])
            for group in groups:
                mask = pair_mask & (local_group == group)
                weights[mask] = 1.0 / (
                    len(present_domains) * len(labels) * len(groups) * mask.sum()
                )
    return weights / np.maximum(weights.mean(), 1e-12)


__all__ = [
    "Prepared26BCI", "FeatureScalers", "Preprocessing26BCI",
    "base_split_indices", "fit_feature_scalers", "hierarchical_sample_weights",
    "prepare_case", "resolve_data_root", "select_inner_validation_groups",
    "transform_features",
]
