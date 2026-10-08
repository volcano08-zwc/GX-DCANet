"""RUN-EA preprocessing used by the LYNet integration.

This is the self-contained replacement for the missing historical
``v4.gdf_data`` dependency referenced by the supplied LYNet source.  It keeps
the source protocol explicit: each subject's T and E sessions are separate,
and every evaluation-run alignment matrix is estimated from all *unlabelled*
pre-cue baselines in that run.  Consequently, this is an offline-transductive
protocol, not an inductive or causal-online protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler


EEG_CHANNELS = (
    "EEG-Fz",
    "EEG-0",
    "EEG-1",
    "EEG-2",
    "EEG-3",
    "EEG-4",
    "EEG-5",
    "EEG-C3",
    "EEG-6",
    "EEG-Cz",
    "EEG-7",
    "EEG-C4",
    "EEG-8",
    "EEG-9",
    "EEG-10",
    "EEG-11",
    "EEG-12",
    "EEG-13",
    "EEG-14",
    "EEG-Pz",
    "EEG-15",
    "EEG-16",
)
EEG_CHANNEL_COUNT = 22
SAMPLES = 1000
TRIALS = 288
RUNS = 6
TRIALS_PER_RUN = 48
POWER_WINDOWS = 4
POWER_BANDS_HZ = (
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 32.0),
)
POWER_FEATURES = POWER_WINDOWS * len(POWER_BANDS_HZ) * EEG_CHANNEL_COUNT
CACHE_FORMAT_VERSION = 2


@dataclass(frozen=True)
class LYNetPreprocessing:
    sampling_rate: int = 250
    task_start_s: float = 0.0
    task_stop_s: float = 4.0
    baseline_start_s: float = -2.0
    baseline_stop_s: float = 0.0
    power_windows: int = POWER_WINDOWS
    power_bands_hz: tuple[tuple[float, float], ...] = POWER_BANDS_HZ
    shrinkage: float = 0.1
    eta: float = 1.0
    covariance_epsilon: float = 1e-6
    main_standardize: bool = True
    power_standardize: bool = True

    @classmethod
    def from_mapping(cls, values: dict | None) -> "LYNetPreprocessing":
        values = dict(values or {})
        if "power_bands_hz" in values:
            values["power_bands_hz"] = tuple(
                (float(low), float(high)) for low, high in values["power_bands_hz"]
            )
        return cls(**values)

    def validate(self) -> None:
        if int(self.sampling_rate) != 250:
            raise ValueError("LYNet source protocol requires sampling_rate=250 Hz.")
        if int(round((self.task_stop_s - self.task_start_s) * self.sampling_rate)) != SAMPLES:
            raise ValueError("LYNet task window must contain exactly 1000 samples.")
        if not self.baseline_start_s < self.baseline_stop_s <= self.task_start_s:
            raise ValueError("Baseline must end no later than the task window starts.")
        if self.power_windows != POWER_WINDOWS:
            raise ValueError("LYNet requires four one-second power windows.")
        if len(self.power_bands_hz) != 6:
            raise ValueError("LYNet requires exactly six power bands.")
        nyquist = self.sampling_rate / 2.0
        for low, high in self.power_bands_hz:
            if not 0.0 <= low < high <= nyquist:
                raise ValueError(f"Invalid power band ({low}, {high}) at {self.sampling_rate} Hz.")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1].")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be in [0, 1].")
        if self.covariance_epsilon <= 0.0:
            raise ValueError("covariance_epsilon must be positive.")


@dataclass(frozen=True)
class GDFSession:
    main_task: np.ndarray
    physical_task: np.ndarray
    baseline: np.ndarray
    labels: np.ndarray
    run_ids: np.ndarray
    artifact_flags: np.ndarray
    channel_names: tuple[str, ...]


@dataclass(frozen=True)
class LYNetViews:
    train_main: np.ndarray
    eval_main: np.ndarray
    train_aligned_main: np.ndarray
    eval_aligned_main: np.ndarray
    train_power: np.ndarray
    eval_power: np.ndarray
    train_labels: np.ndarray
    eval_labels: np.ndarray
    metadata: dict[str, object]


def _repair_gdf_missing_samples(data: np.ndarray) -> np.ndarray:
    """Replace GDF run-separator sentinels/NaNs with a channel-wise mean."""

    repaired = np.asarray(data, dtype=np.float64).copy()
    for channel_index in range(repaired.shape[0]):
        channel = repaired[channel_index]
        minimum = np.nanmin(channel)
        missing = np.isnan(channel) | (channel == minimum)
        if missing.any():
            valid = channel[~missing]
            if valid.size == 0:
                raise ValueError(f"EEG channel {channel_index} has no valid GDF samples.")
            channel[missing] = valid.mean()
    return repaired


def _events(raw: mne.io.BaseRaw, descriptions: tuple[str, ...]) -> np.ndarray:
    event_id = {description: int(description) for description in descriptions}
    events, _ = mne.events_from_annotations(
        raw,
        event_id=event_id,
        use_rounding=True,
        verbose="ERROR",
    )
    return events


def _trial_starts_for_cues(cues: np.ndarray, starts: np.ndarray, sfreq: float) -> np.ndarray:
    """Pair every cue with its immediately preceding 768 trial-start marker."""

    paired: list[int] = []
    start_samples = starts[:, 0]
    for cue_sample in cues[:, 0]:
        candidates = start_samples[start_samples <= cue_sample]
        if not len(candidates):
            raise ValueError(f"Cue at {cue_sample} has no preceding trial-start marker.")
        paired.append(int(candidates[-1]))
    paired_array = np.asarray(paired, dtype=np.int64)
    expected_delay = int(round(2.0 * sfreq))
    if not np.all(np.abs((cues[:, 0] - paired_array) - expected_delay) <= 1):
        raise ValueError("BCI-2a cue/trial-start timing is not the expected two seconds.")
    return paired_array


def _artifact_flags(
    raw: mne.io.BaseRaw,
    trial_starts: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    rejected = _events(raw, ("1023",))
    flags = np.zeros(TRIALS, dtype=bool)
    if not len(rejected):
        return flags
    stop_offset = int(round(7.5 * sfreq))
    for index, start in enumerate(trial_starts):
        flags[index] = bool(
            np.any((rejected[:, 0] >= start) & (rejected[:, 0] < start + stop_offset))
        )
    return flags


def load_gdf_session(
    gdf_path: str | Path,
    labels_path: str | Path,
    preprocessing: LYNetPreprocessing | None = None,
) -> GDFSession:
    """Load one chronological AxxT/AxxE recording without using task labels in alignment."""

    preprocessing = preprocessing or LYNetPreprocessing()
    preprocessing.validate()
    gdf_path = Path(gdf_path).resolve()
    labels_path = Path(labels_path).resolve()
    session = gdf_path.stem.upper()[-1:]
    if session not in {"T", "E"}:
        raise ValueError(f"Unexpected BCI-2a filename: {gdf_path.name}.")

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, preprocessing.sampling_rate):
        raise ValueError(f"Expected {preprocessing.sampling_rate} Hz, got {sfreq} Hz.")
    missing_channels = [name for name in EEG_CHANNELS if name not in raw.ch_names]
    if missing_channels:
        raise ValueError(f"Missing EEG channels in {gdf_path.name}: {missing_channels}.")

    cue_descriptions = ("769", "770", "771", "772") if session == "T" else ("783",)
    cues = _events(raw, cue_descriptions)
    starts = _events(raw, ("768",))
    if len(cues) != TRIALS or len(starts) != TRIALS:
        raise ValueError(
            f"Expected {TRIALS} cues/starts in {gdf_path.name}, "
            f"got {len(cues)}/{len(starts)}."
        )

    labels = np.asarray(loadmat(labels_path)["classlabel"]).reshape(-1).astype(np.int64)
    if labels.shape != (TRIALS,) or not np.array_equal(np.unique(labels), [1, 2, 3, 4]):
        raise ValueError(f"Invalid classlabel array in {labels_path}: {labels.shape}.")
    if session == "T":
        np.testing.assert_array_equal(cues[:, 2] - 768, labels)
    labels = labels - 1

    continuous = raw.get_data(picks=list(EEG_CHANNELS))
    # Physical microvolts make the covariance audit human-readable.  Both main
    # and power paths are subsequently standardized, so this unit conversion
    # does not leak labels or alter their standardized representation.
    continuous = _repair_gdf_missing_samples(continuous) * 1e6
    cue_samples = cues[:, 0] - raw.first_samp
    trial_starts = _trial_starts_for_cues(cues, starts, sfreq) - raw.first_samp

    task_start = int(round(preprocessing.task_start_s * sfreq))
    task_stop = int(round(preprocessing.task_stop_s * sfreq))
    baseline_start = int(round(preprocessing.baseline_start_s * sfreq))
    baseline_stop = int(round(preprocessing.baseline_stop_s * sfreq))
    task_trials: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    for trial_index, cue in enumerate(cue_samples):
        task = continuous[:, cue + task_start : cue + task_stop]
        baseline = continuous[:, cue + baseline_start : cue + baseline_stop]
        if task.shape != (EEG_CHANNEL_COUNT, SAMPLES):
            raise ValueError(f"Bad task epoch {trial_index} in {gdf_path.name}: {task.shape}.")
        if baseline.shape[0] != EEG_CHANNEL_COUNT or baseline.shape[1] < 2:
            raise ValueError(
                f"Bad baseline epoch {trial_index} in {gdf_path.name}: {baseline.shape}."
            )
        task_trials.append(task)
        baselines.append(baseline)

    task_array = np.asarray(np.stack(task_trials), dtype=np.float32)
    baseline_array = np.asarray(np.stack(baselines), dtype=np.float32)
    run_ids = np.repeat(np.arange(RUNS, dtype=np.int64), TRIALS_PER_RUN)
    if run_ids.shape != (TRIALS,):
        raise RuntimeError("Internal run-id construction failed.")
    flags = _artifact_flags(raw, trial_starts + raw.first_samp, sfreq)
    if not np.isfinite(task_array).all() or not np.isfinite(baseline_array).all():
        raise FloatingPointError(f"Non-finite samples remain in {gdf_path.name}.")

    return GDFSession(
        main_task=task_array,
        physical_task=task_array.copy(),
        baseline=baseline_array,
        labels=labels.astype(np.int64, copy=False),
        run_ids=run_ids,
        artifact_flags=flags,
        channel_names=EEG_CHANNELS,
    )


def _run_reference_covariance(
    baselines: np.ndarray,
    shrinkage: float,
    epsilon: float,
) -> np.ndarray:
    centred = baselines - baselines.mean(axis=2, keepdims=True)
    covariances = np.einsum("nct,ndt->ncd", centred, centred, optimize=True)
    covariances /= max(1, baselines.shape[2] - 1)
    covariance = covariances.mean(axis=0)
    scale = float(np.trace(covariance) / covariance.shape[0])
    if not np.isfinite(scale) or scale <= 0.0:
        raise FloatingPointError("Run-baseline covariance has invalid scale.")
    identity = np.eye(covariance.shape[0], dtype=np.float64)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * scale * identity
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, epsilon * scale)
    return (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T


def _session_alignment_transforms(
    baselines: np.ndarray,
    run_ids: np.ndarray,
    shrinkage: float,
    eta: float,
    epsilon: float,
) -> np.ndarray:
    transforms = np.empty((len(baselines), EEG_CHANNEL_COUNT, EEG_CHANNEL_COUNT))
    identity = np.eye(EEG_CHANNEL_COUNT, dtype=np.float64)
    for run_id in np.unique(run_ids):
        selected = run_ids == run_id
        inverse_sqrt = _run_reference_covariance(
            np.asarray(baselines[selected], dtype=np.float64),
            shrinkage,
            epsilon,
        )
        transforms[selected] = (1.0 - eta) * identity + eta * inverse_sqrt
    return transforms.astype(np.float32)


def alignment_transforms(
    train_baseline: np.ndarray,
    eval_baseline: np.ndarray,
    train_run_ids: np.ndarray,
    eval_run_ids: np.ndarray,
    protocol: str = "offline-transductive",
    shrinkage: float = 0.1,
    eta: float = 1.0,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one label-free run-local whitening transform per trial."""

    if protocol != "offline-transductive":
        raise ValueError("LYNet only defines the 'offline-transductive' reference protocol.")
    if not 0.0 <= shrinkage <= 1.0 or not 0.0 <= eta <= 1.0:
        raise ValueError("shrinkage and eta must both be in [0, 1].")
    return (
        _session_alignment_transforms(
            train_baseline,
            train_run_ids,
            shrinkage,
            eta,
            epsilon,
        ),
        _session_alignment_transforms(
            eval_baseline,
            eval_run_ids,
            shrinkage,
            eta,
            epsilon,
        ),
    )


def apply_transforms(transforms: np.ndarray, waveforms: np.ndarray) -> np.ndarray:
    if transforms.shape != (len(waveforms), EEG_CHANNEL_COUNT, EEG_CHANNEL_COUNT):
        raise ValueError(f"Unexpected transform shape {transforms.shape}.")
    if waveforms.ndim != 3 or waveforms.shape[1:] != (EEG_CHANNEL_COUNT, SAMPLES):
        raise ValueError(f"Unexpected waveform shape {waveforms.shape}.")
    aligned = np.einsum("nij,njt->nit", transforms, waveforms, optimize=True)
    return np.asarray(aligned, dtype=np.float32)


def log_band_power_features(
    waveforms: np.ndarray,
    *,
    sampling_rate: float = 250.0,
    bands_hz: tuple[tuple[float, float], ...] = POWER_BANDS_HZ,
    windows: int = POWER_WINDOWS,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return window-major, band-major, channel-major log-periodogram power."""

    if waveforms.ndim != 3 or waveforms.shape[1:] != (EEG_CHANNEL_COUNT, SAMPLES):
        raise ValueError(f"Unexpected waveform shape {waveforms.shape}.")
    if SAMPLES % windows:
        raise ValueError(f"{SAMPLES} samples cannot be split into {windows} windows.")
    window_samples = SAMPLES // windows
    frequencies = np.fft.rfftfreq(window_samples, d=1.0 / sampling_rate)
    taper = np.hanning(window_samples).astype(np.float64)
    normalizer = float(np.square(taper).sum())
    features: list[np.ndarray] = []
    for window_index in range(windows):
        start = window_index * window_samples
        stop = start + window_samples
        segment = np.asarray(waveforms[:, :, start:stop], dtype=np.float64)
        segment = segment - segment.mean(axis=2, keepdims=True)
        spectrum = np.fft.rfft(segment * taper, axis=2)
        periodogram = np.square(np.abs(spectrum)) / normalizer
        for band_index, (low, high) in enumerate(bands_hz):
            if band_index == len(bands_hz) - 1:
                selected = (frequencies >= low) & (frequencies <= high)
            else:
                selected = (frequencies >= low) & (frequencies < high)
            if not selected.any():
                raise ValueError(f"Power band ({low}, {high}) contains no FFT bin.")
            band_power = periodogram[:, :, selected].mean(axis=2)
            features.append(np.log(np.maximum(band_power, epsilon)))
    result = np.stack(features, axis=1).reshape(len(waveforms), -1)
    if result.shape[1] != POWER_FEATURES:
        raise RuntimeError(f"Expected {POWER_FEATURES} power features, got {result.shape}.")
    return np.asarray(result, dtype=np.float32)


def _scaled_main(
    train_task: np.ndarray,
    eval_task: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    train_flat = train_task.reshape(len(train_task), -1)
    eval_flat = eval_task.reshape(len(eval_task), -1)
    if enabled:
        scaler = StandardScaler().fit(train_flat)
        train_flat = scaler.transform(train_flat)
        eval_flat = scaler.transform(eval_flat)
    shape = (-1, 1, EEG_CHANNEL_COUNT, SAMPLES)
    return (
        np.asarray(train_flat.reshape(shape), dtype=np.float32),
        np.asarray(eval_flat.reshape(shape), dtype=np.float32),
    )


def _scaled_power(
    train_waveform: np.ndarray,
    eval_waveform: np.ndarray,
    preprocessing: LYNetPreprocessing,
) -> tuple[np.ndarray, np.ndarray]:
    train = log_band_power_features(
        train_waveform,
        sampling_rate=preprocessing.sampling_rate,
        bands_hz=preprocessing.power_bands_hz,
        windows=preprocessing.power_windows,
    )
    evaluation = log_band_power_features(
        eval_waveform,
        sampling_rate=preprocessing.sampling_rate,
        bands_hz=preprocessing.power_bands_hz,
        windows=preprocessing.power_windows,
    )
    if preprocessing.power_standardize:
        scaler = StandardScaler().fit(train)
        train = scaler.transform(train)
        evaluation = scaler.transform(evaluation)
    return np.asarray(train, dtype=np.float32), np.asarray(evaluation, dtype=np.float32)


def _source_manifest(dataset_root: Path, subject: int) -> list[dict[str, object]]:
    subject_id = f"A{subject:02d}"
    manifest: list[dict[str, object]] = []
    for session in ("T", "E"):
        for folder, suffix in (("2a_gdf", ".gdf"), ("2a_label", ".mat")):
            path = (dataset_root / folder / f"{subject_id}{session}{suffix}").resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            stat = path.stat()
            manifest.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return manifest


def _cache_path(
    dataset_root: Path,
    cache_dir: str | Path,
    subject: int,
    preprocessing: LYNetPreprocessing,
) -> tuple[Path, dict[str, object]]:
    signature: dict[str, object] = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "subject": f"A{subject:02d}",
        "sources": _source_manifest(dataset_root, subject),
        "preprocessing": asdict(preprocessing),
        "alignment_protocol": "offline-transductive-run-local-baseline",
        "power_layout": "window-band-channel",
        "power_implementation": "Hann-windowed real-FFT mean periodogram",
    }
    # Canonical JSON types make the in-memory signature identical to the one
    # loaded back from cache metadata (tuples become lists here).
    signature = json.loads(json.dumps(signature, ensure_ascii=False, sort_keys=True))
    serialized = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir).resolve() / f"A{subject:02d}_{fingerprint}.npz", signature


def _validate_views(views: LYNetViews) -> None:
    expected_main = (TRIALS, 1, EEG_CHANNEL_COUNT, SAMPLES)
    expected_power = (TRIALS, POWER_FEATURES)
    if views.train_main.shape != expected_main or views.eval_main.shape != expected_main:
        raise ValueError(f"Unexpected LYNet main shapes {views.train_main.shape}/{views.eval_main.shape}.")
    if (
        views.train_aligned_main.shape != expected_main
        or views.eval_aligned_main.shape != expected_main
    ):
        raise ValueError(
            "Unexpected LYNet aligned main shapes "
            f"{views.train_aligned_main.shape}/{views.eval_aligned_main.shape}."
        )
    if views.train_power.shape != expected_power or views.eval_power.shape != expected_power:
        raise ValueError(
            f"Unexpected LYNet power shapes {views.train_power.shape}/{views.eval_power.shape}."
        )
    if views.train_labels.shape != (TRIALS,) or views.eval_labels.shape != (TRIALS,):
        raise ValueError("Unexpected LYNet label shapes.")
    arrays = (
        views.train_main,
        views.eval_main,
        views.train_aligned_main,
        views.eval_aligned_main,
        views.train_power,
        views.eval_power,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise FloatingPointError("Non-finite LYNet preprocessed values.")


def _load_cache(cache_path: Path, signature: dict[str, object]) -> LYNetViews | None:
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            if metadata.get("cache_signature") != signature:
                return None
            views = LYNetViews(
                train_main=np.asarray(cached["train_main"], dtype=np.float32),
                eval_main=np.asarray(cached["eval_main"], dtype=np.float32),
                train_aligned_main=np.asarray(
                    cached["train_aligned_main"], dtype=np.float32
                ),
                eval_aligned_main=np.asarray(
                    cached["eval_aligned_main"], dtype=np.float32
                ),
                train_power=np.asarray(cached["train_power"], dtype=np.float32),
                eval_power=np.asarray(cached["eval_power"], dtype=np.float32),
                train_labels=np.asarray(cached["train_labels"], dtype=np.int64),
                eval_labels=np.asarray(cached["eval_labels"], dtype=np.int64),
                metadata=metadata,
            )
        _validate_views(views)
        return views
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write_cache(cache_path: Path, views: LYNetViews) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                train_main=views.train_main,
                eval_main=views.eval_main,
                train_aligned_main=views.train_aligned_main,
                eval_aligned_main=views.eval_aligned_main,
                train_power=views.train_power,
                eval_power=views.eval_power,
                train_labels=views.train_labels,
                eval_labels=views.eval_labels,
                metadata=np.asarray(json.dumps(views.metadata, ensure_ascii=False, sort_keys=True)),
            )
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_lynet_views(
    dataset_root: str | Path,
    subject: int,
    preprocessing: LYNetPreprocessing | None = None,
    cache_dir: str | Path = "dataset/lynet_run_ea_cache",
    *,
    force: bool = False,
) -> LYNetViews:
    """Build or load the complete two-input LYNet view for one subject."""

    if subject not in range(1, 10):
        raise ValueError(f"BCI-2a subject must be 1..9, got {subject}.")
    preprocessing = preprocessing or LYNetPreprocessing()
    preprocessing.validate()
    dataset_root = Path(dataset_root).resolve()
    cache_path, signature = _cache_path(
        dataset_root,
        cache_dir,
        subject,
        preprocessing,
    )
    if cache_path.is_file() and not force:
        cached = _load_cache(cache_path, signature)
        if cached is not None:
            print(f"Loaded LYNet preprocessing cache: {cache_path}")
            return cached

    subject_id = f"A{subject:02d}"
    sessions: list[GDFSession] = []
    for session in ("T", "E"):
        sessions.append(
            load_gdf_session(
                dataset_root / "2a_gdf" / f"{subject_id}{session}.gdf",
                dataset_root / "2a_label" / f"{subject_id}{session}.mat",
                preprocessing,
            )
        )
    train, evaluation = sessions
    if train.channel_names != evaluation.channel_names:
        raise ValueError(f"Channel-order mismatch for {subject_id}.")

    train_main, eval_main = _scaled_main(
        train.main_task,
        evaluation.main_task,
        preprocessing.main_standardize,
    )
    train_transforms, eval_transforms = alignment_transforms(
        train.baseline,
        evaluation.baseline,
        train.run_ids,
        evaluation.run_ids,
        "offline-transductive",
        preprocessing.shrinkage,
        preprocessing.eta,
        preprocessing.covariance_epsilon,
    )
    train_aligned = apply_transforms(train_transforms, train.physical_task)
    eval_aligned = apply_transforms(eval_transforms, evaluation.physical_task)
    train_aligned_main, eval_aligned_main = _scaled_main(
        train_aligned,
        eval_aligned,
        preprocessing.main_standardize,
    )
    train_power, eval_power = _scaled_power(train_aligned, eval_aligned, preprocessing)

    metadata: dict[str, object] = {
        "subject": subject_id,
        "model": "LYNet",
        "source_variant": "RUN-EA",
        "reference": "run-local pre-cue baseline covariance",
        "reference_protocol": "offline-transductive",
        "reference_uses_labels": False,
        "evaluation_reference_scope": "all 48 unlabelled baselines in each E-session run",
        "main_input_changed_by_alignment": False,
        "additional_main_view": "standardized aligned task EEG",
        "train_artifact_marked_trials": int(train.artifact_flags.sum()),
        "eval_artifact_marked_trials": int(evaluation.artifact_flags.sum()),
        "cache_signature": signature,
        "cache_path": str(cache_path),
    }
    views = LYNetViews(
        train_main=train_main,
        eval_main=eval_main,
        train_aligned_main=train_aligned_main,
        eval_aligned_main=eval_aligned_main,
        train_power=train_power,
        eval_power=eval_power,
        train_labels=train.labels.astype(np.int64, copy=False),
        eval_labels=evaluation.labels.astype(np.int64, copy=False),
        metadata=metadata,
    )
    _validate_views(views)
    _write_cache(cache_path, views)
    print(f"Built LYNet preprocessing cache: {cache_path}")
    return views


__all__ = [
    "EEG_CHANNELS",
    "GDFSession",
    "LYNetPreprocessing",
    "LYNetViews",
    "POWER_BANDS_HZ",
    "POWER_FEATURES",
    "SAMPLES",
    "TRIALS",
    "alignment_transforms",
    "apply_transforms",
    "load_gdf_session",
    "log_band_power_features",
    "prepare_lynet_views",
]
