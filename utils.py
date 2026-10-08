import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml


def _load_torch_npu():
    """Import torch_npu only when an NPU run is requested."""
    try:
        import torch_npu
    except Exception as exc:
        raise RuntimeError(
            "preferred_device='npu' requires a working torch_npu installation "
            'compatible with the installed PyTorch and CANN versions.'
        ) from exc
    return torch_npu


def _get_device_id(config):
    """Read the new device_id key while accepting the legacy nGPU key."""
    value = config.get('device_id', config.get('nGPU', 0))
    try:
        device_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'device_id must be a non-negative integer, got {value!r}.') from exc

    if device_id < 0:
        raise ValueError(f'device_id must be non-negative, got {device_id}.')
    return device_id


def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def dict_to_yaml(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def set_random_seed(config):
    """Set RNG/cuDNN behavior from YAML.

    Defaults reproduce the original DCANet script's fast, non-deterministic
    CUDA behavior rather than CSANet's deterministic settings.
    """
    seed = config['random_seed']

    os.environ['PYTHONHASHSEED'] = str(seed)
    if config.get('use_deterministic_algorithms', False):
        # Required by CUDA >= 10.2 when deterministic CuBLAS operations are
        # requested. This matches the supplied LYNet/RUN-EA seed setup.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    preferred_device = str(config.get('preferred_device', 'cpu')).lower()
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if preferred_device in {'npu', 'ascend'}:
        torch_npu = _load_torch_npu()
        if torch_npu.npu.is_available():
            torch_npu.npu.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = config['deterministic']
    torch.backends.cudnn.benchmark = config['cudnn_benchmark']
    torch.use_deterministic_algorithms(config['use_deterministic_algorithms'])


def seed_worker(worker_id, base_seed):
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_seed_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def kaiming_init_weights(module):
    """Original DCANet initialization strategy."""
    if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(
            module.weight,
            mode='fan_in',
            nonlinearity='leaky_relu',
        )
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(config):
    """Select an explicitly configured CPU, CUDA GPU or Ascend NPU."""
    preferred_device = str(config.get('preferred_device', 'cpu')).lower()
    device_id = _get_device_id(config)

    if preferred_device == 'cpu':
        return torch.device('cpu')

    if preferred_device in {'gpu', 'cuda'}:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"preferred_device={preferred_device!r}, but CUDA is not available."
            )
        if device_id >= torch.cuda.device_count():
            raise ValueError(
                f'CUDA device_id={device_id} is out of range; '
                f'{torch.cuda.device_count()} device(s) are visible.'
            )
        torch.cuda.set_device(device_id)
        return torch.device(f'cuda:{device_id}')

    if preferred_device in {'npu', 'ascend'}:
        torch_npu = _load_torch_npu()
        if not torch_npu.npu.is_available():
            raise RuntimeError(
                f"preferred_device={preferred_device!r}, but no Ascend NPU is available."
            )
        if device_id >= torch_npu.npu.device_count():
            raise ValueError(
                f'NPU device_id={device_id} is out of range; '
                f'{torch_npu.npu.device_count()} device(s) are visible.'
            )
        torch_npu.npu.set_device(device_id)
        return torch.device(f'npu:{device_id}')

    raise ValueError(
        f'Unsupported preferred_device={preferred_device!r}; '
        "expected 'cpu', 'gpu'/'cuda', or 'npu'/'ascend'."
    )


def validate_config(config):
    """Validate only constraints already implied by the current model/data."""
    from model.registry import model_args_from_config, validate_model_data_contract

    network_name = str(config['network'])
    args = model_args_from_config(config)

    preferred_device = str(config.get('preferred_device', 'cpu')).lower()
    supported_devices = {'cpu', 'gpu', 'cuda', 'npu', 'ascend'}
    if preferred_device not in supported_devices:
        raise ValueError(
            f'Unsupported preferred_device={preferred_device!r}; '
            f'expected one of {sorted(supported_devices)}.'
        )
    _get_device_id(config)

    validate_model_data_contract(
        network_name,
        args,
        channels=int(config['channels']),
        samples=int(config['samples']),
        num_classes=4,
    )

    if 'sampling_rate' in args and args['sampling_rate'] != config['sampling_rate']:
        raise ValueError(
            f"network_args.sampling_rate={args['sampling_rate']} but "
            f"sampling_rate={config['sampling_rate']}"
        )
    if network_name == 'DCANetV5':
        backend = str(args.get('spectral_backend', 'sinc')).lower()
        if backend not in {'sinc', 'fixed', 'fft'}:
            raise ValueError(
                "DCANetV5 spectral_backend must be 'sinc', 'fixed', or 'fft'."
            )
    if network_name.startswith('DCANet') and args['F_filters'] != 8:
        raise ValueError(
            'Current TimesBlock implementation requires F_filters=8 because '
            'its internal channel contract is fixed at 8.'
        )
    if config['filter_enabled']:
        if not 0 < config['low_freq'] < config['high_freq'] < config['sampling_rate'] / 2:
            raise ValueError('Invalid band-pass frequencies for the configured sampling rate.')
