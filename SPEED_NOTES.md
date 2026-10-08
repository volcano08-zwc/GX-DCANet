# V121_FastFP32

Direct copy of V121, with FP32 execution optimizations; seed 0, 500 epochs, batch 16.
config/lynet_2a.yaml selects A01; config/lynet_2a_full.yaml selects A01-A09.
The completed experiments used physical GPU1.
No AMP, changed losses, early stopping, merged CNN views or changed preprocessing.

Execution changes:
1. Cache trial-local mean/std/eigenvectors/intrinsic projections before stochastic augmentation.
2. Vectorize component selection while preserving CUDA random draw order.
3. Keep input tensors and training decomposition cache resident on the GPU.
4. Aggregate diagnostics on GPU and check all gradients with one host synchronization per batch.
5. Build each learned Sinc kernel once per paired forward, retaining its autograd graph.

Shared CNN branches still execute separately, preserving BatchNorm and dropout behavior.
Sinc kernel reuse can change floating-point gradient addition order slightly; FP32 does
not guarantee a bitwise-identical 500-epoch trajectory. Accuracy must be compared empirically.

Diagnostics CSV records train_seconds/eval_seconds; preparation cost is separately
recorded in cache_timing.json. Exclude warm-up epochs when comparing steady-state speed.

Training commands (run from this version's directory):
    python train_lynet_2a.py --config config/lynet_2a.yaml
    python train_lynet_2a.py --config config/lynet_2a_full.yaml

Both configs currently reuse ../V121/dataset/cache_v121 and ../BCI_2a.
GPU selection is controlled by the launcher environment; device_id=0 is the first
visible GPU, not necessarily physical GPU0.

The completed full run reached mean Best-E accuracy 86.9213% and mean kappa 0.8256,
equal to the original V121 means, with small per-subject differences. Wall time was
59 minutes 43 seconds. This is exploratory evaluation-session checkpoint selection.
