# MiniPetDetectorCOV
A simple pytorch convolutional pet detector.

## Training

Single run:

```bash
python train.py --config configs/base.yaml
```

W&B sweep run:

```bash
python train.py --config configs/base.yaml --sweep configs/sweeps/lenet5_bayes.yaml --count 20
```

Searchable LeNet-style model single run:

```bash
python train.py --config configs/lenet_search_base.yaml
```

Searchable LeNet-style architecture sweep:

```bash
python train.py --config configs/lenet_search_base.yaml --sweep configs/sweeps/lenet_search_bayes.yaml --count 20
```

ConvNeXt-Tiny baseline run:

```bash
python train.py --config configs/convnext_tiny_base.yaml
```

EfficientNet-B0 baseline run:

```bash
python train.py --config configs/efficientnet_b0_base.yaml
```

Launch two sweep agents in background (survives SSH disconnect):

```bash
./launch_dual_agents.sh --config configs/base.yaml --sweep configs/sweeps/lenet5_bayes.yaml --count 20
```

Logs are written to `logs/agent_gpu0_<timestamp>.log` and `logs/agent_gpu1_<timestamp>.log`.

To run two sweep agents on two GPUs, start two processes with different visible devices:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/base.yaml --sweep configs/sweeps/lenet5_bayes.yaml --count 20
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/base.yaml --sweep configs/sweeps/lenet5_bayes.yaml --count 20
```
