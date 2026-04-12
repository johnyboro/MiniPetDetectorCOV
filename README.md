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

To run two sweep agents on two GPUs, start two processes with different visible devices:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/base.yaml --sweep configs/sweeps/lenet5_bayes.yaml --count 20
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/base.yaml --sweep configs/sweeps/lenet5_bayes.yaml --count 20
```
