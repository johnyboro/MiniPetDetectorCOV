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

Quick debug run with subset and no augmentation:

```bash
python train.py --config configs/lenet_search_base.yaml
```

Set `data.subset_fraction` (for example `0.1`) and `augment.train.type` (`none`, `basic`, `randaugment`) in the config.



