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


Dataset stats (calculated on a fresh run then cached):

```bash
Calculating dataset statistics...
class percentages:  tensor([2.8406, 2.5004, 2.7556, 2.6535, 2.7045, 2.8066, 2.7386, 2.7045, 2.6875,
        2.6025, 2.6195, 2.6535, 2.5855, 2.7386, 2.8066, 2.6365, 2.6535, 2.8236,
        2.7556, 2.8236, 2.6025, 2.8066, 2.5855, 2.5515, 2.7045, 2.8746, 2.6195,
        2.7726, 2.6705, 2.7045, 2.7556, 2.6875, 2.7556, 2.7216, 2.6025, 2.6535,
        2.8406])
Mean: tensor([0.4811, 0.4487, 0.3953]), Std: tensor([0.2642, 0.2595, 0.2679])

```
