import torch
from torch import nn
from torchvision.models import convnext_tiny, efficientnet_b0


class LeNet5Like(nn.Module):
    def __init__(self, num_classes: int = 37):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class LeNetSearchLike(nn.Module):
    def __init__(
        self,
        num_classes: int = 37,
        conv_channels=(16, 32, 64),
        kernel_size: int = 5,
        use_batch_norm: bool = False,
        dropout: float = 0.2,
        fc_hidden_dim: int = 128,
        activation: str = "relu",
    ):
        super().__init__()

        if activation == "relu":
            activation_layer = nn.ReLU
        elif activation == "gelu":
            activation_layer = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        channels = [3, *conv_channels]
        feature_layers = []
        for idx in range(len(channels) - 1):
            feature_layers.append(
                nn.Conv2d(channels[idx], channels[idx + 1], kernel_size=kernel_size)
            )
            if use_batch_norm:
                feature_layers.append(nn.BatchNorm2d(channels[idx + 1]))
            feature_layers.append(activation_layer())
            feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        feature_layers.append(nn.AdaptiveAvgPool2d((5, 5)))
        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[-1] * 5 * 5, fc_hidden_dim),
            activation_layer(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(
    name: str, num_classes: int = 37, params: dict | None = None
) -> nn.Module:
    params = params or {}
    if name == "lenet5_like":
        return LeNet5Like(num_classes=num_classes)
    if name == "lenet_search_like":
        return LeNetSearchLike(num_classes=num_classes, **params)
    if name == "convnext_tiny":
        model = convnext_tiny(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        dropout = params.get("dropout", 0.2)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        return model
    raise ValueError(f"Unknown model architecture: {name}")
