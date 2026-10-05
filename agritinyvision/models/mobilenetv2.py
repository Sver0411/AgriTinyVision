from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build(config, download_weights=True):
    pretrained = config["pretrained"] and download_weights
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights, width_mult=config["width_mult"])
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    # Preserve torchvision ReLU6; no silent architecture modification during quantization.
    return model
