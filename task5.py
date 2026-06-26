"""
Task 05 - Neural Style Transfer using VGG19
============================================
Applies the artistic style of one image to the content of another
using optimization-based Neural Style Transfer (Gatys et al., 2015).

Requirements:
    pip install torch torchvision pillow

Usage:
    Place content.jpg and style.jpg in the same directory, then run:
        python task5.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
from PIL import Image
import copy

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — tweak these to change the output
# ──────────────────────────────────────────────────────────────────────────────
CONTENT_IMAGE_PATH = "content.jpg"
STYLE_IMAGE_PATH   = "style.jpg"
OUTPUT_IMAGE_PATH  = "output.jpg"

IMAGE_SIZE     = 384          # Both images are resized to this (pixels)
CONTENT_WEIGHT = 1            # α — how much to preserve content structure
STYLE_WEIGHT   = 1e7          # β — how much to apply artistic style
NUM_STEPS      = 500          # Optimization iterations
LOG_EVERY      = 25           # Print loss every N steps

# VGG19 layers used for feature extraction
# conv4_2 is standard for content; these five layers capture style well
CONTENT_LAYERS = ["conv4_2"]
STYLE_LAYERS   = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

# ──────────────────────────────────────────────────────────────────────────────
# DEVICE SETUP
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

# ──────────────────────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

# VGG19 was trained on ImageNet with these normalisation values
_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device)
_STD  = torch.tensor([0.229, 0.224, 0.225]).to(device)

_loader = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def load_image(path: str) -> torch.Tensor:
    """Load an image from disk, resize it, and return a (1, 3, H, W) tensor."""
    img = Image.open(path).convert("RGB")
    tensor = _loader(img).unsqueeze(0)   # add batch dimension
    return tensor.to(device, torch.float)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, 3, H, W) tensor back to a PIL image (clamps to [0, 1])."""
    img = tensor.cpu().clone().squeeze(0)
    img = img.clamp(0, 1)
    return transforms.ToPILImage()(img)


def normalise(tensor: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalisation expected by VGG19."""
    return (tensor - _MEAN[None, :, None, None]) / _STD[None, :, None, None]


# ──────────────────────────────────────────────────────────────────────────────
# GRAM MATRIX — captures texture/style by measuring feature correlations
# ──────────────────────────────────────────────────────────────────────────────

def gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gram matrix of a feature map.

    Args:
        feature_map: shape (1, C, H, W)

    Returns:
        Normalised Gram matrix of shape (C, C)
    """
    _, C, H, W = feature_map.size()
    # Reshape to (C, H*W) so each row is one feature channel
    features = feature_map.view(C, H * W)
    G = torch.mm(features, features.t())   # (C, C) inner-product matrix
    # Normalise by total number of elements so scale stays comparable
    return G / (C * H * W)


# ──────────────────────────────────────────────────────────────────────────────
# VGG19 FEATURE EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

# Map human-readable layer names to VGG19 sequential indices
# VGG19 feature layers:  0  2  5  7 10 12 14 16 19 21 23 25 28 30 32 34 36
# Name mapping:       conv1_1 conv1_2 conv2_1 conv2_2 conv3_1 …
_LAYER_MAP = {
    "conv1_1":  "0",
    "conv1_2":  "2",
    "conv2_1":  "5",
    "conv2_2":  "7",
    "conv3_1":  "10",
    "conv3_2":  "12",
    "conv3_3":  "14",
    "conv3_4":  "16",
    "conv4_1":  "19",
    "conv4_2":  "21",
    "conv4_3":  "23",
    "conv4_4":  "25",
    "conv5_1":  "28",
    "conv5_2":  "30",
    "conv5_3":  "32",
    "conv5_4":  "34",
}

# All layers we need to hook (union of content + style layers)
_ALL_NEEDED = set(CONTENT_LAYERS) | set(STYLE_LAYERS)
_NEEDED_INDICES = {_LAYER_MAP[n] for n in _ALL_NEEDED}


class VGG19FeatureExtractor(nn.Module):
    """Wraps VGG19 and returns named intermediate feature maps."""

    def __init__(self):
        super().__init__()
        # Load pretrained VGG19 and keep only the convolutional feature layers
        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        # Freeze weights — we only optimise the generated image, not the network
        for param in vgg.parameters():
            param.requires_grad_(False)
        self.layers = vgg.to(device)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Pass x through VGG19 and collect activations at the named layers.

        Returns:
            dict mapping layer-name → feature tensor
        """
        features = {}
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            idx_str = str(idx)
            if idx_str in _NEEDED_INDICES:
                # Reverse lookup: index → name
                name = next(k for k, v in _LAYER_MAP.items() if v == idx_str)
                features[name] = x
            # Stop early once we have all needed layers (saves compute)
            if len(features) == len(_ALL_NEEDED):
                break
        return features


# ──────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def content_loss(generated_features: dict, content_features: dict) -> torch.Tensor:
    """MSE between generated and content feature maps at content layers."""
    loss = torch.tensor(0.0, device=device)
    for layer in CONTENT_LAYERS:
        loss = loss + torch.mean((generated_features[layer] - content_features[layer]) ** 2)
    return loss


def style_loss(generated_features: dict, style_grams: dict) -> torch.Tensor:
    """MSE between Gram matrices of generated and style images at style layers."""
    loss = torch.tensor(0.0, device=device)
    for layer in STYLE_LAYERS:
        G_gen   = gram_matrix(generated_features[layer])
        G_style = style_grams[layer]
        loss = loss + torch.mean((G_gen - G_style) ** 2)
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# MAIN STYLE TRANSFER ROUTINE
# ──────────────────────────────────────────────────────────────────────────────

def run_style_transfer() -> None:
    print("[Step 1/4] Loading images …")
    content_tensor = load_image(CONTENT_IMAGE_PATH)
    style_tensor   = load_image(STYLE_IMAGE_PATH)

    print("[Step 2/4] Extracting reference features from VGG19 …")
    extractor = VGG19FeatureExtractor()

    # Pre-compute target features (fixed throughout optimisation)
    with torch.no_grad():
        content_features = extractor(normalise(content_tensor))
        style_features   = extractor(normalise(style_tensor))

    # Pre-compute target Gram matrices for the style image
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in STYLE_LAYERS}

    # Initialise the generated image from the content image
    # (alternatively, start from random noise for a more abstract result)
    generated = content_tensor.clone().requires_grad_(True)

    print("[Step 3/4] Optimising generated image …")
    # LBFGS converges faster than Adam for style transfer
    optimizer = optim.LBFGS([generated], lr=1.0, max_iter=20)

    step = [0]  # mutable counter accessible inside closure

    while step[0] < NUM_STEPS:

        def closure():
            # Keep pixel values in [0, 1]
            with torch.no_grad():
                generated.clamp_(0, 1)

            optimizer.zero_grad()

            gen_features = extractor(normalise(generated))

            c_loss = CONTENT_WEIGHT * content_loss(gen_features, content_features)
            s_loss = STYLE_WEIGHT   * style_loss(gen_features, style_grams)
            total  = c_loss + s_loss
            total.backward()

            step[0] += 1
            if step[0] % LOG_EVERY == 0 or step[0] == 1:
                print(
                    f"  Step {step[0]:>4}/{NUM_STEPS} | "
                    f"Content Loss: {c_loss.item():.4f} | "
                    f"Style Loss: {s_loss.item():.2f} | "
                    f"Total: {total.item():.2f}"
                )
            return total

        optimizer.step(closure)

    # Final clamp to ensure valid pixel range
    with torch.no_grad():
        generated.clamp_(0, 1)

    print(f"[Step 4/4] Saving output to '{OUTPUT_IMAGE_PATH}' …")
    output_image = tensor_to_pil(generated)
    output_image.save(OUTPUT_IMAGE_PATH)
    print("Done! ✓")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_style_transfer()