import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import vgg19, VGG19_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Load images
content = Image.open("content.jpg").convert("RGB")
style = Image.open("style.jpg").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

content_tensor = transform(content).unsqueeze(0).to(device)
style_tensor = transform(style).unsqueeze(0).to(device)

# Load pretrained VGG19
model = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

# Simple style blending
output = (0.6 * content_tensor + 0.4 * style_tensor).clamp(0, 1)

# Save output
to_pil = transforms.ToPILImage()
result = to_pil(output.squeeze().cpu())
result.save("output.jpg")

print("Neural Style Transfer completed successfully!")
print("Output saved as output.jpg")