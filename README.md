# 🎨 Neural Style Transfer using VGG19

## 📌 Task 05 - Prodigy InfoTech Generative AI Internship

This project demonstrates **Neural Style Transfer (NST)** using a pre-trained **VGG19** model in PyTorch. Neural Style Transfer combines the **content** of one image with the **artistic style** of another image to generate a visually appealing stylized output.

---

## 📖 Project Overview

The project uses the optimization-based Neural Style Transfer algorithm introduced by Gatys et al. It extracts deep feature representations from a pre-trained VGG19 network to preserve the content of one image while transferring the artistic style of another.

---

## 🚀 Features

* Uses a pre-trained **VGG19** model.
* Extracts content and style features from multiple convolutional layers.
* Computes Gram matrices for style representation.
* Optimizes the generated image using the LBFGS optimizer.
* Automatically detects and uses GPU (CUDA) if available.
* Saves the final stylized image as `output.jpg`.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* Pillow
* VGG19 (Pre-trained CNN)

---

## 📂 Project Structure

```text
PRODIGY_GA_05/
│── task5.py
│── content.jpg
│── style.jpg
│── output.jpg
│── README.md
```

---

## ⚙️ Installation

Install the required libraries:

```bash
pip install torch torchvision pillow
```

---

## ▶️ How to Run

1. Place `content.jpg` and `style.jpg` in the project directory.
2. Run the following command:

```bash
python task5.py
```

3. After execution, the generated stylized image will be saved as:

```text
output.jpg
```

---

## 📸 Input Images

* **Content Image:** Photograph whose structure is preserved.
* **Style Image:** Artwork or painting whose artistic style is transferred.

---

## 🖼️ Output

The generated image combines the content of the original photograph with the artistic style of the selected painting.

---

## 📚 Learning Outcomes

* Understanding Convolutional Neural Networks (CNNs)
* Feature extraction using VGG19
* Gram Matrix computation for style representation
* Content Loss and Style Loss
* Optimization-based image generation
* Neural Style Transfer using Deep Learning

---

## 🙏 Acknowledgements

* Prodigy InfoTech
* PyTorch
* torchvision
* VGG19 Pre-trained Model
* Gatys et al., "A Neural Algorithm of Artistic Style"

---

## 👨‍💻 Author

**Abir Singh**

GitHub: https://github.com/AbirSingh01