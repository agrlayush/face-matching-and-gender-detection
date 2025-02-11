# Face Matching & Gender Detection

## Overview
This Python script processes up to **10 images** to:
1. **Compare faces** and identify images that do not match the majority.
2. **Predict gender** for the person in each image.

The solution uses:
- **face_recognition** for face detection and comparison.
- **DeepFace** for gender classification.
- **OpenCV** for image processing.

It is optimized for both **CPU and GPU** environments.

---

## 🚀 Setup Instructions
### 1. **Clone the Repository**
```bash
git clone https://github.com/agrlayush/face-matching-and-gender-detection.git
cd face-matching-and-gender-detection
```

### 2. **Create a Virtual Environment (Optional but Recommended)**
```bash
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate    # For Windows
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Ensure Compatible TensorFlow Version**
```bash
pip uninstall tensorflow keras -y
pip install tensorflow==2.10 keras==2.10
```

---

## Input Images
- Place up to **10 images** inside a folder (e.g., `images/`).
- Supported formats: `.jpg`, `.png`

---

## Running the Script
```bash
python script.py
```

### Example Output:
```
Processing 5 images...
Face Matching Results:
Non-matching images: ['images/person2.jpg']
Gender Prediction:
person1.jpg → Male
person2.jpg → Female
```

---

## Troubleshooting
### "ImportError: cannot import name 'LocallyConnected2D'"
Fix:
```bash
pip uninstall tensorflow keras -y
pip install tensorflow==2.10 keras==2.10
```

### "No face found in image"
- Ensure images contain **clear, front-facing** faces.

### "CUDA Error (GPU users)"
- Install compatible CUDA & cuDNN:
```bash
pip install tensorflow-gpu==2.10
```

---

## License
This project is open-source under the **MIT License**.

---

## Contributing
Pull requests are welcome! If you find issues or have feature suggestions, open an issue on GitHub.

