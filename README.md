# 🐻 Bear Classifier

A machine learning project that classifies images of bears into three categories: **Black Bear**, **Grizzly Bear**, and **Teddy Bear**. This project uses FastAI and is deployed on Hugging Face Spaces.

## 🌐 Live Demo

**Try the Bear Classifier:** [https://huggingface.co/spaces/mr-1o1/bear-me](https://huggingface.co/spaces/mr-1o1/bear-me)

## 📋 Project Overview

This project demonstrates how to build and deploy an image classification model using:
- **FastAI** - Deep learning framework
- **Gradio** - Web interface for ML models
- **Hugging Face Spaces** - Model deployment platform

The classifier can distinguish between:
- 🐻 **Black Bears** - Real black bears in the wild
- 🐻 **Grizzly Bears** - Real grizzly/brown bears in the wild  
- 🧸 **Teddy Bears** - Stuffed toy bears

## 🏗️ Project Structure

```
bear-classifier/
├── app.py                 # Gradio web interface
├── bear-me.ipynb         # Jupyter notebook with model training
├── bear_classifier.pkl   # Trained model file
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── LICENSE              # Apache 2.0 license
├── .gitignore           # Git ignore rules
└── bears/               # Training dataset
    ├── black/           # Black bear images
    ├── grizzly/         # Grizzly bear images
    └── teddy/           # Teddy bear images
```

## 🚀 How It Works

### 1. **Data Collection & Organization**
The project uses a curated dataset of bear images organized into three categories:
- **Black Bears**: ~100 images of real black bears
- **Grizzly Bears**: ~100 images of real grizzly/brown bears  
- **Teddy Bears**: ~100 images of stuffed toy bears

### 2. **Model Training**
The model is trained using FastAI's vision library with:
- **Transfer Learning**: Uses a pre-trained ResNet model
- **Data Augmentation**: Applies random transformations to improve generalization
- **Fine-tuning**: Adapts the pre-trained model to the specific bear classification task

### 3. **Web Interface**
The Gradio interface provides:
- **Image Upload**: Users can upload bear images
- **Real-time Classification**: Instant predictions with confidence scores
- **Example Images**: Pre-loaded examples for testing
- **Responsive Design**: Works on desktop and mobile devices

### 4. **Deployment**
The model is deployed on Hugging Face Spaces, providing:
- **Free Hosting**: No server costs
- **Automatic Scaling**: Handles multiple users
- **Easy Updates**: Simple deployment process

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mr-1o1/bear-me.git
   cd bear-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application locally**
   ```bash
   python app.py
   ```

4. **Access the web interface**
   Open your browser and go to `http://localhost:7860`

### Model Training (Optional)

If you want to retrain the model:

1. **Prepare your dataset**
   - Organize images into folders: `bears/black/`, `bears/grizzly/`, `bears/teddy/`
   - Ensure each category has sufficient images (recommended: 50+ per category)

2. **Run the training notebook**
   ```bash
   jupyter notebook bear-me.ipynb
   ```

3. **Export the trained model**
   - The notebook will save the model as `bear_classifier.pkl`
   - Update `app.py` to use the new model file

## 📊 Model Performance

The classifier achieves:
- **High Accuracy**: Successfully distinguishes between real bears and teddy bears
- **Fast Inference**: Real-time predictions on uploaded images
- **Robust Performance**: Works with various image qualities and angles

## 🔧 Technical Details

### Dependencies
- `fastai`: Deep learning framework
- `gradio`: Web interface library
- `nbdev`: Development tools for Jupyter notebooks

### Model Architecture
- **Base Model**: ResNet (pre-trained on ImageNet)
- **Training Method**: Transfer learning with fine-tuning
- **Output**: 3-class classification with confidence scores

### File Descriptions
- `app.py`: Main application file with Gradio interface
- `bear-me.ipynb`: Jupyter notebook containing model training code
- `bear_classifier.pkl`: Serialized trained model
- `requirements.txt`: Python package dependencies

## 🎯 Usage Examples

### Using the Web Interface
1. Visit [https://huggingface.co/spaces/mr-1o1/bear-me](https://huggingface.co/spaces/mr-1o1/bear-me)
2. Upload an image of a bear (real or teddy)
3. View the classification results with confidence scores

### Programmatic Usage
```python
from fastai.vision.all import *

# Load the model
learn = load_learner('bear_classifier.pkl')

# Classify an image
img = PILImage.create('path/to/bear/image.jpg')
pred, pred_idx, probs = learn.predict(img)

print(f"Prediction: {pred}")
print(f"Confidence: {probs[pred_idx]:.2%}")
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Improve the model**: Add more training data or experiment with different architectures
2. **Enhance the UI**: Improve the Gradio interface design
3. **Add features**: Implement new functionality like batch processing
4. **Fix bugs**: Report and fix any issues you encounter

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAI** for the excellent deep learning framework
- **Gradio** for the user-friendly web interface
- **Hugging Face** for providing free model hosting
- **Image contributors** for the training dataset

## 📞 Contact

For questions or support, please open an issue on the project repository.

---

**Note**: This project is for educational and demonstration purposes. The model may not be suitable for production use without additional validation and testing.
