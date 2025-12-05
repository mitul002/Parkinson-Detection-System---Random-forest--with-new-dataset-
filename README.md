# Parkinson's Disease Detection System

An AI-powered web application that analyzes hand-drawn spiral images to detect signs of Parkinson's disease using machine learning.

## Features

- **Hybrid Feature Extraction**: Combines 512 CNN features (ResNet18) with 40+ handcrafted geometric features
- **Random Forest Classification**: Accurate multi-class detection (Healthy/Parkinson/Noisy)
- **Explainable AI**: SHAP and LIME interpretations for transparent predictions
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Batch Processing**: Analyze multiple images simultaneously
- **Real-time Analysis**: Instant predictions with confidence scores

## Technologies Used

- **Python 3.13**
- **Machine Learning**: Scikit-learn, Random Forest
- **Deep Learning**: PyTorch, Torchvision (ResNet18)
- **Computer Vision**: OpenCV, scikit-image
- **Explainability**: SHAP, LIME
- **Web Framework**: Streamlit
- **Data Processing**: NumPy, Pandas, SciPy

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "Parkinson Detection System - Random forest (with new dataset)"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (optional - pre-trained model included):
```bash
python train_model.py
```

## Usage

Run the application:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Project Structure

```
├── app.py                      # Main Streamlit application
├── feature_extraction.py       # Feature extraction pipeline
├── train_model.py             # Model training script
├── generate_dataset.py        # Dataset generation
├── requirements.txt           # Python dependencies
├── *.pkl                      # Trained models and preprocessors
└── DATASET/                   # Training images (not included)
    ├── Healthy/
    ├── Parkinson/
    └── Noisy/
```

## Model Details

- **Algorithm**: Random Forest Classifier (200 estimators)
- **Features**: 552 total (512 CNN + 40 handcrafted)
- **Classes**: Healthy, Parkinson, Noisy
- **Preprocessing**: StandardScaler, SimpleImputer

## Features Analyzed

### Handcrafted Features
- Motion analysis (jerkiness, wobble, tremor patterns)
- Geometric metrics (loop variation, spiral fit, curvature)
- Image quality (entropy, stroke width, fractal dimension)
- Morphological features (skeleton analysis, branch points)

### CNN Features
- Deep features extracted from ResNet18 pretrained network

## License

Educational project for demonstration purposes.

## Author

Developed as part of a machine learning healthcare project.
