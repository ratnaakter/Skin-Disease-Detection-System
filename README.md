# Skin Disease Detection System

## Comparative Analysis of Machine Learning Models for Robust Skin Lesion Classification

![Skin Disease Detection](https://via.placeholder.com/800x400?text=Skin+Disease+Detection+System)

## Project Overview

To classify skin lesions into multiple classes, this project applies and contrasts various machine learning algorithms. By utilizing computer vision and machine learning techniques, the system aims to support the early diagnosis and classification of skin illnesses.

# Aim  
To develop and compare different machine learning models for multi-class skin lesion classification, incorporating data augmentation and sensitivity analysis to improve robustness and generalizability.

# Objectives  
- Collect and preprocess publicly available skin lesion dataset.  
- Train and evaluate machine learning models such as Random Forest, SVM, XGBoost, CNNs, ResNet, and EfficientNet.  
- Assess the impact of data augmentation techniques (rotation, flipping, contrast adjustments, noise injection) on model robustness.  
- Perform sensitivity analysis to evaluate the effects of dataset size, augmentation intensity, and hyperparameter tuning on classification accuracy.  
- Develop a web application for real-time image classification.


### Disease Categories
The system can detect and classify the following skin conditions:
- Actinic keratoses
- Basal cell carcinoma
- Benign keratosis-like lesions
- Chickenpox
- Cowpox
- Dermatofibroma
- Healthy
- HFMD (Hand, Foot, and Mouth Disease)
- Measles
- Melanocytic nevi
- Melanoma
- Monkeypox
- Squamous cell carcinoma
- Vascular lesions

## Technologies Used

- **Backend:** Python, Django
- **Machine Learning:** Scikit-learn, TensorFlow, XGBoost
- **Image Processing:** OpenCV
- **Development Tools:** Git, Virtual Environment

## Machine Learning Models

The project implements and compares the following models:
- Traditional ML: SVM, Random Forest, XGBoost
- Deep Learning: CNN, MobileNet, EfficientNet
- Hybrid approaches

## Key Features

- Multiple model comparison for skin lesion classification
- Feature extraction using color histograms and deep learning techniques
- Data augmentation techniques to improve model robustness
- Comprehensive evaluation metrics (accuracy, precision, recall, F1 score, sensitivity, specificity)
- Real-time prediction capabilities with confidence scores
- Web interface for easy interaction and result visualization

## Installation and Setup

1. **Clone the repository**
   ```bash
   git clone 'repository link'
   cd skin-disease-detection
   ```

2. **Create and activate a virtual environment**
   ```bash Use
   Use python 3.9.0 version
   --------------------------
   python -m venv venv
   ```
   
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run database migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Start the development server**
   ```bash
   python manage.py runserver
   ```

6. **Access the application**
   Open your web browser and go to: http://127.0.0.1:8000/

## Project Structure

```
skin-disease-detection/
├── ML_Models/                # Machine learning model training module/ APP
├── dataset/                  # Dataset folder
│   └── train/                # Training images organized by category
├── config/                    # Main Django application and settings
├── static/                   # Static files (CSS, JS, images)
├── templates/                # HTML templates
├── manage.py                 # Django management script
└── requirements.txt          # Project dependencies
```

## Usage

1. Navigate to the web interface at http://127.0.0.1:8000/
2. Upload an image of a skin lesion
3. Select one or more models for prediction
4. View the classification results and confidence scores
5. Compare performance metrics across different models

## Model Performance Demo Output

| Model | Accuracy | Precision | Recall | F1 Score | Sensitivity | Specificity |
|-------|----------|-----------|--------|----------|------------|-------------|
| SVM | ~85% | ~82% | ~83% | ~82% | ~83% | ~90% |
| Random Forest | ~87% | ~84% | ~83% | ~83% | ~83% | ~88% |
| XGBoost | ~88% | ~86% | ~85% | ~85% | ~85% | ~91% |
| CNN | ~91% | ~90% | ~89% | ~89% | ~89% | ~91% |
| MobileNet | ~93% | ~92% | ~91% | ~91% | ~91% | ~92% |
| EfficientNet | ~94% | ~93% | ~92% | ~92% | ~92% | ~93% |

*Note: Actual performance may vary based on dataset quality and training parameters*

## Research Objectives

- Develop and compare different machine learning models for multi-class skin lesion classification
- Incorporate data augmentation and sensitivity analysis to improve robustness
- Evaluate the impact of various preprocessing techniques on model performance
- Assess model generalizability across different skin types and imaging conditions
- Create a practical web application for real-time image classification

## Future Improvements

- Skin Digest Scanner: Develop an advanced scanning system that can analyze skin conditions over time
- Real-time Detection System

## Dataset

This project uses publicly available skin lesion datasets at https://www.kaggle.com/datasets/ahmedxc4/skin-ds that have been preprocessed and augmented to improve model training. The dataset is organized by disease categories in the training directory.

## License

[MIT License](LICENSE)

## Author

***Ratna Akter***

Role: Lead Full-Stack Developer
GitHub: https://github.com/ratnaakter
LinkedIn: https://www.linkedin.com/in/ratnaakter/

Feel free to reach out for collaborations or questions about this project.

## Acknowledgments

- Dataset source: Skin-DS on Kaggle
- Special thanks to all open-source contributors whose libraries made this project possible
- The machine learning community for continuous research and improvement in this field.
