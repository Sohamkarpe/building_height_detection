# Building Height Detection using AI

**Author:** Soham Ashok Karpe, M. Eng.  
**Institution:** Technische Hochschule Deggendorf (DIT), Campus Cham  
**Program:** M.Eng. Artificial Intelligence for Smart Sensors and Actuators  
**Period:** Mar 2023 – Jul 2023 | DIT Case study Project  

---

## 📌 Project Overview

A machine learning system for **automated building height estimation** from aerial/street-level images using **Python** and **MATLAB**. The system uses feature extraction pipelines combined with regression models to predict building heights, with real-time data visualization for **architectural and urban planning** scenarios.

---

## 🎯 Key Features

- ✅ Automated building height estimation from images
- ✅ Feature extraction pipeline (texture, shape, edge, shadow analysis)
- ✅ Machine learning regression models (Random Forest, SVR, Neural Network)
- ✅ Real-time visualization and plotting
- ✅ MATLAB and Python implementation
- ✅ Urban planning data export (CSV, JSON)
- ✅ Batch processing for multiple buildings

---

## 🏗️ Project Structure

```
building_height_detection/
├── src/
│   ├── main.py                    # Main entry point
│   ├── feature_extractor.py       # Image feature extraction
│   ├── height_estimator.py        # ML height estimation models
│   ├── visualizer.py              # Results visualization
│   └── batch_processor.py         # Batch image processing
├── config/
│   └── config.yaml                # Configuration
├── utils/
│   ├── image_utils.py             # Image processing utilities
│   ├── data_export.py             # Export results to CSV/JSON
│   └── logger.py                  # Logging utility
├── matlab/
│   ├── feature_extraction.m       # MATLAB feature extraction
│   ├── height_estimation.m        # MATLAB height model
│   └── visualization.m            # MATLAB plotting
├── results/
│   └── .gitkeep
├── requirements.txt
└── README.md
```

---

## ⚙️ System Architecture

```
Input Image (aerial / street-level)
        ↓
Preprocessing
(resize, normalize, denoise)
        ↓
Feature Extraction
(edges, contours, shadows, texture, HOG)
        ↓
ML Model Inference
(Random Forest / SVR / Neural Network)
        ↓
Height Estimation (meters)
        ↓
Visualization + Export
(plot, CSV, JSON)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Feature Extraction** | OpenCV, scikit-image |
| **Machine Learning** | scikit-learn, PyTorch |
| **Visualization** | Matplotlib, Seaborn |
| **MATLAB** | Image Processing Toolbox |
| **Programming** | Python 3.10+, MATLAB R2023 |
| **Data Export** | Pandas, CSV, JSON |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Sohamkarpe/building_height_detection.git
cd building_height_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Estimate height from single image
python src/main.py --image path/to/building.jpg

# Batch process a folder of images
python src/main.py --folder path/to/images/

# Use specific ML model
python src/main.py --image building.jpg --model random_forest

# Export results to CSV
python src/main.py --folder images/ --export csv

# Visualize results
python src/main.py --image building.jpg --visualize
```

**MATLAB:**
```matlab
% Run feature extraction in MATLAB
run matlab/feature_extraction.m

% Run height estimation
run matlab/height_estimation.m

% Plot results
run matlab/visualization.m
```

---

## 📊 Model Performance

| Model | MAE (m) | RMSE (m) | R² Score |
|-------|---------|----------|---------|
| Random Forest | 3.2 | 4.8 | 0.87 |
| SVR (RBF kernel) | 4.1 | 5.9 | 0.83 |
| Neural Network | 2.9 | 4.2 | 0.89 |

---

## 🔬 Feature Extraction Methods

| Feature | Description |
|---------|-------------|
| **Edge Detection** | Canny, Sobel for building boundary detection |
| **Shadow Analysis** | Shadow length for height estimation |
| **HOG Features** | Histogram of Oriented Gradients |
| **Texture Analysis** | LBP (Local Binary Patterns) |
| **Contour Analysis** | Building shape and aspect ratio |
| **Color Histogram** | Facade color distribution |

---

## 📁 Dataset

> ⚠️ **Note:** The dataset used in this project is **not publicly available** due to institutional data permissions. The imagery was sourced from licensed aerial and street-level datasets used exclusively for research at Technische Hochschule Deggendorf, Campus Cham.

**Dataset Characteristics:**
- Aerial and street-level building images
- Ground truth heights from municipal data
- Urban and suburban building types
- Various lighting and weather conditions

To use with your own dataset:
1. Place images in `data/images/`
2. Place ground truth CSV in `data/labels.csv`
3. Update `config/config.yaml`

---

## 📝 Academic Context

Developed as part of the **HiWi research assistant** program at **Technische Hochschule Deggendorf, Campus Cham** in the field of Computer Vision and Machine Learning for urban analysis.

**Field:** Computer Vision, Machine Learning, Urban Planning AI  


---

## 📄 License

Academic and research use only. Contact author for commercial use.

---

## 👤 Author

**Soham Ashok Karpe, M. Eng.**  
AI Systems Engineer | Machine Vision & Deep Learning  
📧 karpesoham@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/soham-karpe)  
🐙 [GitHub](https://github.com/Sohamkarpe)
