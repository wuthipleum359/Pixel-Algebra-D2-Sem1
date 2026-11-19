# Pixel Algebra — Automated Pixel Art Generator & Regression Model

An end-to-end data processing and machine learning application that converts images into pixel art using custom NumPy-based K-Means color quantization and predicts optimal scaling parameters via a Random Forest Regression model.

## Project Workflow & Key Features
- **Automated Dataset Collection (`import_pic.py`):** Fetches categorized image datasets (`people`, `food`, `plants`, `environment`, `buildings`) via the Pixabay REST API, followed by manual data cleaning and re-labeling to ensure dataset quality.
- **K-Means Quantization & Feature Extraction (`pixel_art.py`):** Iteratively scales 150 cleaned images across multiple resolutions, applies custom NumPy K-Means clustering for color reduction, and computes Mean Squared Error (MSE) to build a dataset of **6,800+ records** (`pixel_summary.csv`).
- **Elbow-Point MSE Recommendation (`mse_recomment.py`):** Uses geometric vector projection to detect the optimal Elbow Point on the Scale–MSE curve for each image category.
- **Regression Modeling & Visualization (`new_model_create.py`):** Trains a `RandomForestRegressor` to predict the optimal downscaling factor given a target MSE, image resolution, and category, while generating correlation heatmaps and regression plots.
- **Interactive Desktop GUI (`model_with_ui.py`):** Built with Python `Tkinter` and `OpenCV`, allowing users to upload images or capture photos via webcam, select categories, adjust target MSE sliders, and export generated pixel art.

## Tech Stack
- **Language:** Python 3
- **Data & Machine Learning:** NumPy, Pandas, Scikit-learn (`RandomForestRegressor`), Joblib
- **Image Processing & GUI:** Pillow (PIL), OpenCV (`cv2`), Tkinter
- **Visualization & API:** Matplotlib, Seaborn, Requests (Pixabay API)

## Getting Started
1. Clone this repository and install the required dependencies:
   ```bash
   pip install numpy pandas scikit-learn pillow opencv-python matplotlib seaborn joblib requests
   ```
2. Run the training script to generate the regression model (`scale_predictor_model.pkl`) from `pixel_summary.csv`:
   ```bash
   python new_model_create.py
   ```
3. Launch the desktop application:
   ```bash
   python model_with_ui.py
   ```

---
*Note: Developed as a collaborative team project for Elementary Differential Equations and Linear Algebra at King Mongkut's Institute of Technology Ladkrabang (KMITL).*
