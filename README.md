
# Emotion Detection - Test Evaluation

This repository contains a script that evaluates an unseen **test dataset** (similar in format to the SMILE PLUS dataset) against a previously trained **Emotion Detection** model. The best model, scaler, and feature selector are stored in the `artifacts/` folder, and the script will load these components to preprocess the test images, predict their labels, and then output performance metrics.

---

## 1. Prerequisites

1. **Python Version**: Recommended Python 3.7 or higher.  
2. **Dependencies**:  
   - [NumPy](https://numpy.org/)  
   - [Pandas](https://pandas.pydata.org/)  
   - [OpenCV (cv2)](https://pypi.org/project/opencv-python/)  
   - [scikit-learn](https://scikit-learn.org/stable/)  
   - [scikit-image](https://scikit-image.org/)  
   - [tqdm](https://pypi.org/project/tqdm/)  
   - [joblib](https://joblib.readthedocs.io/)  
   - [Matplotlib](https://matplotlib.org/)  
   - [Seaborn](https://seaborn.pydata.org/)  

Install the dependencies after creating a virtual environment using below commands (you can check the folder structure to see where you can find `requirements.txt` and ):
```bash
python -m venv .venv

# For Windows
.venv\Scripts\activate

# For Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Folder Structure

Organize your project folder in below format based on the submitted materials:

```
project/
├── test_emotion_detection.py
├── artifacts/
│   ├── emotion_detection_scaler.pkl
│   ├── emotion_detection_selector.pkl
│   └── emotion_detection_model.pkl
├── test_data/
│   ├── images.jpg
│   └── annotations.csv
├── requirements.txt
└── README.md
```

- `test_emotion_detection.py`: The script that loads the model, preprocesses images, and evaluates performance.  
- `artifacts/`: Contains the three saved components from training:
  - `emotion_detection_scaler.pkl`
  - `emotion_detection_selector.pkl`
  - `emotion_detection_model.pkl`
- `test_data/`: Your **test dataset** folder, containing:
  - **Images** in `.jpg` format
  - An `annotations.csv` file specifying filenames and their corresponding emotion (either `happy` or `neutral`)

---

## 3. Usage

1. **Open a Terminal** in the root of your project.
2. **Run**:
   ```bash
   python test_emotion_detection.py
   ```
3. **Respond to prompts**:
   - **Path to your test image folder** (e.g., `test_data`)
   - **Path to your annotations file** (e.g., `test_data/annotations.csv`)

Example:
```
=== Emotion Detection Tester ===
Enter the path to the folder containing test images (similar to SMILE PLUS Training Set): test_data
Enter the path to the CSV file containing test annotations (e.g., annotations.csv): test_data/annotations.csv
```

Then the script will:
1. Load your images and annotations.
2. Preprocess each image (grayscale, histogram equalization, etc.).
3. Load the scaler and feature selector from `artifacts/`.
4. Load the saved best model from `artifacts/`.
5. Extract features from the images, apply scaling and feature selection.
6. Predict the emotions (happy or neutral).
7. Output:
   - **Classification report** (precision, recall, F1, etc.)
   - **Confusion matrix** (in both text and heatmap form)
   - **ROC curve** (with AUC)

---

## 4. Output

You’ll see in your console:
- A **classification report** (with metrics for each class).
- A **confusion matrix** printed out, plus a Matplotlib window displaying it as a heatmap.
- A **ROC curve** plot and the corresponding AUC metric.

---

## 5. Troubleshooting

- **File Not Found**: Double-check the paths to your images and CSV file when prompted.
- **Missing Artifacts**: Ensure you have all three `.pkl` files in the `artifacts/` folder.
- **Dependency Errors**: Install the modules listed in the [Prerequisites](#1-prerequisites).



