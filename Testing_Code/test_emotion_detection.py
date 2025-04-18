import os
import cv2
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# --------------------
# 1. Ask user for test data paths
# --------------------
def main():
    print("=== Emotion Detection Tester ===")
    test_base_path = input("Enter the path to the folder containing test images (similar to SMILE PLUS Training Set): ")
    test_annotations_path = input("Enter the path to the CSV file containing test annotations (e.g., annotations.csv): ")

    if not os.path.exists(test_base_path):
        raise FileNotFoundError(f"Test images path not found: {test_base_path}")
    if not os.path.isfile(test_annotations_path):
        raise FileNotFoundError(f"Test annotations file not found: {test_annotations_path}")

    # --------------------
    # 2. Load test annotations
    # --------------------
    test_annotations = pd.read_csv(test_annotations_path, header=None, names=['filename', 'emotion'])
    print(f"Total test images (according to CSV): {len(test_annotations)}")
    print("Class distribution in test annotations:")
    print(test_annotations['emotion'].value_counts())

    # --------------------
    # 3. Preprocess / Load images
    # --------------------
    X_test, y_test, _ = load_and_preprocess_images(test_annotations, test_base_path)
    print(f"Successfully loaded {len(X_test)} images from test set.")

    # --------------------
    # 4. Extract features
    # --------------------
    test_features = extract_features(X_test)

    # --------------------
    # 5. Load scaler, selector, and model
    # --------------------
    print("Loading scaler, selector, and best model from artifacts folder...")
    scaler = joblib.load('artifacts/emotion_detection_scaler.pkl')
    selector = joblib.load('artifacts/emotion_detection_selector.pkl')
    best_model = joblib.load('artifacts/emotion_detection_model.pkl')

    # --------------------
    # 6. Combine the test features and transform them
    # --------------------
    # Combine all feature sets
    all_test_features_list = []
    for feature_name, f_data in test_features.items():
        all_test_features_list.append(f_data)
    combined_test_features = np.hstack(all_test_features_list)

    # Scale
    scaled_test = scaler.transform(combined_test_features)

    # Feature selection
    final_test_features = selector.transform(scaled_test)
    print(f"Final test features shape: {final_test_features.shape}")

    # --------------------
    # 7. Predict and Evaluate
    # --------------------
    print("\n=== Evaluating on Test Set ===")
    y_probs = best_model.predict_proba(final_test_features)[:, 1]
    y_pred = best_model.predict(final_test_features)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=['neutral', 'happy'], output_dict=True)
    precision = report['happy']['precision']
    recall = report['happy']['recall']
    f1_score = report['happy']['f1-score']

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # False Positive Rate (FPR)
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_probs)

    # Print required metrics
    print(f"\n=== Test Set Results ===")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-Measure (F1 Score): {f1_score:.4f}")
    print(f"ROC Area (AUC): {roc_auc:.4f}")
    print("Confusion Matrix ([0,1] = [neutral,happy]):\n", cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Neutral', 'Happy'],
                yticklabels=['Neutral', 'Happy'])
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Plot ROC-AUC
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_curve, tpr_curve, label=f"Test AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.title("ROC Curve - Test Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.show()

    print("=== Done. ===")


def load_and_preprocess_images(annotations, base_path, target_size=(128, 128)):
    """
    Load and preprocess images with basic grayscale, histogram equalization,
    and potential blur/CLAHE, just like in training.
    """
    X = []
    y = []
    filenames = []

    # Find all .jpg files recursively
    all_image_files = []
    for ext in ['*.jpg']:
        found = glob.glob(os.path.join(base_path, '**', ext), recursive=True)
        all_image_files.extend(found)
    # Lowercase dictionary for quick matching
    file_dict = {}
    for path in all_image_files:
        filename = os.path.basename(path).lower()
        file_dict[filename] = path

    # Match annotations with file paths
    for idx, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Loading test images"):
        filename = row['filename'].lower()

        variations = [
            filename,
            filename.replace('.jpg', '.jpeg'),
            filename.replace('.jpeg', '.jpg'),
            filename.replace('.JPG', '.jpg'),
            filename.replace('.JPG', '.jpeg')
        ]

        found_path = None
        for var in variations:
            if var in file_dict:
                found_path = file_dict[var]
                break

        if not found_path:
            # Fallback: try matching by base name
            base_name = os.path.splitext(filename)[0]
            matching_files = [p for f, p in file_dict.items() if f.startswith(base_name)]
            if matching_files:
                found_path = matching_files[0]

        if found_path:
            img = cv2.imread(found_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Hist. equalization
                gray = cv2.equalizeHist(gray)
                # Gaussian blur
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                # CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                # Resize
                resized = cv2.resize(gray, target_size)

                X.append(resized)
                # 1 for happy, 0 for neutral
                y.append(1 if row['emotion'] == 'happy' else 0)
                filenames.append(row['filename'])
            else:
                print(f"Failed to load image: {found_path}")
        else:
            print(f"No matching file found for: {row['filename']}")

    return np.array(X), np.array(y), filenames


def extract_features(images):
    """
    Extract HOG, LBP, histogram, edge, and region-based features,
    mirroring the exact approach used in training.
    """
    features = {}
    # ---------- HOG features ----------
    hog_features = []
    for image in images:
        feat = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )
        hog_features.append(feat)
    features['hog'] = np.array(hog_features)

    # ---------- LBP features ----------
    lbp_features1 = []
    lbp_features2 = []
    for image in images:
        # LBP radius=1
        lbp1 = local_binary_pattern(image, P=8, R=1, method='uniform')
        hist1, _ = np.histogram(lbp1.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
        hist1 = hist1.astype("float")
        hist1 /= (hist1.sum() + 1e-7)
        lbp_features1.append(hist1)

        # LBP radius=2
        lbp2 = local_binary_pattern(image, P=16, R=2, method='uniform')
        hist2, _ = np.histogram(lbp2.ravel(), bins=np.arange(0, 16 + 3), range=(0, 16 + 2))
        hist2 = hist2.astype("float")
        hist2 /= (hist2.sum() + 1e-7)
        lbp_features2.append(hist2)

    features['lbp1'] = np.array(lbp_features1)
    features['lbp2'] = np.array(lbp_features2)

    # ---------- Histogram features ----------
    hist_features = []
    for image in images:
        hist, _ = np.histogram(image.ravel(), bins=64, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        hist_features.append(hist)
    features['hist'] = np.array(hist_features)

    # ---------- Edge features ----------
    edge_features = []
    for image in images:
        edges1 = cv2.Canny(image.astype(np.uint8), 50, 150)
        edges2 = cv2.Canny(image.astype(np.uint8), 10, 50)
        edge_density1 = np.sum(edges1) / (edges1.shape[0] * edges1.shape[1])
        edge_density2 = np.sum(edges2) / (edges2.shape[0] * edges2.shape[1])

        h, w = image.shape
        region_feats = []
        for i in range(4):
            for j in range(4):
                reg1 = edges1[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
                reg2 = edges2[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
                rden1 = np.sum(reg1) / (reg1.shape[0] * reg1.shape[1])
                rden2 = np.sum(reg2) / (reg2.shape[0] * reg2.shape[1])
                region_feats.extend([rden1, rden2])

        edge_features.append([edge_density1, edge_density2] + region_feats)
    features['edge'] = np.array(edge_features)

    # ---------- Region-based features (mouth/eye areas) ----------
    region_features = []
    for image in images:
        h, w = image.shape
        # approximate eyes and mouth region
        left_eye_region = image[h//4:h//2, w//4:w//2]
        right_eye_region = image[h//4:h//2, w//2:3*w//4]
        mouth_region = image[2*h//3:, w//4:3*w//4]

        local_stats = []
        for region in [left_eye_region, right_eye_region, mouth_region]:
            local_stats.extend([
                np.mean(region),
                np.std(region),
                np.min(region),
                np.max(region)
            ])
            # small histogram in each region
            hist_r, _ = np.histogram(region.ravel(), bins=8, range=(0, 256))
            hist_r = hist_r.astype("float")
            hist_r /= (hist_r.sum() + 1e-7)
            local_stats.extend(hist_r)

        region_features.append(local_stats)
    features['region'] = np.array(region_features)

    return features


if __name__ == "__main__":
    main()
