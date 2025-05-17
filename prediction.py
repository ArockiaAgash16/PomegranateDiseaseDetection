import numpy as np
import pickle
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import warnings
import os
import sys
import json

# ✅ Suppress TensorFlow INFO logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ✅ Load all models and artifacts
model = tf.keras.models.load_model("models/DenseNet_Model.keras", compile=False)
feature_extractor = load_model("models/feature_extractor.keras", compile=False)
healthy_mean = np.load("models/healthy_mean.npy")
healthy_std = np.load("models/healthy_std.npy")

with open("models/scaler_dict.pkl", "rb") as f:
    scaler_dict = pickle.load(f)
with open("models/thresholds_dict.pkl", "rb") as f:
    thresholds_dict = pickle.load(f)
with open("models/gmm_dict.pkl", "rb") as f:
    gmm_dict = pickle.load(f)

disease_categories = ["Alternaria", "Anthracnose", "Bacterial_Blight", "Cercospora", "Healthy"]

def grad_cam_plus_plus(model, img_array, layer_name="conv5_block16_concat"):
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img_array / 255.0, axis=0))
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(conv_outputs[0], weights), axis=-1).numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    heatmap_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
    return heatmap_resized

def get_roi(img_array):
    heatmap = grad_cam_plus_plus(model, img_array)
    heatmap_8bit = (heatmap * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(heatmap_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask, heatmap

def compute_severity_score(img_array, roi_mask):
    roi_bool = (roi_mask > 0).astype(np.float32)
    masked_img = img_array * np.expand_dims(roi_bool, axis=-1)
    masked_img = masked_img / 255.0
    masked_img = np.expand_dims(masked_img, axis=0)
    roi_features = feature_extractor.predict(masked_img)[0]
    deviation = np.sqrt(np.sum(((roi_features - healthy_mean) ** 2) / (healthy_std ** 2)))
    return deviation

def predict_disease_and_severity(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img).astype(np.uint8)
    img_input = np.expand_dims(img_array / 255.0, axis=0)
    pred_probs = model.predict(img_input)[0]
    pred_idx = np.argmax(pred_probs)
    predicted_class = disease_categories[pred_idx]

    if predicted_class == "Healthy":
        return predicted_class, "Healthy", 0.0, None, img_array

    roi_mask, heatmap = get_roi(img_array)
    raw_score = compute_severity_score(img_array, roi_mask)
    scaler = scaler_dict[predicted_class]
    normalized_score = scaler.transform([[raw_score]])[0][0] * 100

    gmm = gmm_dict[predicted_class]
    cluster = gmm.predict([[normalized_score]])[0]
    means = gmm.means_.flatten()
    order = np.argsort(means)
    severity_mapping = {order[0]: "Low", order[1]: "Medium", order[2]: "High"}
    severity_label = severity_mapping[cluster]

    return predicted_class, severity_label, normalized_score, heatmap, img_array

def overlay_gradcam(original_img, heatmap):
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
    return overlay

# ------------------------
# Visualization Function (optional)
# ------------------------
def visualize_gradcam_outputs(img_path, base_filename):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img).astype(np.uint8)

    heatmap = grad_cam_plus_plus(model, img_array)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_array.astype(np.uint8))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Grad-CAM++ Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay_rgb)
    plt.title("Overlay on Image")
    plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join("static/uploads", f"gradcam_panel_{base_filename}.jpg")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return f"gradcam_panel_{base_filename}.jpg"

# ------------------------
# Main Inference Execution
# ------------------------
def run_prediction(image_path):
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = "static/uploads"
    os.makedirs(output_dir, exist_ok=True)

    predicted_class, severity_label, normalized_score, heatmap, original_img = predict_disease_and_severity(image_path)

    output = {
        "predicted_class": predicted_class,
        "severity": severity_label,
        "severity_pct": round(normalized_score, 2),
        "full_gradcam_image": None
    }

    if predicted_class != "Healthy":
        gradcam_overlay = overlay_gradcam(original_img, heatmap)
        full_vis_path = os.path.join(output_dir, f"full_gradcam_{base_filename}.jpg")
        cv2.imwrite(full_vis_path, gradcam_overlay)

        panel_filename = visualize_gradcam_outputs(image_path, base_filename)
        output["full_gradcam_image"] = panel_filename

    return output