import cv2
import numpy as np

def segment_image(img):
    # Dummy segmentation: seuillage simple
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return mask

def apply_model_and_color(img, model):
    img_resized = cv2.resize(img, (150, 150))
    img_input = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=(0, -1))  # (1,150,150,1)
    activation = model.predict(img_input)[0]

    heatmap = cv2.resize(activation[:, :, 0], (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, heatmap_color, 0.4, 0)

    return overlay, heatmap
