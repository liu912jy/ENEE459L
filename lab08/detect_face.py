import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]


# Face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """Remove overlapping boxes and keep the highest confidence detections."""
    scores = box_scores[:, 4]
    boxes = box_scores[:, :4]

    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]

    picked = []

    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        indexes = indexes[:-1]

        if len(indexes) == 0:
            break
        if top_k > 0 and len(picked) >= top_k:
            break

        box = boxes[current]
        others = boxes[indexes]

        x1 = np.maximum(box[0], others[:, 0])
        y1 = np.maximum(box[1], others[:, 1])
        x2 = np.minimum(box[2], others[:, 2])
        y2 = np.minimum(box[3], others[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])

        union = area1 + area2 - intersection
        iou = intersection / np.maximum(union, 1e-8)

        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked]


def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    """Detect faces using an ONNX model and save the image with bounding boxes."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    resized = image.resize((320, 240))
    data = np.array(resized).astype(np.float32)
    data = (data - 127.0) / 128.0

    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=0)

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=available_providers
    )

    input_name = session.get_inputs()[0].name
    confidences, boxes = session.run(None, {input_name: data})

    confidences = confidences[0]
    boxes = boxes[0]

    scores = confidences[:, 1]
    mask = scores >= conf_threshold

    selected_boxes = boxes[mask]
    selected_scores = scores[mask]

    box_scores = np.column_stack((selected_boxes, selected_scores))

    if len(box_scores) > 0:
        box_scores = hard_nms(box_scores, 0.3)

    draw = ImageDraw.Draw(image)

    for row in box_scores:
        x1, y1, x2, y2, score = row

        x1 = int(x1 * width)
        x2 = int(x2 * width)
        y1 = int(y1 * height)
        y2 = int(y2 * height)

        draw.rectangle((x1, y1, x2, y2), outline="green", width=5)
        draw.text((x1, y1), f"{score:.2f}", fill="green")

    image.save(output_path)
    print("Face detection saved:", output_path)


# Edge detection
def non_maximum_suppression(magnitude, angle):
    """Thin edges by keeping only local maximum gradient values."""
    height, width = magnitude.shape
    result = np.zeros((height, width), dtype=np.float32)

    angle = np.rad2deg(angle) % 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            a = angle[i, j]

            if a < 22.5 or a >= 157.5:
                p = magnitude[i, j - 1]
                q = magnitude[i, j + 1]

            elif a < 67.5:
                p = magnitude[i + 1, j - 1]
                q = magnitude[i - 1, j + 1]

            elif a < 112.5:
                p = magnitude[i - 1, j]
                q = magnitude[i + 1, j]

            else:
                p = magnitude[i - 1, j - 1]
                q = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= p and magnitude[i, j] >= q:
                result[i, j] = magnitude[i, j]

    return result



def hysteresis_thresholding(img, low_thresh, high_thresh):
    """Keep strong edges and weak edges directly connected to strong edges."""
    height, width = img.shape
    result = np.zeros((height, width), dtype=np.uint8)

    strong = img >= high_thresh
    weak = (img >= low_thresh) & (img < high_thresh)

    result[strong] = 255

    for i in range(height):
        for j in range(width):
            if weak[i, j]:
                neighbors = strong[
                    max(0, i - 1):min(height, i + 2),
                    max(0, j - 1):min(width, j + 2)
                ]

                if np.any(neighbors):
                    result[i, j] = 255

    return result



def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    """Apply Canny edge detection using Gaussian, Sobel, NMS and thresholds."""
    image = Image.open(image_path).convert("L")
    img = np.array(image).astype(np.float32)

    blurred = gaussian_filter(img, sigma=1.4)

    kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    ky = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    gx = convolve(blurred, kx)
    gy = convolve(blurred, ky)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255

    angle = np.arctan2(gy, gx)

    thin = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thin, low, high)

    output = Image.fromarray(edges.astype(np.uint8))
    output.save(output_path)

    print("Edge detection saved:", output_path)


if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")