import os
import cv2
import numpy as np
import tensorflow as tf

# ======================== MODEL LOADING ==========================
def load_keras_model(path, input_shape=(160, 160, 3)):
    """Load Keras model and ensure it has a functional graph."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Model not found: {path}")

    model = tf.keras.models.load_model(path, compile=False)
    print(f"[INFO] ✅ Model loaded from: {path}")

    # --- START FIX ---
    # Luôn build lại model thành Functional để đảm bảo .inputs và .output tồn tại
    # Đây là yêu cầu bắt buộc của Grad-CAM, ngay cả khi model.predict() vẫn chạy.
    print("[INFO] Rebuilding as Functional model to ensure .inputs/.output for Grad-CAM...")
    try:
        # --- START FIX 2 ---
        # Đặt tên_duy_nhất cho Input layer mới để tránh xung đột
        # với tên "input_layer" đã có sẵn trong model được load.
        inputs = tf.keras.Input(shape=input_shape, name="gradcam_input_adapter")
        # --- END FIX 2 ---
        outputs = model(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Chạy một lượt dummy input để build graph
        dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
        _ = model(dummy_input)
        print("[INFO] ✅ Functional model rebuilt and graph built successfully.")
        
    except Exception as e:
        print(f"[ERROR] Failed to rebuild Functional model: {e}")
        print("[INFO] Attempting to proceed with the original loaded model (may fail Grad-CAM)...")
        # Thử gọi dummy call trên model gốc nếu build lại thất bại
        try:
            dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
            _ = model(dummy_input)
            print("[INFO] ✅ Original model graph built successfully (fallback).")
        except Exception as e2:
            print(f"[ERROR] Dummy call on original model also failed: {e2}")
            raise e # Ném lỗi gốc
    # --- END FIX ---

    return model


# ======================== IMAGE PREPROCESSING ==========================
def preprocess_input_image(img_path, target_size):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"[ERROR] Image not found: {img_path}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise IOError(f"[ERROR] Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size[1], target_size[0]))
    x = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)
    return x, img_bgr


# ======================== LAYER HELPERS ==========================
def find_last_conv_layer(model):
    """Find last Conv2D layer recursively."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            try:
                # Đệ quy vào model con (ví dụ: MobileNetV2 base model)
                return find_last_conv_layer(layer)
            except ValueError:
                continue
    raise ValueError("[ERROR] No Conv2D layer found in model.")


def get_layer_by_name_recursive(model, layer_name):
    """Find a layer by name recursively."""
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
        if hasattr(layer, "layers"):
            try:
                found_layer = get_layer_by_name_recursive(layer, layer_name)
                if found_layer:
                    return found_layer
            except Exception:
                continue
    # Nếu không tìm thấy ở cấp hiện tại, trả về None để vòng lặp bên ngoài tiếp tục
    return None

# ======================== GRAD-CAM ==========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """Generate Grad-CAM heatmap."""
    
    # --- START FIX 4 ---
    # Model functional (đã được build lại) mong đợi input là một list,
    # ngay cả khi chỉ có 1 input.
    try:
        _ = model([img_array]) # Bọc img_array trong list
    except Exception:
        _ = model([np.zeros_like(img_array)]) # Bọc img_array trong list
    # --- END FIX 4 ---

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
        print(f"[INFO] Auto-detected last conv layer: {last_conv_layer_name}")

    last_conv_layer = get_layer_by_name_recursive(model, last_conv_layer_name)
    if last_conv_layer is None:
         raise ValueError(f"[ERROR] Could not find layer by name: {last_conv_layer_name}")

    # Check này bây giờ sẽ pass vì model đã được build lại ở load_keras_model
    if not hasattr(model, "output") or model.output is None:
        raise ValueError("[ERROR] Model has no defined output after calling — cannot compute Grad-CAM.")

    # Check tương tự cho inputs
    if not hasattr(model, "inputs") or model.inputs is None:
        raise ValueError("[ERROR] Model has no defined inputs after calling — cannot compute Grad-CAM.")

    # --- START FIX 3 ---
    # Sửa lỗi: model.inputs đã là một list.
    # Truyền [model.inputs] sẽ tạo ra một list lồng [[<tensor>]]
    # gây ra lỗi KeyError và cảnh báo input mismatch.
    grad_model = tf.keras.models.Model(
        model.inputs, [last_conv_layer.output, model.output]
    )
    # --- END FIX 3 ---

    with tf.GradientTape() as tape:
        # --- START FIX 4 (tiếp theo) ---
        # Áp dụng tương tự khi gọi grad_model
        conv_outputs, predictions = grad_model([img_array])
        # --- END FIX 4 (tiếp theo) ---
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError("[ERROR] Gradient is None — cannot compute Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Tránh chia cho 0 nếu heatmap tối đa là 0
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        heatmap = tf.maximum(heatmap, 0)
    else:
        heatmap = tf.maximum(heatmap, 0) / max_val
        
    return heatmap.numpy()


# ======================== VISUALIZATION ==========================
def overlay_heatmap_on_image(orig_bgr, heatmap, alpha=0.5):
    h, w = orig_bgr.shape[:2]
    heatmap_resized = cv2.resize((heatmap * 255).astype("uint8"), (w, h))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, orig_bgr, 1 - alpha, 0)
    return overlay, heatmap_resized


def heatmap_to_bboxes(heatmap_uint8, threshold=128, min_area=100):
    _, thr = cv2.threshold(heatmap_uint8, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    return boxes, thr


def draw_boxes_on_image(img_bgr, boxes, color=(0, 0, 255), thickness=2):
    img_out = img_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(img_out, (x, y), (x + w, y + h), color, thickness)
    return img_out


# ======================== PIPELINE ==========================
def gradcam_boxes_pipeline(model_path, img_path, output_path,
                           target_size=(160, 160), threshold=150,
                           min_area=120, alpha=0.5):
    print("\n=== Grad-CAM Defect Detection ===")
    model = load_keras_model(model_path, input_shape=(target_size[0], target_size[1], 3))
    x, orig_bgr = preprocess_input_image(img_path, target_size)

    preds = model.predict(x)
    pred_class = int(np.argmax(preds[0]))
    pred_conf = float(np.max(preds[0]))
    print(f"[INFO] Prediction: class={pred_class}, confidence={pred_conf:.4f}")

    heatmap = make_gradcam_heatmap(x, model, pred_index=pred_class)
    overlay_img, heatmap_uint8 = overlay_heatmap_on_image(orig_bgr, heatmap, alpha)
    boxes, mask = heatmap_to_bboxes(heatmap_uint8, threshold, min_area)
    print(f"[INFO] Detected {len(boxes)} defect region(s).")

    result_img = draw_boxes_on_image(overlay_img, boxes)
    
    # Đảm bảo thư mục output tồn tại
    output_dir = os.path.dirname(output_path)
    if output_dir: # Chỉ tạo nếu output_path không phải là file ở thư mục gốc
        os.makedirs(output_dir, exist_ok=True)
        
    base = os.path.splitext(output_path)[0]

    cv2.imwrite(output_path, result_img)
    cv2.imwrite(base + "_heatmap.png", cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET))
    cv2.imwrite(base + "_mask.png", mask)

    print(f"[INFO] Results saved:\n - {output_path}\n - {base + '_heatmap.png'}\n - {base + '_mask.png'}")

    return {
        "pred_class": pred_class,
        "pred_confidence": pred_conf,
        "num_boxes": len(boxes),
        "boxes": boxes,
        "output_image": output_path,
    }


# ======================== MAIN ==========================
if __name__ == "__main__":
    # Sử dụng tên file bạn đã tải lên.
    # Đảm bảo các file này nằm trong cùng thư mục với script
    # hoặc thay đổi đường dẫn tuyệt đối tại đây.
    model_path = "D:\FPTU\Semester 5\DPL302m\datasets\SheetMetal\output\sheet_metal_mobilenetv2_optimized.h5"
    img_path = "D:\FPTU\Semester 5\DPL302m\datasets\scratches00001.png"
    out_path = "D:\FPTU\Semester 5\DPL302m\datasets\SheetMetal\outputgradcam_result.png" # Lưu vào thư mục con

    try:
        result = gradcam_boxes_pipeline(
            model_path=model_path,
            img_path=img_path,
            output_path=out_path,
            target_size=(160, 160),
            threshold=150,
            min_area=120,
            alpha=0.5
        )
        print("\n[INFO] Pipeline completed successfully.")
        print(result)
    except Exception as e:
        print(f"\n[ERROR] An error occurred during the pipeline: {e}\n")
        import traceback
        traceback.print_exc()

