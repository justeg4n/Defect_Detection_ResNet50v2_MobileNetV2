import os
import shutil
import re
from collections import defaultdict

#1. Các class hợp lệ
VALID_CLASSES = [
    "good", "manipulated_front", "scratch_head",
    "scratch_neck", "thread_side", "thread_top"
]

# 2. Các từ khóa nhận dạng class trong tên file
CLASS_KEYWORDS = {
    "good": ["good", "ok", "bagus", "normal"],
    "manipulated_front": ["manipulated_front", "defect", "ng", "manipulated", "broken"],
    "scratch_head": ["scratch_head", "head_scratch", "scratchhead", "headdefect"],
    "scratch_neck": ["scratch_neck", "neck_scratch", "scratchneck"],
    "thread_side": ["thread_side", "side_thread", "threadside"],
    "thread_top": ["thread_top", "top_thread", "threadtop"],
}

def detect_class_from_name(name: str):
    """Xác định class dựa trên tên file hoặc thư mục (phiên bản mạnh hơn, không bỏ sót)"""
    name = name.lower()

    # Chuẩn hóa tên file (loại bỏ ký tự gây nhiễu)
    name = re.sub(r"[^a-z0-9]", "_", name)  # thay ký tự lạ bằng "_"
    name = re.sub(r"_+", "_", name)  # gộp nhiều dấu "_" liên tiếp
    name = name.replace(".rf_", "_").replace("_png", "").replace("_jpg", "")

    # Ưu tiên defect class trước để tránh "good" match nhầm
    for cls in ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]:
        for kw in CLASS_KEYWORDS[cls]:
            if kw in name:
                return cls

    # Cuối cùng mới kiểm tra class good
    for kw in CLASS_KEYWORDS["good"]:
        if kw in name:
            return "good"

    return None


def detect_split(name: str):
    """Nhận dạng split (train/test/valid) từ tên file hoặc thư mục"""
    name = name.lower()
    if "train" in name:
        return "train"
    elif "test" in name:
        return "test"
    elif "val" in name or "valid" in name:
        return "valid"
    else:
        return "train"  # fallback


def standardize_screw_dataset(base_dir, output_dir):
    # Tạo folder đích
    for split in ["train", "test", "valid"]:
        for cls in VALID_CLASSES:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    # Bộ đếm
    counters = defaultdict(lambda: defaultdict(int))
    log_path = os.path.join(output_dir, "mapping_log.txt")
    skipped_path = os.path.join(output_dir, "skipped_log.txt")

    with open(log_path, "w", encoding="utf-8") as log, open(skipped_path, "w", encoding="utf-8") as skipped:
        log.write("File\t→\tSplit\t→\tClass\n")
        skipped.write("File bị bỏ qua hoặc nghi ngờ phân loại sai:\n")

        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue

                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, base_dir)

                # Xác định split + class
                split = detect_split(rel_path)
                cls = detect_class_from_name(rel_path)

                # Nếu không nhận ra class
                if cls is None:
                    skipped.write(f" Không nhận ra class: {rel_path}\n")
                    continue

                # Nếu nhận ra class good nhưng có keyword của defect → nghi ngờ
                defect_keywords = sum(list(CLASS_KEYWORDS.values()), [])
                if cls == "good" and any(kw in rel_path.lower() for kw in defect_keywords if kw not in CLASS_KEYWORDS["good"]):
                    skipped.write(f"⚠️ Có thể phân loại sai (good?): {rel_path}\n")

                # Đặt tên mới
                counters[split][cls] += 1
                new_name = f"{cls}{counters[split][cls]:05d}.png"
                dest_path = os.path.join(output_dir, split, cls, new_name)

                shutil.copy2(filepath, dest_path)
                log.write(f"{rel_path}\t→\t{split}\t→\t{cls}\n")

    # In thống kê
    print("\n Dataset prepared successfully:")
    total = 0
    for split in ["train", "test", "valid"]:
        print(f"\n{split.upper()}:")
        for cls in VALID_CLASSES:
            count = counters[split][cls]
            print(f"  - {cls}: {count}")
            total += count
    print(f"\nTotal images processed: {total}")
    print(f"\n Log saved at: {log_path}")
    print(f"  Skipped/Warning files logged at: {skipped_path}")


# Ví dụ chạy
base_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Screw Datasets\screw_parts"
output_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Screw Datasets\screw_all_clean"

standardize_screw_dataset(base_dir, output_dir)
