import os
import shutil
import pandas as pd

def standardize_screw_dataset(base_dir, output_dir):
    """
    Chuẩn hóa toàn bộ dataset Screw:
    - Gộp tất cả ảnh từ các folder screw1, screw2,... vào 3 folder train/test/valid
    - Đặt lại tên file theo định dạng (good/defect)+(stt).png
    - Tất cả loại lỗi (cacat, scratch_head, thread_side, v.v.) → defect
    - 'bagus' hoặc 'ok' → good
    - Không bỏ qua ảnh nào
    """
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid'), exist_ok=True)

    counters = {"train": {"good": 0, "defect": 0},
                "test": {"good": 0, "defect": 0},
                "valid": {"good": 0, "defect": 0}}

    # Duyệt qua tất cả các folder screw*
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path) or not folder_name.startswith("screw"):
            continue

        print(f"Processing: {folder_name}")

        # Đọc file sample_submit.csv nếu có
        csv_path = os.path.join(folder_path, "sample_submit.csv")
        sample_labels = {}
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["label"] = df["Predicted"].apply(lambda x: "good" if x == 0 else "defect")
            sample_labels = {k.lower(): v for k, v in zip(df["id"], df["label"])}

        # Duyệt qua train/test/valid
        for split in ["train", "test", "valid"]:
            split_path = os.path.join(folder_path, split)
            if not os.path.exists(split_path):
                continue

            subfolders = [f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))]

            # Trường hợp có subfolder (vd: bagus, cacat, scratch_head,...)
            if subfolders:
                for defect_type in subfolders:
                    defect_path = os.path.join(split_path, defect_type)

                    defect_type = defect_type.lower()
                    # Phân loại lại
                    if any(x in defect_type for x in ["good", "bagus", "ok"]):
                        final_type = "good"
                    else:
                        final_type = "defect"

                    for file in os.listdir(defect_path):
                        if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                            continue

                        counters[split][final_type] += 1
                        new_name = f"{final_type}{counters[split][final_type]:05d}.png"

                        shutil.copy2(
                            os.path.join(defect_path, file),
                            os.path.join(output_dir, split, new_name)
                        )

            # Trường hợp folder chứa trực tiếp ảnh
            else:
                for file in os.listdir(split_path):
                    if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                        continue

                    file_key = file.lower()
                    if split == "test" and file_key in sample_labels:
                        final_type = sample_labels[file_key]
                    else:
                        # Suy ra từ tên file
                        name = ''.join([c for c in file.split('.')[0] if not c.isdigit()]).lower()
                        if any(x in name for x in ["good", "bagus", "ok"]):
                            final_type = "good"
                        else:
                            final_type = "defect"

                    counters[split][final_type] += 1
                    new_name = f"{final_type}{counters[split][final_type]:05d}.png"

                    shutil.copy2(
                        os.path.join(split_path, file),
                        os.path.join(output_dir, split, new_name)
                    )

    # In thống kê kết quả
    print("\nFinished!")
    for split in ["train", "test", "valid"]:
        total = sum(counters[split].values())
        good_count = counters[split]["good"]
        defect_count = counters[split]["defect"]
        print(f"\n{split.upper()}: {total} images")
        print(f"   - good: {good_count} images ({good_count/total*100:.2f}%)")
        print(f"   - defect: {defect_count} images ({defect_count/total*100:.2f}%)")


# Ví dụ chạy
base_dir = "D:\FPTU\Semester 5\DPL302m\datasets\Screw Datasets"
output_dir = "D:\FPTU\Semester 5\DPL302m\datasets\Screw Datasets\screw_all"
standardize_screw_dataset(base_dir, output_dir)
