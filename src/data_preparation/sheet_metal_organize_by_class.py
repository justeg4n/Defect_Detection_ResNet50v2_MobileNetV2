import os
import shutil

def organize_images_into_folders(base_path, class_labels=None):
    """
    Sắp xếp lại các ảnh trong thư mục dataset có cấu trúc:

    Trước khi chạy:
        base_path/
            train/
                sheetmetal_train_defect_00001.png
                sheetmetal_train_good_00002.png
            valid/
                sheetmetal_valid_good_00003.png
            test/
                sheetmetal_test_defect_00004.png

    Sau khi chạy:
        base_path/
            train/
                good/
                    sheetmetal_train_good_00002.png
                defect/
                    sheetmetal_train_defect_00001.png
            valid/
                good/
                    sheetmetal_valid_good_00003.png
                defect/
                    ...
            test/
                defect/
                    sheetmetal_test_defect_00004.png
    """
    print(f"--- Starting directory organization process at: {base_path} ---")

    if class_labels is None:
        class_labels = ['good', 'defect']

    sub_dirs = ['train', 'test', 'valid']
    total_files_moved = 0

    for dir_name in sub_dirs:
        current_path = os.path.join(base_path, dir_name)
        if not os.path.isdir(current_path):
            print(f"Warning: Directory '{current_path}' does not exist. Skipping.")
            continue
        
        print(f"\nProcessing: '{current_path}'")

        # Tạo thư mục con cho từng class nếu chưa tồn tại
        for label in class_labels:
            os.makedirs(os.path.join(current_path, label), exist_ok=True)

        # Duyệt tất cả file trong thư mục (bỏ qua thư mục con)
        files = [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]
        for filename in files:
            src = os.path.join(current_path, filename)
            dst = None

            lower_name = filename.lower()

            # Kiểm tra nếu nhãn nằm trong tên file
            for label in class_labels:
                if label in lower_name:
                    dst = os.path.join(current_path, label, filename)
                    break

            if dst:
                try:
                    shutil.move(src, dst)
                    print(f"  Moved: {filename} -> {label}/")
                    total_files_moved += 1
                except Exception as e:
                    print(f"  Error moving {filename}: {e}")
            else:
                print(f"  Skipped: {filename} (no matching label found)")

    print("\n-----------------------------------------------------")
    if total_files_moved > 0:
        print(f"Finished! Moved a total of {total_files_moved} files.")
    else:
        print("Finished! All files were already organized or no label found.")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    base_data_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Sheet Metal Datasets\sheet_metal_all"
    organize_images_into_folders(base_data_dir)
