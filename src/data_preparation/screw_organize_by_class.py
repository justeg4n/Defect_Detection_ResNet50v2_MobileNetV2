import os
import shutil # Thư viện để thực hiện các thao tác với file như di chuyển

def organize_images_into_folders(base_path):
    """
    Tự động quét và di chuyển các file ảnh vào các thư mục con có nhãn
    dựa trên tiền tố 'good' hoặc 'defect' trong tên file.

    Cấu trúc ban đầu mong đợi:
    - base_path/
      - train/
        - goodxxxx.png
        - defectxxxx.png
      - valid/
      - test/

    Cấu trúc sau khi chạy script:
    - base_path/
      - train/
        - good/
          - goodxxxx.png
        - defect/
          - defectxxxx.png
    """
    print(f"--- Starting directory organization process at: {base_path} ---")

    # Các thư mục con chính cần xử lý
    sub_dirs = ['train', 'test', 'valid']
    
    # Tên các lớp (nhãn)
    class_labels = ['good', 'defect']
    
    total_files_moved = 0

    # Lặp qua từng thư mục con chính (train, test, valid)
    for dir_name in sub_dirs:
        current_path = os.path.join(base_path, dir_name)

        # Kiểm tra xem thư mục có tồn tại không
        if not os.path.isdir(current_path):
            print(f"Warning: Directory '{current_path}' does not exist. Skipping.")
            continue
        
        print(f"\nProcessing directory: '{current_path}'")

        # 1. Tạo các thư mục con cho từng lớp nếu chúng chưa tồn tại
        for label in class_labels:
            label_path = os.path.join(current_path, label)
            os.makedirs(label_path, exist_ok=True) # exist_ok=True sẽ không báo lỗi nếu thư mục đã tồn tại

        # 2. Quét và di chuyển các file
        files_in_dir = os.listdir(current_path)
        for filename in files_in_dir:
            source_path = os.path.join(current_path, filename)
            
            # Quan trọng: Chỉ xử lý file, bỏ qua các thư mục con (như 'good', 'defect' mà ta vừa tạo)
            if not os.path.isfile(source_path):
                continue
            
            destination_path = None
            
            # Kiểm tra tên file để xác định thư mục đích
            if filename.lower().startswith('good'):
                destination_path = os.path.join(current_path, 'good', filename)
            elif filename.lower().startswith('defect'):
                destination_path = os.path.join(current_path, 'defect', filename)

            # Nếu tìm thấy đích đến hợp lệ, di chuyển file
            if destination_path:
                try:
                    shutil.move(source_path, destination_path)
                    # Lấy tên của thư mục cha để in ra cho rõ ràng (ví dụ: 'good' hoặc 'defect')
                    destination_folder_name = os.path.basename(os.path.dirname(destination_path))
                    print(f"  Moved: {filename} -> {destination_folder_name}/")
                    total_files_moved += 1
                except Exception as e:
                    print(f"Error moving file {filename}: {e}")

    # In ra bản tóm tắt cuối cùng
    print("\n-----------------------------------------------------")
    if total_files_moved > 0:
        print(f"Finished! Moved a total of {total_files_moved} files.")
    else:
        print("Finished! All files were already organized.")
    print("-----------------------------------------------------")


# --- PHẦN THỰC THI ---
if __name__ == "__main__":
    # Thay đổi đường dẫn này thành thư mục 'Screw' của bạn
    base_data_dir = "D:\FPTU\Semester 5\DPL302m\datasets\Screw Datasets\screw_all"
    
    # Gọi hàm để bắt đầu sắp xếp
    organize_images_into_folders(base_data_dir)