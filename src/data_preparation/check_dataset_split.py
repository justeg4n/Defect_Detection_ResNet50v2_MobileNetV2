import os

base_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Screw Datasets\screw_all_standardized"

for split in ['train', 'valid', 'test']:
    path = os.path.join(base_dir, split)
    if not os.path.exists(path):
        print(f" {split} không tồn tại")
        continue

    classes = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    print(f" {split}: {len(classes)} class folders")
    for c in classes[:5]:
        count = len(os.listdir(os.path.join(path, c)))
        print(f"   └── {c}: {count} ảnh")
