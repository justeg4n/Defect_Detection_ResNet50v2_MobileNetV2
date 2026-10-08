import os
import shutil
import pandas as pd
import re

def clean_token(s: str) -> str:
    """Làm sạch chuỗi: chỉ giữ chữ và số, thay ký tự khác bằng '_'."""
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9]', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s

def merge_sheet_metal_datasets(base_dir, output_dir):

    canonical = [
        'crazing', 'rolled-in_scale', 'inclusion',
        'patches', 'scratches', 'pitted_surface', 'good'
    ]


    alias_map = {
        'rolled': 'rolled-in_scale',
        'rolled_in': 'rolled-in_scale',
        'rolledin': 'rolled-in_scale',
        'rolled-in': 'rolled-in_scale',
        'rolledpit': 'rolled-in_scale',
        'rolled_pit': 'rolled-in_scale',
        'pitted': 'pitted_surface',
        'pittedsurface': 'pitted_surface',
        'scratch': 'scratches',
        'patch': 'patches',
        'good': 'good',
        'ok': 'good',
        'normal': 'good',
        'regular': 'good'
    }


    for split in ['train', 'valid', 'test']:
        for cls in canonical:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)


    counters = {split: {cls: 0 for cls in canonical} for split in ['train', 'valid', 'test']}

    def normalize_class_from_raw(raw):
        """Trả về tên class hợp lệ hoặc None nếu không hợp lệ."""
        if raw is None:
            return None
        tok = clean_token(raw)

        # Ưu tiên kiểm tra alias
        for a, targ in alias_map.items():
            if a in tok:
                return targ

        # Kiểm tra chứa từ khóa class hợp lệ
        for cls in canonical:
            if cls.replace('-', '_') in tok:
                return cls

        return None  # Nếu không nhận diện được class thì bỏ qua

    def load_annotations(folder):
        lookup = {}
        for f in os.listdir(folder):
            if f.lower().endswith(('_annotation.csv', '_annotations.csv', 'annotations.csv')):
                try:
                    df = pd.read_csv(os.path.join(folder, f))
                    cols = [c.lower().strip() for c in df.columns]
                    df.columns = cols
                    if 'filename' not in df.columns:
                        df = df.rename(columns={df.columns[0]: 'filename'})
                    if 'class' in df.columns:
                        for _, row in df.iterrows():
                            fn = os.path.basename(str(row['filename']).strip()).lower()
                            lookup[fn] = str(row['class']).lower()
                except Exception as e:
                    print(f"⚠️ Lỗi đọc file annotation {f}: {e}")
        return lookup


    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if not folder_name.lower().startswith(('sheet_metal', 'sheetmetal')):
            continue

        print(f"\n Đang xử lý: {folder_name}")

        for sub in os.listdir(folder_path):
            split_path = os.path.join(folder_path, sub)
            if not os.path.isdir(split_path):
                continue

            # Xác định tập train/valid/test
            s_low = sub.lower()
            if s_low.startswith('val'):
                out_split = 'valid'
            elif s_low.startswith('test'):
                out_split = 'test'
            else:
                out_split = 'train'

            ann_lookup = load_annotations(split_path)

            # Kiểm tra nếu có subfolders theo class
            nested = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            if nested:
                for n in nested:
                    guess_cls = normalize_class_from_raw(n)
                    if guess_cls not in canonical:
                        continue  # bỏ qua class không hợp lệ
                    nested_path = os.path.join(split_path, n)
                    for img in os.listdir(nested_path):
                        if not img.lower().endswith(('.jpg', '.jpeg', '.png')) or '_mask' in img.lower():
                            continue
                        img_basename = os.path.basename(img).lower()
                        cls = normalize_class_from_raw(ann_lookup.get(img_basename, n))
                        if cls not in canonical:
                            continue
                        counters[out_split][cls] += 1
                        new_name = f"{cls}{counters[out_split][cls]:05d}.png"
                        shutil.copy2(os.path.join(nested_path, img),
                                     os.path.join(output_dir, out_split, cls, new_name))
            else:
                # Xử lý ảnh trực tiếp trong split_path
                for img in os.listdir(split_path):
                    if not img.lower().endswith(('.jpg', '.jpeg', '.png')) or '_mask' in img.lower():
                        continue
                    img_basename = os.path.basename(img).lower()
                    cls = normalize_class_from_raw(ann_lookup.get(img_basename, img_basename))
                    if cls not in canonical:
                        continue
                    counters[out_split][cls] += 1
                    new_name = f"{cls}{counters[out_split][cls]:05d}.png"
                    shutil.copy2(os.path.join(split_path, img),
                                 os.path.join(output_dir, out_split, cls, new_name))


    rows = []
    for split in ['train', 'valid', 'test']:
        for cls, count in counters[split].items():
            rows.append({'split': split, 'class': cls, 'count': count})
    summary_path = os.path.join(output_dir, 'summary_counts.csv')
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\n Chuẩn hóa hoàn tất! File thống kê: {summary_path}")



base_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Sheet Metal Datasets\sheet_metal_parts"
output_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Sheet Metal Datasets\sheet_metal_all"
merge_sheet_metal_datasets(base_dir, output_dir)
