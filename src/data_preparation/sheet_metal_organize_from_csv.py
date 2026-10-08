import os
import shutil
import pandas as pd

def merge_sheet_metal_datasets(base_dir, output_dir):
    """
    Merge sheet_metal datasets into a single dataset with folders:
      output_dir/train, output_dir/test, output_dir/valid
    All images are relabeled into two classes: 'good' or 'defect'.
    Rules:
      - Use annotation CSV (if present) to determine class (class != normal => defect).
      - If filename or subfolder indicates 'good' or 'regular' => good.
      - If filename/subfolder contains defect keywords => defect.
      - Skip mask files (files with '_mask' in name).
      - Keep all images (none are dropped), assign either good or defect.
      - New filenames: sheetmetal_<split>_<label>_<index>.png
    """
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid'), exist_ok=True)

    counters = {'train': {'good': 0, 'defect': 0},
                'test': {'good': 0, 'defect': 0},
                'valid': {'good': 0, 'defect': 0}}

    # keywords to detect good / defect from filename or folder name
    good_keywords = ['good', 'regular', 'non_defected', 'nondefected', 'ok', 'bagus']
    defect_keywords = [
        'bad', 'mixed', 'overexposed', 'shift', 'crazing', 'rolled', 'rolled-in', 'rolled_in',
        'scale', 'inclusion', 'patch', 'patches', 'scratch', 'scratches', 'pitted', 'defect', 'cacat'
    ]

    # helper to decide label from a text string
    def decide_label_from_text(text):
        t = text.lower()
        for kw in good_keywords:
            if kw in t:
                return 'good'
        for kw in defect_keywords:
            if kw in t:
                return 'defect'
        return None

    # iterate over sheet_metal* folders
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path) or not folder_name.startswith('sheet_metal'):
            continue

        print(f"Processing folder: {folder_name}")

        # load any annotation csv in this folder (choose first *_annotations.csv if exists)
        annotation_df = None
        for fname in os.listdir(folder_path):
            if fname.endswith('_annotations.csv') or fname.endswith('_annotation.csv') or fname.endswith('annotations.csv'):
                try:
                    annotation_df = pd.read_csv(os.path.join(folder_path, fname))
                    # normalize filename column name if not exactly 'filename'
                    if 'filename' not in annotation_df.columns and 'file' in annotation_df.columns:
                        annotation_df = annotation_df.rename(columns={'file': 'filename'})
                except Exception as e:
                    print(f"  Warning: failed to read CSV {fname}: {e}")
                break

        # build a lookup from annotation filenames -> classes (if csv exists)
        ann_lookup = {}
        if isinstance(annotation_df, pd.DataFrame) and 'filename' in annotation_df.columns and 'class' in annotation_df.columns:
            for _, row in annotation_df.iterrows():
                fn = str(row['filename']).strip()
                cls = str(row['class']).strip().lower()
                ann_lookup[fn] = cls

        # possible split folder names inside this source folder
        possible_splits = ['train', 'test', 'valid', 'validation', 'test_private', 'test_private_mixed', 'test_public']
        for sub in possible_splits:
            split_path = os.path.join(folder_path, sub)
            if not os.path.exists(split_path):
                continue

            # map various split folder names to output split
            if sub.startswith('val'):
                out_split = 'valid'
            elif sub.startswith('test'):
                out_split = 'test'
            else:
                out_split = 'train'

            # When sub has nested subfolders (e.g., test_public/good, test_public/bad)
            nested_subs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            if nested_subs:
                # iterate nested folders first
                for nested in nested_subs:
                    nested_path = os.path.join(split_path, nested)
                    # if nested folder is ground_truth containing masks, skip masks but keep corresponding images from other folders
                    for img_name in os.listdir(nested_path):
                        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            continue
                        # skip mask files explicitly
                        if '_mask' in img_name.lower():
                            continue
                        # decide label: priority: nested folder name -> annotation csv -> filename keywords -> default defect
                        label = decide_label_from_text(nested)
                        if label is None and img_name in ann_lookup:
                            # if annotated class indicates a defect type -> defect
                            ann_cls = ann_lookup[img_name]
                            if ann_cls and ann_cls.strip().lower() not in good_keywords:
                                label = 'defect'
                            else:
                                label = 'good'
                        if label is None:
                            label = decide_label_from_text(img_name)
                        if label is None:
                            # fallback: consider 'regular' and 'good' as good; otherwise defect
                            if 'regular' in img_name.lower():
                                label = 'good'
                            else:
                                label = 'defect'

                        counters[out_split][label] += 1
                        new_name = f"sheetmetal_{out_split}_{label}_{counters[out_split][label]:05d}.png"
                        src = os.path.join(nested_path, img_name)
                        dst = os.path.join(output_dir, out_split, new_name)
                        shutil.copy2(src, dst)

            else:
                # no nested subs: files are directly in split_path
                for img_name in os.listdir(split_path):
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    if '_mask' in img_name.lower():
                        continue

                    # priority for label: annotation CSV (exact filename match) -> filename keywords -> folder name -> fallback defect
                    label = None
                    # annotation lookup keys may be stored with/without path; try both
                    if img_name in ann_lookup:
                        ann_cls = ann_lookup[img_name]
                        label = 'good' if any(k in ann_cls for k in good_keywords) else 'defect'
                    else:
                        # sometimes CSV lists filenames with paths; try checking basename matches ignoring directories
                        found = False
                        for k in ann_lookup.keys():
                            if os.path.basename(k) == img_name:
                                ann_cls = ann_lookup[k]
                                label = 'good' if any(x in ann_cls for x in good_keywords) else 'defect'
                                found = True
                                break
                        if not found:
                            # infer from filename
                            label = decide_label_from_text(img_name)
                            if label is None:
                                # use split folder name clues
                                label = decide_label_from_text(sub)
                            if label is None:
                                # final fallback: regular => good else defect
                                if 'regular' in img_name.lower():
                                    label = 'good'
                                else:
                                    label = 'defect'

                    counters[out_split][label] += 1
                    new_name = f"sheetmetal_{out_split}_{label}_{counters[out_split][label]:05d}.png"
                    src = os.path.join(split_path, img_name)
                    dst = os.path.join(output_dir, out_split, new_name)
                    shutil.copy2(src, dst)

        # Additionally, handle images placed directly in folder_path (rare)
        for f in os.listdir(folder_path):
            p = os.path.join(folder_path, f)
            if os.path.isfile(p) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
                # decide default to train
                out_split = 'train'
                if '_mask' in f.lower():
                    continue
                label = None
                if f in ann_lookup:
                    ann_cls = ann_lookup[f]
                    label = 'good' if any(k in ann_cls for k in good_keywords) else 'defect'
                if label is None:
                    label = decide_label_from_text(f) or 'defect'
                counters[out_split][label] += 1
                new_name = f"sheetmetal_{out_split}_{label}_{counters[out_split][label]:05d}.png"
                shutil.copy2(p, os.path.join(output_dir, out_split, new_name))

    # After processing, write a summary CSV and print summary
    summary_rows = []
    print("\nSummary:")
    for split in ['train', 'test', 'valid']:
        total = counters[split]['good'] + counters[split]['defect']
        print(f"  {split}: total={total}, good={counters[split]['good']}, defect={counters[split]['defect']}")
        summary_rows.append({'split': split, 'good': counters[split]['good'], 'defect': counters[split]['defect'], 'total': total})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'summary_counts.csv'), index=False)
    print(f"Saved counts CSV to {os.path.join(output_dir, 'summary_counts.csv')}")

# Example usage: set your actual paths
base_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Sheet Metal Datasets"
output_dir = r"D:\FPTU\Semester 5\DPL302m\datasets\Sheet Metal Datasets\Sheet_Metal_All"
merge_sheet_metal_datasets(base_dir, output_dir)
