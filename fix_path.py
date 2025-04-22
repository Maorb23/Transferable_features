import os
import shutil

def reorganize_val_set():
    # Adjust these paths based on your directory structure
    val_dir = r"C:\Users\maorb\Classes\DL\Transferable features\tiny-imagenet-200\val"
    img_dir = os.path.join(val_dir, "images")
    ann_file = os.path.join(val_dir, "val_annotations.txt")
    target_dir = r"C:\Users\maorb\Classes\DL\Transferable features\tiny-imagenet-200\val_fixed"

    # Create output directory if not exists
    os.makedirs(target_dir, exist_ok=True)

    # Read val_annotations.txt
    annotations = {}
    with open(ann_file, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            filename = parts[0]
            class_name = parts[1]
            annotations[filename] = class_name

    # Copy each image to its class-specific folder in val_fixed
    for img_filename in os.listdir(img_dir):
        if img_filename not in annotations:
            continue
        class_name = annotations[img_filename]
        class_dir = os.path.join(target_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        src = os.path.join(img_dir, img_filename)
        dst = os.path.join(class_dir, img_filename)
        shutil.copy(src, dst)

    print("✅ Validation set reorganized successfully into val_fixed.")

if __name__ == "__main__":
    reorganize_val_set()
