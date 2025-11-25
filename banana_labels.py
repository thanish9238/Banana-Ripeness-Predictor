# create_labels.py
import os
import csv

folders = {
    "Class A": {"class_id": 0, "days_left": 6, "Category": "Green"},
    "Class B": {"class_id": 1, "days_left": 3, "Category": "Yellow"},
    "Class C": {"class_id": 2, "days_left": 1, "Category": "Brown Spots"},
    "Class D": {"class_id": 3, "days_left": 0, "Category": "Black/Mushy"},
}

out_csv = "banana_labels.csv"
extensions = (".jpg", ".jpeg", ".png")

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "class_id", "days_left", "Category"])
    for folder, info in folders.items():
        if not os.path.isdir(folder):
            print(f"Warning: {folder} not found")
            continue
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(extensions):
                path = os.path.join(folder, fn)
                writer.writerow([path, info["class_id"], info["days_left"], info["Category"]])
print("Wrote", out_csv)
