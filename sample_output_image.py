import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# Load CSV
df = pd.read_csv(r"C:\Users\ramad\OneDrive\Desktop\Banana_APP\Training_dataset\banana_labels.csv")
print("Columns in CSV:", df.columns)

print(df.columns)
# Take first row
sample = df.iloc[0]

# Adjust if your column name is different
image_path = sample["image_path"]   # change this if needed

if not os.path.exists(image_path):
    print("Image not found:", image_path)
else:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(f"Class: {sample['Category']} | Days Left: {sample['days_left']}")
    plt.axis("off")
    plt.show()
