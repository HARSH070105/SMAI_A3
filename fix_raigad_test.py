import os
from pathlib import Path

train = r"C:\Users\padma\OneDrive\Desktop\SEM 6\SMAI\A3\SMAI_A3\data\24_monuments\Indian-monuments\images\train\Raigad Fort"
test  = r"C:\Users\padma\OneDrive\Desktop\SEM 6\SMAI\A3\SMAI_A3\data\24_monuments\Indian-monuments\images\test\Raigad Fort"

imgs = list(Path(test).iterdir())
existing_train_count = len(list(Path(train).iterdir()))
for i, img in enumerate(imgs, start=existing_train_count + 1):
    os.rename(str(img), os.path.join(train, f"{i}{img.suffix}"))
print("Moved", len(imgs), "images back to train")
