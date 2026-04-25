import os
import shutil

import kagglehub

# Download latest version to Kaggle's hidden cache directory
path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")
print("Downloaded to cache at:", path)

# Define our clean project data directory
target_dir = "./data"

# Copy the files from cache into our /ml/data folder
if not os.path.exists(target_dir):
    shutil.copytree(path, target_dir)
    print(f"Success! Dataset copied to: {target_dir}")
else:
    print(f"Dataset already exists in {target_dir}, skipping copy.")
