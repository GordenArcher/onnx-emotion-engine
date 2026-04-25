# ML Training Pipeline

This directory contains the offline training pipeline for the emotion detection model. The final artifact (`emotion_model.onnx`) is exported from here and consumed directly by the Go backend. Python is strictly not used at runtime.

## Directory Structure

```text
/ml
  ├── venv/                  # Isolated Python environment (ignored by git)
  ├── data/                  # Raw dataset downloaded via script (ignored by git)
  │   ├── train/             # 28,000+ training images
  │   └── validation/        # ~7,000 validation images
  ├── download_data.py       # Utility to fetch data from Kaggle
  ├── train.py               # Model definition, training loop, and ONNX export
  ├── emotion_model.onnx     # The compiled model artifact (committed to git)
  └── README.md              # This documentation file
```

## Repository Structure Decisions

*   **No Jupyter Notebooks:** I wrote this pipeline in raw `.py` scripts instead of Jupyter notebooks. Notebooks are great for exploration, but they are terrible for version control and CI/CD. Scripts ensure the training pipeline can be run headlessly in a Docker container or CI pipeline without manual cell execution.
*   **Separation of Data Fetching:** I separated `download_data.py` from `train.py`. This allows us to run the data ingestion independently if we ever need to update the dataset without re-running the training loop, and vice versa.
*   **Ignoring Data but Keeping the Artifact:** The raw images in `/data` are explicitly ignored in `.gitignore` because pushing binary files bloats Git history. However, the final `emotion_model.onnx` file is committed. This is crucial because our Go backend relies on this specific artifact to function immediately upon cloning the repo, without requiring the user to wait hours to train the model themselves.
*   **No Saved PyTorch Models (`.pt`):** We do not commit the PyTorch checkpoint. We only commit the ONNX export. The Go backend cannot read PyTorch files, so keeping them in the repo is just unnecessary clutter.

## Dataset

We use the FER-2013 (Facial Expression Recognition) dataset. 
*   **Resolution:** Images are natively 48x48 pixels.
*   **Classes (7):** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
*   **Preprocessing:** The dataset is heavily imbalanced (e.g., ~7000 Happy images vs ~400 Disgust). We address this via loss weighting, not synthetic data generation, to keep the pipeline simple and fast.

## Architecture & Decisions

We use Transfer Learning with `MobileNetV2` instead of building a CNN from scratch.
*   **Speed:** MobileNetV2 is optimized for edge devices. It allows our Go backend to maintain sub-10ms inference times.
*   **Input Shape:** The model expects a strict tensor shape of `[1, 3, 48, 48]` (Batch size 1, 3 color channels, 48x48 resolution).
*   **Grayscale to RGB:** Although FER-2013 is grayscale, we duplicate the single channel into 3 channels (`transforms.Grayscale(num_output_channels=3)`). We do this because MobileNetV2 was pre-trained on ImageNet (RGB). Forcing it to accept 1 channel would break the pre-trained weights.
*   **Fine-Tuning:** We froze the base convolutional layers and only un-froze the last 3 blocks (`model.features[-3:]`) to slightly adapt the feature extractors to facial micro-expressions without destroying the base ImageNet weights.
*   **Classifier Head:** The original 1000-class ImageNet head was replaced with a Dropout layer (0.2) and a Linear layer mapping to our 7 classes.

## Loss Function

We use `CrossEntropyLoss` with dynamically calculated inverse class frequencies. This penalizes the model heavily when it misclassifies minority classes (like Disgust), preventing the model from simply defaulting to "Happy" to achieve a false high accuracy.

## ONNX Export Details

The final step of `train.py` exports the PyTorch model to ONNX format (Opset 11).
*   **Input Name:** `input` (Go backend looks for this exact key)
*   **Output Name:** `output` (Go backend reads from this exact key)
*   **Dynamic Axes:** Disabled. We strictly enforce batch size 1 to ensure predictable memory allocation in the Go server.

## Reproducibility

If you need to retrain the model from scratch:

1. Ensure you are in the `/ml` directory.
2. Activate the virtual environment: `source venv/bin/activate`
3. Install dependencies: `pip install torch torchvision tqdm "numpy<2"`
4. Download the dataset: `python download_data.py`
5. Run training: `python train.py`

Upon completion, `emotion_model.onnx` will appear in this directory. Copy this file to the `/backend` directory to update the running Go server.
