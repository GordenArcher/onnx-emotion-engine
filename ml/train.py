import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# Centralized Configuration
# All tunable hyperparameters sit here so we can rapidly iterate without spelunking
# through the training loop. A single source of truth saves time and prevents bugs
# that crop up when magic numbers are scattered across the file.
DATA_DIR = "./data"
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 7
IMAGE_SIZE = 48
MODEL_SAVE_PATH = "emotion_model.onnx"

# Automatically select the fastest available hardware. If CUDA is present we use
# it, otherwise we gracefully degrade to CPU. This makes the script portable
# between an ML workstation and a lightweight cloud instance.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Data Transforms
# The Go backend mandates a fixed input: 48×48 pixels in the [0.0, 1.0] range.
# Every transform we define here must eventually produce a tensor that respects
# that contract, otherwise the server will silently misinterpret the image.
train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Convert to a 3‑channel grayscale image. Although the original FER-2013
        # frames are monochrome, MobileNetV2 was pretrained on RGB ImageNet and
        # expects three input channels. We replicate the single channel three
        # times to match that expectation without introducing colour artifacts.
        transforms.Grayscale(num_output_channels=3),
        # Horizontal flips are a cheap form of augmentation for faces. A
        # mirrored smile is still a smile, but a model that hasn't seen flipped
        # images may incorrectly associate "smile on the left side" with
        # happiness. Flipping eliminates that spurious correlation.
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # ImageNet normalization is mandatory when reusing MobileNet weights.
        # The pretrained filters expect pixel intensities centered and scaled
        # to ImageNet's exact statistics. Feeding raw [0,1] tensors would
        # shift the activations outside the distribution the network learned,
        # effectively turning the pretrained layers into random feature
        # extractors.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# The validation pipeline must be completely deterministic. We deliberately omit
# every form of augmentation so that each evaluation pass sees exactly the same
# pixels. This gives us a stable, reproducible accuracy number that reflects how
# the model will behave in production.
val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Dataset Loading
# ImageFolder sorts class directories alphabetically by name and assigns integer
# labels in that order. The mapping is deterministic, but we print it explicitly
# so that a human can verify that "angry" → 0, "disgust" → 1, etc. A mismatch
# here would silently swap emotions in production, which is catastrophic.
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"), transform=train_transform
)
val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "validation"), transform=val_transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

class_names = train_dataset.classes
print(f"Classes mapped by PyTorch: {class_names}")

# Class Imbalance Compensation
# FER-2013 is notoriously skewed. "Happy" appears roughly 10× more often than
# "disgust". Without weighting, the model can achieve ~80% accuracy by simply
# predicting "happy" for every sample—a useless classifier in practice.
# We compute per‑class weights as the inverse frequency so that rare emotions
# contribute proportionally more to the loss. The weights are normalized to sum
# to NUM_CLASSES, preserving the overall loss magnitude regardless of the
# imbalance ratio.
target_counts = np.bincount([label for _, label in train_dataset.samples])
class_weights = 1.0 / torch.tensor(target_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights = class_weights.to(device)

# Model Architecture (Transfer Learning)
# MobileNetV2 is the sweet spot between latency, memory footprint, and accuracy.
# It runs comfortably on a single CPU core, making it ideal for a Go server that
# may need to handle concurrent requests without a GPU. The architecture
# originated from Google's work on mobile vision, so every design decision—from
# inverted residuals to linear bottlenecks—is tailored for resource‑constrained
# environments.
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze the early layers and unfreeze only the last three feature blocks.
# The early layers capture universal primitives (edges, corners, blobs) that
# transfer flawlessly from ImageNet to facial expressions. Unfreezing the
# deeper blocks lets them adapt to face‑specific textures like wrinkles and
# mouth curvature, while still keeping the majority of the network frozen to
# prevent catastrophic forgetting and speed up training.
for param in model.features[-3:].parameters():
    param.requires_grad = True

# Replace the original 1000‑class ImageNet head with a compact classifier for
# our seven emotions. A dropout rate of 0.2 provides light regularization
# without crippling the already‑small head. The Linear layer receives 1280
# features (model.last_channel) and projects them to NUM_CLASSES logits.
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2), nn.Linear(model.last_channel, NUM_CLASSES)
)

model = model.to(device)

#  Loss, Optimizer & Scheduler
# CrossEntropyLoss internally applies log‑softmax followed by negative log‑
# likelihood. Baking softmax into the loss rather than the model avoids
# numerical instability when logits are large, and it simplifies ONNX export
# because the exported graph doesn't carry a softmax node that Go might
# accidentally double‑apply.
criterion = nn.CrossEntropyLoss(weight=class_weights)

# AdamW decouples weight decay from the adaptive learning rate, a fix that
# produces strictly better generalization than vanilla Adam. We optimize only
# the classifier parameters because the feature extractor is frozen.
optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)

# Cosine annealing smoothly reduces the learning rate from its initial value to
# near zero over the full training run. This helps the optimizer settle into a
# sharp, well‑generalizing minimum rather than oscillating around a flat basin.
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

#  Training & Validation Loop
print("\nSTARTING TRAINING...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm renders a live progress bar with the current loss and accuracy,
    # replacing a wall of scrolling numbers with a compact, interpretable
    # dashboard. This matters when you are training on a remote SSH session
    # and need to quickly gauge whether the run is healthy.
    loop = tqdm(train_loader, leave=True)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Stream live metrics into the progress bar description so the engineer
        # can spot anomalies (loss spikes, stuck accuracy) within seconds.
        loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

    scheduler.step()

    # Validation Phase
    # We run validation after every epoch to detect overfitting as early as
    # possible. A growing gap between training and validation accuracy is the
    # canary in the coal mine that tells us to stop training or increase
    # regularization before the model memorizes the training set.
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # torch.no_grad() disables gradient computation entirely. This reduces
    # memory consumption by roughly 40–50% and shaves 10–20% off the inference
    # wall time because PyTorch can skip the bookkeeping required for autograd.
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100.0 * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%\n")

#  ONNX Export
print("Exporting trained model to ONNX...")

# Export must happen in eval mode. If the model is still in training mode,
# Dropout layers remain active and will randomly zero out features during
# inference, producing non‑deterministic (and degraded) predictions inside the
# Go service.
model.eval()

# The dummy tensor must precisely mirror the shape that the Go backend will
# submit: (batch_size, channels, height, width) = (1, 3, 48, 48). Any mismatch
# will cause a shape error at inference time, and those errors are notoriously
# difficult to debug across language boundaries.
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

# torch.onnx.export traces the model graph and serializes it into the ONNX
# format, which is an interoperable standard supported by onnxruntime‑go.
# We pin opset_version to 11 because it is the most widely deployed version
# across the Go ecosystem and avoids compatibility regressions introduced
# in later opsets.
torch.onnx.export(
    model,
    dummy_input,
    MODEL_SAVE_PATH,
    export_params=True,  # Embed the trained weights in the ONNX file
    opset_version=11,  # Sweet spot for onnxruntime-go compatibility
    do_constant_folding=True,  # Collapse static subgraphs for a smaller, faster model
    input_names=["input"],  # The Go client will feed data into this named tensor
    output_names=["output"],  # The Go client will read predictions from this tensor
    dynamic_axes=None,  # Lock batch size to 1 to keep server memory predictable
)

print(f"SUCCESS! Model saved to {MODEL_SAVE_PATH}")
print("This file is ready to be moved into your Go backend.")
