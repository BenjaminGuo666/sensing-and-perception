
#  Human-aware Collaborative Robots in Manufacturing Settings

The goal is to perform action recognition on the **HRI30** dataset for industrial human–robot interaction and to generate predictions for the hidden test set.

---

## 1. Project Structure

```text
hri30_project/
├─ README.md
├─ requirements.txt
├─ data/
│   ├─ train_set/                # all training videos
│   ├─ test_set/                 # test videos
│   └─ annotations/
│       ├─ train_set_labels.csv  # original labels provided by staff
│       ├─ classInd.txt          # class index mapping
│       ├─ train_split.csv       # 80% train split
│       ├─ val_split.csv         # 20% validation split
├─ src/
│   ├─ __init__.py
│   ├─ dataset.py        # dataset definitions (train / val / test, OpenCV reader)
│   ├─ models.py         # model builder (3D ResNet-18)
│   ├─ utils.py          # accuracy meter, AverageMeter, helper functions
│   ├─ split_labels.py   # create train_split.csv and val_split.csv
│   ├─ train.py          # training script
│   ├─ predict_test.py   # inference on test_set -> test_set_labels.csv
└─ results/
    ├─ r3d18_16f/        # logs & checkpoints for 16-frame baseline
    ├─ r3d18_32f/
    │   ├─ best_model.pt # best checkpoint used for final submission
    │   └─ train_log.csv # training / validation curves
    └─ test_set_labels.csv   # final predictions for submission
````

---

## 2. Environment & Dependencies

The project was developed and tested with:

* **OS**: Windows 10 (64-bit)
* **Python**: 3.9
* **GPU**: NVIDIA RTX 4090 (32 GB RAM)
* **Key libraries**:

  * `torch`, `torchvision`
  * `opencv-python`
  * `pandas`, `numpy`
  * `scikit-learn`
  * `tqdm`

You can install the main dependencies via:

```bash
pip install -r requirements.txt
```

A  `requirements.txt`:

```txt
torch = 2.6.0+cu124
torchvision = 0.21.0+cu124
opencv-python = 4.12.0
pandas = 2.2.3
numpy = 2.0.2
scikit-learn = 1.6.1
tqdm = 4.67.1
```

> Note: TensorFlow is **not** required for this project.
> PyTorch with CUDA support is recommended to leverage the GPU.

---

## 3. Data Preparation

1. Download the SAP final project dataset and place it under `data/`:

   ```text
   data/
   ├─ train_set/
   ├─ test_set/
   └─ annotations/
       ├─ train_set_labels.csv
       └─ classInd.txt
   ```

2. **Split train / validation**
   Run the following script once to create `train_split.csv` and `val_split.csv` (stratified 80/20 split):

   ```bash
   cd hri30_project
   python -m src.split_labels
   ```

   This script:

   * Reads `annotations/train_set_labels.csv`
   * Performs a stratified split by `class_idx`
   * Writes:

     * `annotations/train_split.csv`
     * `annotations/val_split.csv`


   This will iterate over all video IDs in `train_set_labels.csv`

---

## 4. Code Overview

### 4.1 `src/dataset.py`

* Implements three dataset classes:

  * `HRI30Dataset`: used for **train** and **validation**.

    * Reads videos with **OpenCV** (`cv2.VideoCapture`).
    * Converts them to RGB, stacks into `(T, H, W, C)`.
    * Uniformly samples a fixed number of frames (`num_frames`, e.g., 32).
    * Applies per-frame transforms: `Resize(128) → CenterCrop(112) → ToFloat → Normalize`.
    * Returns a clip tensor of shape `(C, T, H, W)` and a label in `[0, 29]`.

  * `HRI30TestDataset`: used for the **test set**, where labels are not available.

    * Same preprocessing as `HRI30Dataset`, but returns `(clip, video_id)`.

  * `read_video_opencv(path)`:
    Helper function that reads a `.avi` file with OpenCV and returns a `(T, H, W, C)` tensor.
    If the video cannot be opened or has zero frames, a `RuntimeError` is raised.

* `HRI30Dataset.__getitem__` contains a small retry mechanism to skip corrupted videos (up to 3 retries).

### 4.2 `src/models.py`

* Provides a single function:

  ```python
  def build_r3d18(num_classes=30, pretrained=True)
  ```

* This function:

  * Loads `torchvision.models.video.r3d_18`.
  * If `pretrained=True`, uses **Kinetics-400** pretrained weights.
  * Replaces the final fully-connected layer with a new `nn.Linear` for 30 classes.

### 4.3 `src/utils.py`

* Utility components:

  * `accuracy(output, target)`: computes top-1 accuracy.
  * `AverageMeter`: tracks running average of loss / accuracy during training and validation.

### 4.4 `src/split_labels.py`

* Reads `train_set_labels.csv` and produces:

  * `train_split.csv` (80% samples)
  * `val_split.csv` (20% samples)

* The split is **stratified** by `class_idx` using `sklearn.model_selection.train_test_split`.

### 4.5 `src/train.py`

* Main training script:

  * Builds `HRI30Dataset` for train / val using `train_split.csv` and `val_split.csv`.
  * Constructs `DataLoader`s with configurable:

    * `batch_size`
    * `num_frames`
    * `num_workers`
  * Builds the 3D ResNet-18 model (`build_r3d18`).
  * Uses:

    * Loss: `CrossEntropyLoss`
    * Optimizer: `AdamW` with `lr = 3e-4`, `weight_decay = 1e-4`
    * LR scheduler: `StepLR(step_size=20, gamma=0.1)`
  * Trains for a number of epochs (e.g., 40).
  * After each epoch:

    * Evaluates on the validation set.
    * Logs train/val loss and accuracy to `results/r3d18_32f/train_log.csv`.
    * Saves the best model (highest `val_acc`) to `results/r3d18_32f/best_model.pt`.

### 4.6 `src/predict_test.py`

* Loads the best trained model from `results/r3d18_32f/best_model.pt`.
* Builds `HRI30TestDataset` and iterates over all videos in `test_set/`.
* For each clip:

  * Runs a forward pass.
  * Takes the argmax over class logits.
  * Converts predicted class index (0–29) back to `class_idx` (1–30) and `class_name` using `train_set_labels.csv`.
* Writes final predictions into:

  ```text
  results/test_set_labels.csv
  ```

  with format:

  ```text
  video_id, class_name, class_idx
  ```

---

## 5. How to Run

### 5.1 Split train / validation (only once)

```bash
python -m src.split_labels
```

### 5.2 Train the model

The final model in this project is a 3D ResNet-18 with 32-frame input:

```bash
python -m src.train
```

Key hyperparameters are defined inside `src/train.py`:

* `num_frames = 32`
* `batch_size = 16`
* `num_epochs = 40`
* `pretrained = True`
* `num_workers = 8`

During training, logs and checkpoints are written to:

```text
results/r3d18_32f/
    ├─ best_model.pt
    └─ train_log.csv
```

### 5.3 Generate predictions for the test set

After training has finished and `best_model.pt` is available:

```bash
python -m src.predict_test
```

This will create:

```text
results/test_set_labels.csv
```

which can be submitted to the course leaderboard / grading system.

---

## 6. Notes

* All video reading is done via **OpenCV**, not `torchvision.io.read_video`, in order to avoid potential issues with PyAV on Windows.
* A small number of corrupted or missing videos in the training set are automatically skipped during training.
  They are also listed in `annotations/bad_videos.txt` for reference.
* The model and code are designed to run efficiently on a single GPU (RTX 4090), but they can also fall back to CPU (much slower).

---

## 7. Acknowledgements

* Dataset:
  HRI30 – An Action Recognition Dataset for Industrial Human-Robot Interaction.
* Model architecture:
  `r3d_18` from `torchvision.models.video`, pretrained on Kinetics-400.
* Course:
  SAP (Sensing and Perception) – KCL.
