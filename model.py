import torch

if torch.cuda.is_available():
    print("GPU is available")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available")

# Install Kaggle API if not already installed
!pip install -q kaggle

# Make a Kaggle directory
!mkdir -p ~/.kaggle

# Upload the kaggle.json file (Use Colab's file upload feature to upload it first)
from google.colab import files
files.upload()

# Move kaggle.json to the Kaggle directory
!mv kaggle.json ~/.kaggle/

# Set permissions for the Kaggle API key
!chmod 600 ~/.kaggle/kaggle.json

# Download the specific dataset
!kaggle datasets download -d ishworsubedii/vehicle-number-plate-datasetnepal

# Unzip the downloaded dataset
!unzip vehicle-number-plate-datasetnepal.zip -d dataset

!pip install ultralytics

from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/drive/MyDrive/YOLOv8_Checkpoints

data_yaml = """
train: /content/dataset/vehicle_number_plate_detection/images
val: /content/dataset/vehicle_number_plate_detection/images
nc: 1  # Number of classes (1 for vehicle number plates)
names: ['vehicle_number_plate']  # Class names
"""

# Save the data.yaml file
with open("data.yaml", "w") as f:
    f.write(data_yaml)

from ultralytics import YOLO

# Load the YOLOv8 Nano model
model = YOLO('yolov8n.yaml')

# Train the model
model.train(
    data="data.yaml",  # Path to the dataset configuration
    epochs=8,         # Number of epochs
    batch=16,          # Batch size
    imgsz=640,         # Image size
    project="/content/drive/MyDrive/YOLOv8_Checkpoints",  # Save directory
    name="vehicle_number_plate_model",  # Model name
    save_period=1,     # Save checkpoints every epoch
    device=0           # Use GPU
)

metrics = model.val(data="data.yaml", imgsz=640)
print(metrics)

  Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50      2.17G      2.593       2.97      2.873         20        640: 100%|██████████| 505/505 [02:53<00:00,  2.92it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 253/253 [01:13<00:00,  3.44it/s]
                   all       8078       8720      0.633      0.548      0.572      0.338

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50      2.19G      1.404        1.2      1.642         24        640: 100%|██████████| 505/505 [02:46<00:00,  3.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 253/253 [01:11<00:00,  3.52it/s]
                   all       8078       8720      0.659       0.55      0.618      0.448

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50      2.24G      1.185     0.9264      1.417         32        640: 100%|██████████| 505/505 [02:45<00:00,  3.05it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 253/253 [01:10<00:00,  3.60it/s]
                   all       8078       8720      0.879      0.798      0.887      0.682

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/50      2.15G      1.044     0.8093      1.303         16        640: 100%|██████████| 505/505 [02:45<00:00,  3.05it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 253/253 [01:09<00:00,  3.62it/s]
                   all       8078       8720      0.907      0.859       0.93      0.736

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/50      2.17G     0.9774     0.7325       1.24         22        640: 100%|██████████| 505/505 [02:46<00:00,  3.04it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 253/253 [01:11<00:00,  3.56it/s]
                   all       8078       8720      0.932      0.876      0.946      0.758

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/50      2.15G     0.9285     0.6777      1.209         19        640: 100%|██████████| 505/505 [02:44<00:00,  3.06it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 253/253 [01:10<00:00,  3.59it/s]
                   all       8078       8720      0.924      0.898      0.953      0.774

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/50      2.24G     0.8711     0.6407      1.167         37        640: 100%|██████████| 505/505 [02:43<00:00,  3.08it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 253/253 [01:09<00:00,  3.62it/s]
                   all       8078       8720      0.915      0.868      0.946      0.783

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/50      2.16G     0.8424     0.6084      1.152         29        640: 100%|██████████| 505/505 [02:42<00:00,  3.10it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 253/253 [01:11<00:00,  3.55it/s]
                   all       8078       8720      0.934      0.905      0.963      0.819



