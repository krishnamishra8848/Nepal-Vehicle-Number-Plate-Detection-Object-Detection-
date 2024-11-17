# Nepal Vehicle Number Plate Detection

This project utilizes the YOLO (You Only Look Once) model for detecting vehicle number plates in images and videos. The model is specifically trained to detect license plates from various vehicles, including bikes, cars, trucks, and buses, commonly found in Nepal. The model can process images as well as video feeds from sources such as CCTV cameras by making simple adjustments in the code.

The system processes uploaded images or video streams and identifies number plates, drawing bounding boxes around the detected plates with confidence scores. If no license plates are detected, the system will display a message indicating that no plates were found.

## Model Demo

The YOLO model used in this project is trained to detect vehicle number plates from a variety of vehicles, including bikes, cars, trucks, and buses commonly seen on the roads of Nepal. Below are some demo images showing the output of the model:

### Demo 1: Bike License Plate Detection
![demo1](https://github.com/user-attachments/assets/c872b790-da34-4329-9b9f-837a19bc07c2)

### Demo 2: Truck License Plate Detection
![demo2](https://github.com/user-attachments/assets/1680bce3-6d6a-4242-82eb-69dc553da85b)

### Demo 3: Car License Plate Detection
![demo3](https://github.com/user-attachments/assets/721539a6-3ae2-4336-bbae-a09296b693e5)

## Access the Model

You can access the deployed model [here](https://nepal-vehicle-number-plate-detection.onrender.com/).


## Key Features

- **Vehicle Types Supported**: This model can detect number plates from **bikes, cars, trucks, and buses** in images and video feeds, making it adaptable for various use cases such as traffic monitoring, parking management, and surveillance in Nepal.
- **Image and Video Processing**: The model can be used to process both static images and live video streams (e.g., from CCTV cameras). With minor code changes, it can seamlessly switch between processing image files and video feeds.

## How It Works

1. **Model Training**: The model is trained using YOLO, an efficient and powerful object detection algorithm. It has been specifically trained on a dataset that includes images of vehicles from Nepal to ensure accurate detection of number plates.
2. **Inference**: The model performs inference on uploaded images or live video streams, detecting license plates and providing the location (bounding boxes) along with a confidence score.
3. **Result**: If a number plate is detected, the system draws bounding boxes around it. If no plate is detected, a message is displayed stating "No license plates detected."

## How to Use

1. **Image Processing**: Upload an image containing a vehicle number plate, and the model will automatically detect and display the result with bounding boxes around the detected plates and confidence scores.
2. **Video/CCTV Processing**: By adjusting the code, you can process video files or live video feeds from sources like CCTV cameras. The model will continuously detect license plates in real-time and display the results.
3. **No Detection**: If no plates are detected, the system will show a message saying "No license plates detected."

