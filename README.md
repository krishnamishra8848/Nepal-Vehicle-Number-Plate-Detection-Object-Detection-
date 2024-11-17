# Nepal Vehicle Number Plate Detection

This project utilizes the YOLO (You Only Look Once) model for detecting vehicle number plates in images. The model processes uploaded images and identifies number plates, drawing bounding boxes around the detected plates with confidence scores. If no license plates are detected, the system will display a message indicating that no plates were found.

## Model Demo

The YOLO model used in this project is trained to detect vehicle number plates in various scenarios. Below are some demo images showing the output of the model:

### Demo 1: License Plate Detection
![demo1](https://github.com/user-attachments/assets/c872b790-da34-4329-9b9f-837a19bc07c2)

### Demo 2: License Plate Detection
![demo2](https://github.com/user-attachments/assets/1680bce3-6d6a-4242-82eb-69dc553da85b)

### Demo 3: License Plate Detection
![demo3](https://github.com/user-attachments/assets/721539a6-3ae2-4336-bbae-a09296b693e5)

## How It Works

1. **Model Training**: The model is trained using YOLO, which is an efficient and powerful object detection algorithm. It is optimized to detect number plates in images quickly and accurately.
2. **Inference**: The model performs inference on uploaded images, detecting license plates and providing the location (bounding boxes) along with a confidence score.
3. **Result**: If a number plate is detected, the system draws bounding boxes around it. If no plate is detected, a message is displayed stating "No license plates detected."

## How to Use

1. Upload an image containing a vehicle number plate.
2. The model will automatically process the image and display the results, including bounding boxes around detected plates and confidence scores.
3. If no plates are detected, the message "No license plates detected" will appear.


