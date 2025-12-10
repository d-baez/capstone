# Scope of Work

This component of the project is to create a CV model that can detect the ripeness of a fruit with the input from a camera. 
The model will be trained on a dataset of images of fruits at various stages of ripeness. The main tasks involved in this component are:
1. Data Collection: Gather a diverse dataset of fruit images labeled with their ripeness levels.
2. Data Preprocessing: Clean and preprocess the images to ensure they are suitable for training the model.
3. Model Selection: Choose an appropriate computer vision model architecture for the ripeness detection task.
4. Model Training: Train the selected model on the preprocessed dataset.
5. Model Evaluation: Evaluate the model's performance using appropriate metrics and validate its accuracy.
6. Deployment: Integrate the trained model into a system that can take input from a camera and output the ripeness level of the fruit.
7. Testing: Conduct thorough testing to ensure the model works effectively in real-world scenarios.
8. Documentation: Document the entire process, including data collection methods, model architecture, training procedures, and deployment steps.
The successful completion of this component will result      

Project Goals

	•	Develop a lightweight image-classification model to detect:
        •	Unripe
        •	Ripe
        •	Overripe
		•	Spoiled
	•	Use a camera feed for real-time inference.
	•	Optimize the model for edge deployment:
	•	TensorFlow Lite (TFLite)
	•	ONNX Runtime
	•	Jetson-accelerated inference (when available)
	•	Compare lightweight CNN performance to large VLMs (like Phi-3.5 Vision).

⸻

System Architecture

1. Data Collection
	•	Images captured using a camera or a manually curated dataset.
	•	Preprocessing pipeline for:
	•	Resizing
	•	Normalization
	•	Augmentation (brightness, rotation, blur)

2. Model Design:

Two model pathways are explored:

    a. Lightweight CNN ( Raspberry Pi)
        •	Small custom CNN
        •	Optional MobileNetV2 or EfficientNet-Lite
        •	Outputs 4 classes: unripe / ripe / overripe / spoiled
        •	Export to TFLite for fast inference
    
    B. Phi-3.5 Vision (Desktop Testing Only)
        •	Used for:
        •	Annotation assistance
        •	Prototype testing
        •	Validation
        •	Not suitable for live inference on low-power devices
