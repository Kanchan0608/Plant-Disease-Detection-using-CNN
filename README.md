🌿 Plant Disease Detection using CNN

A deep learning project to detect plant diseases from leaf images using a custom Convolutional Neural Network (CNN) built with PyTorch.

It uses the PlantVillage dataset with 20,000+ images across 15 classes.

📌 Key Features

✅ Custom CNN architecture with 5 convolutional blocks
✅ Early stopping & learning rate scheduler for optimal training
✅ Data augmentation for better generalization
✅ Achieved 98%+ validation accuracy
✅ Model exported in PyTorch and ONNX formats

🚀 How to Run

1️⃣ Clone the repo

git clone https://github.com/Kanchan0608/Plant-Disease-Detection-using-CNN
cd Plant-Disease-Detection-CNN-Model


2️⃣ Install dependencies

pip install r requirements.txt

3️⃣ Train the model

python train.py data_dir /path/to/PlantVillage epochs 40 batch_size 32

4️⃣ Predict on an image

python inference.py image_path /path/to/image.jpg

📊 Results

* Validation Accuracy: 98%+
* Test Accuracy: 97.6%
* Detailed accuracy breakdown by plant type available.

📈 Learning Curve

![Learning Curves]C:\Users\Dinesh Yadav\Downloads\download.png


📂 Files

* plant_disease_model.py: CNN model + dataset + data prep
* train.py: Training and evaluation script + learning curve save
* inference.py: Predict single image
* best_model.pth: Trained model weights
* class_names.json: Class names
* label_encoder.pkl: Label encoder
* inference_transform.pkl: Transform for inference
* learning_curves.png: Plot + save
* requirements.txt: Dependencies
* README.md: Project overview
