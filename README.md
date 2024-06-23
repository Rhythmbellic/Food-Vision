🍔👁 Food Vision

Food Vision tackles food image recognition, achieving higher accuracy. This project leverages transfer learning, utilizes the full dataset, and implements optimization techniques for faster training.

Overview
In this project, we aim to build a highly accurate food image recognition model using the entire Food101 dataset. The goal is to achieve higher top-1 accuracy than the DeepFood paper, which achieved 77.4% top-1 accuracy using a Convolutional Neural Network trained for 2-3 days.

Key Concepts
Transfer Learning
We utilize transfer learning to enhance our model's performance by leveraging pre-trained models on large datasets.

Full Dataset Utilization
We use the entire Food101 dataset, which consists of:

Training Images: 75,750
Testing Images: 25,250
Optimization Techniques
To improve the training speed and efficiency, we implement:

Prefetching: Preloading data to reduce waiting times during training.
Mixed Precision Training: Using a combination of single precision (float32) and half-precision (float16) to speed up model training, achieving up to 3x faster training on modern GPUs.
Project Structure
Data Handling:

Using TensorFlow Datasets to download and explore data.
Creating preprocessing functions for data augmentation and normalization.
Batching and preparing datasets for efficient loading and training.
Model Building:

Setting up mixed precision training.
Building a feature extraction model.
Fine-tuning the model for improved accuracy.
Training and Evaluation:

Implementing training callbacks.
Monitoring training progress using TensorBoard.
Evaluating the model's performance against the baseline DeepFood paper.
Getting Started
To get started with this project, clone the repository and follow the setup instructions:

bash
Copy code
git clone https://github.com/yourusername/food-vision.git
cd food-vision
Prerequisites
Ensure you have the following installed:

Python 3.x
TensorFlow 2.x
TensorFlow Datasets
NVIDIA GPU with compute capability score of 7.0+ (for mixed precision training)
Installation
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Checking GPU Compatibility
For mixed precision training, ensure your GPU is compatible:

python
Copy code
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
Compare your GPU name with the list of GPUs on NVIDIA's developer page to confirm compatibility.

Running the Project
Follow the Jupyter notebooks provided in the repository to run the project step-by-step. It is recommended to write the code yourself to get hands-on experience and understand each part of the process.
