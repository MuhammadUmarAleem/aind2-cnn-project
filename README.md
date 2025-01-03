# Convolutional Neural Networks Project

This repository contains code for the Convolutional Neural Networks project as part of the Artificial Intelligence Nanodegree. The project demonstrates how to use transfer learning to classify dog breeds using a pre-trained VGG-16 model.

## Repository Structure

- `transfer-learning/bottleneck_features.ipynb`: Notebook to calculate VGG-16 bottleneck features on a toy dataset.
- `transfer-learning/transfer_learning.ipynb`: Notebook to train a CNN to classify dog breeds using transfer learning.

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/MuhammadUmarAleem/aind2-cnn-project.git
    cd aind2-cnn-project
    ```

2. Download the necessary datasets:
    - Dog images dataset: [Download here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
    - VGG-16 bottleneck features: [Download here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz)

3. Extract the datasets:
    - Extract `dogImages.zip` into the `dogImages/` folder.
    - Place `DogVGG16Data.npz` into the `bottleneck_features/` folder.

## Usage

1. Open the Jupyter notebooks:
    ```bash
    jupyter notebook
    ```

2. Navigate to the `transfer-learning` folder and open the notebooks:
    - `bottleneck_features.ipynb`
    - `transfer_learning.ipynb`

3. Follow the instructions in the notebooks to run the code cells and train the model.

## Requirements

- Python 3.5+
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

Install the required packages using:
```bash
pip install -r requirements.txt
```
