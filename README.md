#Image Classification with CNN and Transfer learning

### Overview

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras. The dataset consists of cat and dog images, and the goal is to train a model to distinguish between the two.

### Dataset

The dataset used in this project can be downloaded from the following link: [Dataset Link](https://mega.nz/folder/6o1R1YLI#qSh8Te0cpt22a26bfjvmcg)

### Requirements

- TensorFlow
- Keras
- Matplotlib

### File Structure

- `main_cnn.py`: Python script containing the code for building, training, and evaluating the CNN model.
- `my_cnn_model.h5`: Saved CNN model after training.
- `transfer_learning_cnn.py`: Python script demonstrating transfer learning using the VGG16 model.
- `README.md`: Project documentation.

### How to Use

1. **Download the Dataset:**
   - Download the dataset from the provided link and extract it.

2. **Run the CNN Model:**
   - Execute the `main_cnn.py` script to train the CNN model on the cat and dog images.

3. **Evaluate the Model:**
   - The script will display the training and validation accuracy over epochs. Evaluate the model's performance.

4. **Save and Load the Model:**
   - The trained CNN model will be saved as `my_cnn_model.h5`. You can load the model for further use.

5. **Transfer Learning:**
   - Optionally, you can explore transfer learning using the VGG16 model by running `transfer_learning_cnn.py`.

### Code Highlights

```python
# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Data Preprocessing
# ...

# Building the CNN
# ...

# Training the CNN
# ...

# Plotting Training and Validation Accuracy
# ...

# Making a Single Prediction
# ...

# Save the Model
# ...

# Transfer Learning with VGG16
# ...

# Interpret Model Decisions
# ...
```

### Notes

- The project includes data augmentation techniques for improved model generalization.
- Transfer learning with the VGG16 model is demonstrated as an advanced technique.
- The README provides clear instructions on how to use and run the project.

### Conclusion

This project serves as a practical example of building and training a CNN for image classification, offering insights into data preprocessing, model building, and interpretation of model decisions.
