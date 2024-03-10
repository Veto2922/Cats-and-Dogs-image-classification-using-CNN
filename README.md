# Image Classification with CNN and Transfer learning


## Overview

implementation of a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras. The dataset consists of cat and dog images, and the goal is to train a model to distinguish between the two.

Transfer learning with the VGG16 model is demonstrated as an advanced technique, showcasing how pre-trained models can be leveraged for image classification tasks, potentially improving model performance and reducing training time.

The notebook also provides insights into the interpretation of model decisions, allowing users to understand how the trained CNN makes predictions based on the learned features from the images. 

![image](https://github.com/Veto2922/Cats-and-Dogs-image-classification-using-CNN/assets/114834171/f4425f43-5b24-4ed3-a65f-d6a1431b454b)



### Dataset

The dataset used in this project can be downloaded from the following link: [Dataset Link](https://mega.nz/folder/6o1R1YLI#qSh8Te0cpt22a26bfjvmcg)

## Requirements

- TensorFlow
- Keras
- Matplotlib

## File Structure

- `CNN_CODE.ipynb`: Jupyter Notebook containing the complete code for building, training, and evaluating the CNN model.
- `my_cnn_model.h5`: Saved CNN model after training.
- `README.md`: Project documentation.

## How to Use

1. **Download the Dataset:**
   - Download the dataset from the provided link and extract it.

2. **Run the Jupyter Notebook:**
   - Open `CNN_CODE.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute each cell in the notebook sequentially to run the code.

3. **Evaluate the Model:**
   - The notebook will display the training and validation accuracy over epochs. Evaluate the model's performance.

4. **Save and Load the Model:**
   - The trained CNN model will be saved as `my_cnn_model.h5`. You can load the model for further use.

## Notes

- The notebook includes data augmentation techniques for improved model generalization.
- Transfer learning with the VGG16 model is demonstrated as an advanced technique.
- Instructions are provided within the notebook for each step of the process.

## Conclusion

This Jupyter Notebook serves as a comprehensive guide to building and training a CNN for image classification, offering insights into data preprocessing, model building, and interpretation of model decisions.
