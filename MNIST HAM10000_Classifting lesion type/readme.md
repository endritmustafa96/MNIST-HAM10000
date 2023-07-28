# Skin Cancer Lesion Classification Project

## Skin Cancer Lesion

This project aims to build a classifying model for skin cancer lesions using the MNIST HAM10000 dataset. The dataset has been preprocessed and analyzed to create a classifying model, which is saved as classifying_model_v3.h5. Additionally, an executable script has been developed to make predictions on the test set, and the results are stored in predictions_v3.json.

## Dataset

The dataset used in this project is the MNIST HAM10000 dataset, which contains images of skin cancer lesions. The dataset is organized as follows:

The metadata for the dataset is provided in the metadata.csv file, which was analyzed in the exploratory_data_analysis directory.

The images are located in the dataset/images directory, which consists of three subdirectories: train, test, and validation. Each of these subdirectories contains images in the JPEG format. However here i have provided a link.txt in data directory - to download pre procesed image data (main_image_data.csv file)

## Image Preprocessing

The image data is preprocessed and relevant information are extracted to create a main_image_data.csv file. This file contains all the necessary image information required for training the classifying model. The image preprocessing steps are documented in the ImagePreprocessing.ipynb notebook found in the model_training directory. To perform this notebook, you need dataset.zip file that was shared

## Classifying Model

The classifying model is created using the main_image_data.csv file (sharet by link.txt in data directory) and is trained in the Creating_Classifying_model_notebook notebook, which is located in the model_training directory. The trained model is then saved as classifying_model_v3.h5.

## Model Inference

To perform model inference on the test set, an executable script has been developed. The script reads the preprocessed image data from main_image_data.csv, loads the trained model from classifying_model_v3.h5, and then predicts the classes of the images in the test set. The predictions are saved in the predictions_v3.json file, which is located in the model_inference directory.
