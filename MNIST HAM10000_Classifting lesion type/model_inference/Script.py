import os
import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Loading and Preprocessing Data
file_path = "../data/main_image_data.csv"
df = pd.read_csv(file_path)

# Extracting only informations of the images that are splited in testing set
df = df.loc[df['subset'] == 'test'].copy()

# Extracing the pixel columns as a NumPy array
pixel_columns = df[[f'pixel_{i+1}' for i in range(32*32*3)]].values

# Reshapeing the pixel columns to create the 'image' column
df['image'] = [pixels.reshape(32, 32, 3) for pixels in pixel_columns]

# Dropping the individual pixel columns
df.drop(columns=[f'pixel_{i+1}' for i in range(32*32*3)], inplace=True)

# Loading the Pre-trained Model and Making Predictions
model = load_model("../model_training/classifying_model_v3.h5")

df['image'] = df['image'] / 255

# Reshaping the features 
X = np.array(df['image'].tolist())

# Predicting on testing set 
predictions = model.predict(X)

# Convert testing set classes to one hot vectors
predicted_classes = np.argmax(predictions, axis=1)

# Mapping Predicted Classes to real Class Labels
class_labels = [
    'BKL',
    'NV',
    'DF',
    'MEL',
    'VASC',
    'BCC',
    'AKIEC'
]

# Creating predicted labels for each class
predicted_labels = [class_labels[i] for i in predicted_classes]

# Creating JSON Data and Save to Json File
data_list = []
for image_id, pred_class in zip(df['image_id'], predicted_labels):
    data_list.append({"image_id": image_id, "lesion_type": pred_class})

json_data = json.dumps(data_list, indent=4)

with open('predictions_v3.json', 'w') as json_file:
    json_file.write(json_data)

