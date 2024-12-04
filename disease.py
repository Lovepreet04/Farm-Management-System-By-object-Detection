import numpy as np
import tensorflow as tf
import cv2
import os

class PlantDiseasePredictor:
    def __init__(self):
        path = os.path.join('PlantDisease', 'valid')
        self.model = tf.keras.models.load_model(os.path.join('PlantDisease', 'trained_plant_disease_model'))
        self.validation_set = tf.keras.utils.image_dataset_from_directory(
            path,
            labels="inferred",
            label_mode="categorical",
            image_size=(128, 128),
            batch_size=32,
            shuffle=True
        )
        self.class_names = self.validation_set.class_names

    def predict(self, image_path):
        # img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        predictions = self.model.predict(input_arr)
        result_index = np.argmax(predictions)
        model_prediction = self.class_names[result_index]

        return model_prediction


# if __name__ == "__main__":
#     predictor = PlantDiseasePredictor()
#     image_path = 'history/leaf_snapshot.png'
#     prediction = predictor.predict(image_path)
#     print(f"Disease Name: {prediction}")