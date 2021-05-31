import zipfile
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os


class MachineLearningModel:
    def __init__(self, model_filename):
        abspath = os.path.abspath(model_filename)
        dirname = os.path.dirname(abspath)
        with zipfile.ZipFile(model_filename, "r") as zip_ref:
            zip_ref.extractall("machine_learning_model")
        self.model_path = f"{dirname}/machine_learning_model"
        self.model = tensorflow.keras.models.load_model(f"{self.model_path}/keras_model.h5")
        self.label_dict = {}
        with open(f"{self.model_path}/labels.txt") as f:
            for line in f.readlines():
                line_split = line.split()
                self.label_dict[int(line_split[0])] = line_split[1]

        print("Model is ready!")

    def predict(self, image_filename, print_=True):
        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Load image
        image = Image.open(image_filename)
        image = image.convert('RGB')

        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = self.model.predict(data)
        
        # convert the output probabilities to an acutal label
        predicted_label = self.label_dict[prediction.argmax()]

        if print_:
            print(f"The prediction is: {predicted_label}!")

        return predicted_label
