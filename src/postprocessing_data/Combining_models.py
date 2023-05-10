from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt





class Make_predictions:
    def __init__(self, name=''):
        self.name = name

    def brain_scan_predictions(self):
        test_dir = "/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_test"

        test_datagen = ImageDataGenerator(rescale=1./255)

        binary_model = load_model('Tumor_detection.h5')

        multi_class_model = load_model('Tumor_identification.h5')

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(150, 150),
            batch_size=1,
            class_mode='binary',
            shuffle=False
        )

        predictions = binary_model.predict(test_generator)
        tumor_keys={0:'glioma', 1:'meningioma', 2:'pituitary_tumor'}
        for i in range(len(predictions)):
            if predictions[i] > 0.5:
                print(f"Image {i+1}: The brain scan is unhealthy.")
                img = tf.image.resize(test_generator[i][0], (224, 224))
                tumor_prediction = multi_class_model.predict(img,verbose=0)
                print('The predicted tumor type is:', tumor_keys[np.argmax(tumor_prediction)])

                print()
            else:
                print(f"Image {i+1}: The brain scan is healthy.")
