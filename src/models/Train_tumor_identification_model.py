
import os
import shutil
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
from visualization import visualize



class tumor_identifciation_model:
    def __init__(self, name=''):
        self.name = name

    def train_indentify_model(self):

        data_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/largefiles/tumor'
        train_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/splitfiles/new_train'
        val_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/splitfiles/new_val'
        test_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/splitfiles/new_test'


        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_data = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
        val_data = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
        test_data = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

        print('Class indices:', test_data.class_indices)


        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(train_data, epochs=10, validation_data=val_data)

        test_loss, test_acc = model.evaluate(test_data)
        val_loss, val_acc = model.evaluate(val_data)

        print(test_loss)
        print(test_acc)

        print("--------------")
        print(val_loss)
        print(val_acc)

        print("--------------")

        for epoch in range(10):
            print(f"Epoch {epoch+1}: loss = {history.history['loss'][epoch]}, accuracy = {history.history['accuracy'][epoch]}")



        model.save('Tumor_detection.h5')






        