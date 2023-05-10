
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



class tumor_detection_model:
    def __init__(self, name=''):
        self.name = name

    def train_detection_model(self):
       
        train_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_train'
        val_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_val'
        test_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_test'



        img_width, img_height = 150, 150

        batch_size = 32

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1, 
            height_shift_range=0.1,
            shear_range=0.1, 
            zoom_range=0.1, 
            horizontal_flip=True, 
            fill_mode='nearest' 
        )
        val_datagen = ImageDataGenerator(
            rescale=1./255
        )

        test_datagen = ImageDataGenerator(rescale=1./255)


        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary'
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary'
        )

        test_generator= test_datagen.flow_from_directory(
            test_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary'
        )

        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples//batch_size,
            epochs=50,
            validation_data=val_generator,
            validation_steps=val_generator.samples//batch_size
        )


        test_loss, test_acc = model.evaluate(test_generator)
        val_loss, val_acc = model.evaluate(val_generator)

        print(test_loss)
        print(test_acc)

        print("--------------")
        print(val_loss)
        print(val_acc)

        print("--------------")

        for epoch in range(10):
            print(f"Epoch {epoch+1}: loss = {history.history['loss'][epoch]}, accuracy = {history.history['accuracy'][epoch]}")



        model.save('Tumor_detection.h5')


        return
    
