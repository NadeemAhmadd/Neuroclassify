import os
import shutil
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator

class tumor_type_preprocess:
    def __init__(self, name=''):
        self.name = name

    def type_preprocess(self):

        data_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/largefiles/tumor'
        train_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/splitfiles/new_train'
        val_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/splitfiles/new_val'
        test_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/splitfiles/new_test'


        for dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(dir, 'glioma'))
            os.makedirs(os.path.join(dir, 'meningioma'))
            os.makedirs(os.path.join(dir, 'pituitary_tumor'))


        for tumor_type in ['glioma', 'meningioma', 'pituitary_tumor']:
            tumor_dir = os.path.join(data_dir, tumor_type)
            images = os.listdir(tumor_dir)
            train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
            train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)
            
            print("Writing train images for {}".format(tumor_type))
            for image in train_images:
                src_path = os.path.join(tumor_dir, image)
                dest_path = os.path.join(train_dir, tumor_type, image)
                shutil.copy(src_path, dest_path)
            print("Writing validation images for {}".format(tumor_type))   
            for image in val_images:
                src_path = os.path.join(tumor_dir, image)
                dest_path = os.path.join(val_dir, tumor_type, image)
                shutil.copy(src_path, dest_path)

            print("Writing test images for {}".format(tumor_type))  
            for image in test_images:
                src_path = os.path.join(tumor_dir, image)
                dest_path = os.path.join(test_dir, tumor_type, image)
                shutil.copy(src_path, dest_path)


        
        return
    
    def remove_directories(self):
        train_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/splitfiles/new_train'
        val_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/splitfiles/new_val'
        test_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/splitfiles/new_test'


        for dir in [train_dir, val_dir, test_dir]:
            os.rmdir(os.path.join(dir, 'glioma'))
            os.rmdir(os.path.join(dir, 'meningioma'))
            os.rmdir(os.path.join(dir, 'pituitary_tumor'))

        return