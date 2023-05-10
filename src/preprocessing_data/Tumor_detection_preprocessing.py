import os
import random
import shutil
from sklearn.model_selection import train_test_split

class tumor_detection_preprocess:
    def __init__(self, name=''):
        self.name = name

    def detection_preprocess(self):
        healthy_data_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/largefiles/no_tumor'
        unhealthy_data_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/largefiles/all_tumor_combined'
        train_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_train'
        val_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_val'
        test_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_test'
        data_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles'



        for dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(dir, 'healthy'))
            os.makedirs(os.path.join(dir, 'unhealthy'))


        for tumor_type in ['healthy']:
            tumor_dir = os.path.join(healthy_data_dir, tumor_type)
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





        for tumor_type in ['unhealthy']:
            tumor_dir = os.path.join(unhealthy_data_dir, tumor_type)
            images = os.listdir(tumor_dir)
            val_images = random.sample(images, 600)
            train_images, test_images = train_test_split(val_images, test_size=0.2, random_state=42)
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
       
        train_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_train'
        val_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_val'
        test_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_test'


        for dir in [train_dir, val_dir, test_dir]:
            os.rmdir(os.path.join(dir, 'healthy'))
            os.rmdir(os.path.join(dir, 'unhealthy'))
            
        return