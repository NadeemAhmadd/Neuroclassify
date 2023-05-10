#NeuroClassify

Project Instructions
==============================

This repo contains the instructions for a machine learning project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── README.md      <- Youtube Video Link
    │   └── final_project_report <- final report .pdf format and supporting files
    │   └── presentation   <-  final power point presentation 
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── preprocessing data           <- Scripts to download or generate data and pre-process the data
       │   └── Tumor_detection_preprocessing.py <- preproccesses data and sorts  into folders for detection model
       │   └── Tumor_type_preprocessing.py   <- preproccesses data and sorts into folders for type model
       │
       ├── models          <- Scripts to generate models
       │   └── Train_tumor_detection_model.py <- script trains and saves a tumor detection model
       │   └── Train_tumor_identification_model.py <- script trains and saces a tumor indentification model
       │
       │
       ├── postprocessing data           <- Scripts to combine predictions made by the models
       │   └── Combining_models.py  <- Takes the predictions of detection model and feeds it into detection model 
       │
       │
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py     <- Contains different functions to highlight data returned from model and results such as evaluations     


NOTE : The actual models once made are run using Combining_models.py. This is whats used to get a response from the model on a particular image. Evaluation functions are housed in visualize.py
# Brain Tumor Classification with machine learning

## Abstract

Magnetic Resonance Imaging (MRI) is a non-invasive medical imaging technique used to diagnose various medical conditions, including brain tumors. However, the interpretation of MRI scans can be challenging, requiring a great deal of expertise, time, and effort from radiologists and other healthcare professionals. The use of machine learning (ML) algorithms can aid in the accurate identification and classification of brain tumors, potentially leading to earlier diagnosis and improved patient outcomes.

The main objective of this project is to develop an ML model that can accurately classify different types of brain tumors using MRI scans. The model will be trained using a large dataset of MRI images, including various types of brain tumors such as gliomas, meningiomas, and pituitary adenomas. We aim to achieve high accuracy in tumor classification to assist medical professionals in the identification of brain tumors and improve patient outcomes.

I plan to use Convolutional Neural Networks (CNNs), which have shown promising results in medical image analysis tasks. The dataset will be pre-processed to ensure high-quality images for training and validation. I will evaluate the performance of the model using metrics such as accuracy, precision, recall, and F1 score. One key thing to note is that the images will be kept to greyscale as colour is not provided usually during MRI scans. This decision will also allow a reduction in computation requirements and training time.

In summary, this project aims to develop a machine learning model that can accurately classify different types of brain tumors using MRI scans. The proposed methodology involves the use of CNNs and transfer learning techniques to train the model, with the goal of achieving high accuracy and aiding medical professionals in the diagnosis of brain tumors. This project has the potential to improve patient outcomes and reduce medical errors, contributing to better healthcare delivery.
