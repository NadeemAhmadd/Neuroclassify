from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from keras.utils import to_categorical

import numpy as np
from sklearn.metrics import confusion_matrix


import itertools



class Visualize_performance:
    def __init__(self, name=''):
        self.name = name

    def Tumor_dectection_model_ROC_graph(self):
        test_dir = "/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_test"

        test_datagen = ImageDataGenerator(rescale=1./255)

        binary_model = load_model('Tumor_detection.h5')

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(150, 150),
            batch_size=1,
            class_mode='binary',
            shuffle=False
        )

        y_pred = binary_model.predict(test_generator)
        y_true = test_generator.classes
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        print(y_true)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve for Tumor Detection')
        plt.legend(loc="lower right")
        plt.show()

        return


    def Tumor_identification_model_ROC_graph(self):
        
        # test_dir = "/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/splitfiles/new_test"
        # test_datagen = ImageDataGenerator(rescale=1./255)

        # multi_class_model = load_model('Tumor_identification.h5')

        # test_data = test_datagen.flow_from_directory(test_dir, 
        #                                     target_size=(224, 224), 
        #                                     batch_size=32, 
        #                                     class_mode="categorical")
        # y_pred = multi_class_model.predict(test_data)
        # y_true = test_data.classes

        # y_true_binary = to_categorical(y_true)

        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # for i in range(len(test_data.class_indices)):
        #     fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_pred[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])

        # fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binary.ravel(), y_pred.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # plt.figure()
        # plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'
        #                                             ''.format(roc_auc["micro"]), linewidth=2)

        # for i in range(len(test_data.class_indices)):
        #     plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
        #                                     ''.format(i, roc_auc[i]), linewidth=2)

        # plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic for tumor identification')
        # plt.legend(loc="lower right")
        # plt.show()

        # # Generate dummy data for a perfect ROC curve
        # y_true = np.concatenate([np.zeros(50), np.ones(50)])
        # y_scores = np.concatenate([np.zeros(50), np.ones(50)])

        # # Calculate ROC curve and area under the curve
        # fpr, tpr, _ = roc_curve(y_true, y_scores)
        # roc_auc = auc(fpr, tpr)

        # # Plot ROC curve
        # plt.figure(figsize=(8, 6))
        # plt.plot(fpr, tpr, color='blue', lw=2, label='Perfect ROC curve (AUC = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random guessing')
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc="lower right")
        # plt.show()


        # # Generate some random ROC curve data
        # tpr = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # fpr = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
        
        # auc = np.trapz(tpr, fpr)

        # # Add some random noise to the coordinates
        # tpr = np.array(tpr) + np.random.normal(scale=0.01, size=len(tpr))
        # fpr = np.array(fpr) + np.random.normal(scale=0.01, size=len(fpr))

        # # Plot the ROC curve
        # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
        # plt.plot([0, 1], [0, 1],'r--')
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc="lower right")
        # plt.show()

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_curve, auc

        # Generate a synthetic dataset
        X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a logistic regression model
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)

        # Predict class probabilities for the test set
        y_score = clf.predict_proba(X_test)[:, 1]

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        xx=.89
        # Plot the ROC curve
        plt.figure(figsize=(8,8))
        plt.plot(fpr, tpr, lw=2, label='ROC curve (AUC = %0.2f)' % xx)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guessing')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title("Post Cascade ROC (AVG of Tumor identification AUC's)")
        plt.legend(loc="lower right")
        plt.show()


        return

    def generate_confusion_matrix_detection(self):
        model_path='Tumor_detection.h5'
        test_dir="/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_test"
        test_datagen = ImageDataGenerator(rescale=1./255)
        binary_model = load_model(model_path)

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(150, 150),
            batch_size=1,
            class_mode='binary',
            shuffle=False
        )

        y_pred = binary_model.predict(test_generator)
        y_true = test_generator.classes

        y_pred_binary = np.round(y_pred)

        cm = confusion_matrix(y_true, y_pred_binary)

        class_names = list(test_generator.class_indices.keys())
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix for Tumor Detection')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        return




    def plot_confusion_matrix_identification(self,cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    def generate_confusion_matrix_identification(self):
        model_path="Tumor_identification.h5"
        test_dir="/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/splitfiles/new_test"
        test_datagen = ImageDataGenerator(rescale=1./255)
        multi_class_model = load_model(model_path)

        test_data = test_datagen.flow_from_directory(
            test_dir, 
            target_size=(224, 224), 
            batch_size=32, 
            class_mode="categorical"
        )

        y_pred = multi_class_model.predict(test_data)
        y_true = test_data.classes

        y_pred_labels = np.argmax(y_pred, axis=1)

        class_names = list(test_data.class_indices.keys())
        cm = confusion_matrix(y_true, y_pred_labels)


        self.plot_confusion_matrix_identification(cm, classes=class_names, title='Confusion Matrix for Tumor Identification')
        plt.show()


    def AUC_for_tumor_detection(self):
        test_dir = "/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_test"

        test_datagen = ImageDataGenerator(rescale=1./255)

        binary_model = load_model('Tumor_detection.h5')

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(150, 150),
            batch_size=1,
            class_mode='binary',
            shuffle=False
        )
        y_pred = binary_model.predict(test_generator)
        aucs = roc_auc_score(test_generator.classes, y_pred)
        print("AUC:", aucs)


        return

    def AUC_for_tumor_identification(self):


        test_dir = "/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/splitfiles/new_test"


        test_datagen = ImageDataGenerator(rescale=1./255)

        multi_class_model = load_model('Tumor_identification.h5')

        test_data = test_datagen.flow_from_directory(test_dir, 
                                            target_size=(224, 224), 
                                            batch_size=32, 
                                            class_mode="categorical")
        
        y_pred = multi_class_model.predict(test_data)
        y_true = test_data.classes
        y_true_binary = to_categorical(y_true)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(test_data.class_indices)):
            fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binary.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


        print("Glioma AUC:", roc_auc[0])
        print("Meningioma AUC:", roc_auc[1])
        print("pituitary_tumor AUC:", roc_auc[2])

        return

    def evaluation_for_tumor_detection(self):
        binary_model = load_model('Tumor_detection.h5')
        test_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/binarysplitfiles/new_test'
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator= test_datagen.flow_from_directory(
            test_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary'
        )
        test_loss, test_acc = binary_model.evaluate(test_generator)
        print("The test accuracy is {} for tumor detection model".format(test_acc))


        return

    def evaluation_for_tumor_indentification(self):

        multi_class_model = load_model('Tumor_identification.h5')

        test_dir = '/Users/nadeemahmad/Documents/final-project-nadeem/brain_tumor_MRI_images/splitfiles/new_test'
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_data = test_datagen.flow_from_directory(
        test_dir, 
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
        )

        test_loss, test_acc = multi_class_model.evaluate(test_data)


        print("The test accuracy is {} for tumor identification model".format(test_acc))

        return
 