from preprocessing_data import Tumor_detection_preprocessing
from preprocessing_data import Tumor_type_preprocessing
from models import Train_tumor_detection_model
from models import Train_tumor_identification_model
from postprocessing_data import Combining_models
from visualization import visualize
def main():

    # Sort image data for tumor detection modwel
    # tumor_detection_pre=Tumor_detection_preprocessing.tumor_detection_preprocess()
    # tumor_detection_pre.detection_preprocess()

    # # Sort image data for tumor identification model
    # tumor_type_pre=Tumor_type_preprocessing.tumor_type_preprocess()
    # tumor_type_pre.type_preprocess()

    # # Create tumor detection model
    # detection_model=Train_tumor_detection_model.tumor_detection_model()
    # detection_model.train_detection_model()


    # # Create tumor identification model
    # identification_model=Train_tumor_identification_model.tumor_identifciation_model()
    # identification_model.train_identify_model()


    #Make predictions on a randomly created test set by combining both models
    given_prediction= Combining_models.Make_predictions()
    given_prediction.brain_scan_predictions()


    given_visualization=visualize.Visualize_performance()

    #Calculate AUC for models
    given_visualization.AUC_for_tumor_detection()
    given_visualization.AUC_for_tumor_identification()

    # Evaluate both models and return score
    given_visualization.evaluation_for_tumor_detection()
    given_visualization.evaluation_for_tumor_indentification()

    # Create ROC graph for both models
    given_visualization.Tumor_dectection_model_ROC_graph()
    given_visualization.Tumor_identification_model_ROC_graph()

    # Create Confusion matrices for both models
    given_visualization.generate_confusion_matrix_identification()
    given_visualization.generate_confusion_matrix_detection()

  


    return




main()