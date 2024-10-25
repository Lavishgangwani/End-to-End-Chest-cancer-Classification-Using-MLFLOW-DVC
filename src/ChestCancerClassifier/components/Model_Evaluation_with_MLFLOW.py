from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from ChestCancerClassifier.entity.config_entity import EvaluationConfig
from ChestCancerClassifier.utils.common import save_json
import tensorflow as tf



class Evaluation:
    def __init__(self, config:EvaluationConfig):
        self.config = config

    def train_valid_generator(self):
        """
        Sets up data generators for training and validation. 
        Applies data augmentation to the training data if enabled, 
        while validation data remains unaugmented.
        """
        # Common generator arguments
        datagenerator_kwargs = dict(
            rescale=1./255,          # Rescale images
            validation_split=0.20     # Split for validation (20%)
        )

        # Dataflow options
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Image size without channels
            batch_size=self.config.params_batch_size,        # Batch size
            interpolation="bilinear"                        # Interpolation method
        )

        # Validation data generator (no augmentation)
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Validation generator setup
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self.train_valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)


    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")