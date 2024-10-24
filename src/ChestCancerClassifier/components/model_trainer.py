import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path

class Training:
    def __init__(self, config):
        """
        Initializes the Training class with the provided configuration.
        
        Args:
            config (TrainingConfig): Configuration object containing paths and training parameters.
        """
        self.config = config

    def get_base_model(self):
        """
        Loads the base model from the updated base model path and applies L2 regularization 
        and Dropout to the layers to help prevent overfitting. Recompiles the model with a 
        reduced learning rate.
        """
        # Load the base model
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

        # Apply L2 regularization to layers if applicable
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = tf.keras.regularizers.l2(0.001)  # L2 Regularization

        # Recompile the model with a lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Reduce the learning rate
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

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

        # Apply data augmentation to the training data if enabled
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,          # Random rotation
                horizontal_flip=True,       # Random horizontal flip
                width_shift_range=0.2,      # Horizontal shift
                height_shift_range=0.2,     # Vertical shift
                shear_range=0.2,            # Shear transformation
                zoom_range=0.2,             # Random zoom
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator  # No augmentation for training

        # Training generator setup
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the trained model to the specified path.
        
        Args:
            path (Path): File path to save the model.
            model (tf.keras.Model): Trained model to save.
        """
        model.save(path)

    def train(self):
        """
        Trains the model using the training and validation generators. 
        Includes early stopping and learning rate reduction as callbacks to 
        prevent overfitting and improve performance.
        """
        # Define steps per epoch and validation steps
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Define callbacks to prevent overfitting and adjust learning rate
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

        # Train the model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=[early_stopping, reduce_lr]  # Apply callbacks
        )

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )