# quantum_ml_simulation/ml_model/trainer.py
# Defines and trains the Dense Neural Network using TensorFlow/Keras

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
import numpy as np
import os
import time

# Use relative import
from ..config import simulation_params as cfg

class ModelTrainer:
    """Defines, trains, and handles the DNN model using Keras."""

    def __init__(self, input_dim: int, output_dim: int = cfg.ML_MODEL_PARAMS['output_dim']):
        """
        Initializes the ModelTrainer.

        Args:
            input_dim: The number of input features.
            output_dim: The number of output neurons (usually 1).
        """
        if input_dim <= 0:
            raise ValueError("Input dimension must be positive.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._build_model()
        print(f"Initialized ModelTrainer:")
        print(f"  Input Dim: {self.input_dim}, Output Dim: {self.output_dim}")
        # self.model.summary() # Optional: Print model summary

    def _build_model(self) -> keras.Sequential:
        """Defines the DNN architecture using Keras Sequential API."""
        model = keras.Sequential(name="QuantumSim_DNN")
        model.add(layers.InputLayer(input_shape=(self.input_dim,), name="Input"))

        # Normalization Layer: Crucial for stable training
        model.add(layers.Normalization(axis=-1, name="Normalization"))

        # Hidden Layers based on config
        hidden_layers = cfg.ML_MODEL_PARAMS.get("hidden_layers", [64, 32])
        activation = cfg.ML_MODEL_PARAMS.get("activation", "relu")
        for i, units in enumerate(hidden_layers):
            if units <= 0: continue # Skip non-positive layer sizes
            model.add(layers.Dense(units, activation=activation, name=f"Dense_{i+1}"))
            # Optional: Add Dropout for regularization
            # model.add(layers.Dropout(0.2))

        # Output Layer
        model.add(layers.Dense(self.output_dim, name="Output")) # Linear activation for regression

        # Compile Model
        optimizer_name = cfg.ML_MODEL_PARAMS.get("optimizer", "adam")
        learning_rate = cfg.ML_MODEL_PARAMS.get("learning_rate", 0.001)
        loss_function = cfg.ML_MODEL_PARAMS.get("loss", "mean_squared_error")

        if optimizer_name.lower() == "adam":
             optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == "sgd":
             optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        # Add other optimizers if needed
        else:
             print(f"Warning: Unknown optimizer '{optimizer_name}'. Defaulting to Adam.")
             optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss=loss_function, metrics=['mae', 'mse']) # Add metrics
        print(f"Model compiled with Optimizer: {optimizer_name}, LR: {learning_rate}, Loss: {loss_function}")
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> keras.callbacks.History:
        """
        Trains the Keras model on the provided data.

        Args:
            X_train: Training features.
            y_train: Training target values.
            X_val: Validation features.
            y_val: Validation target values.

        Returns:
            Keras History object containing training metrics.
        """
        if self.model is None:
            raise RuntimeError("Model has not been built or loaded.")
        if X_train is None or y_train is None or X_val is None or y_val is None:
             raise ValueError("Training/Validation data cannot be None.")
        if len(X_train) == 0 or len(X_val) == 0:
             raise ValueError("Training/Validation sets cannot be empty.")


        print(f"\n--- Starting Model Training ---")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Adapt the Normalization layer using *only* the training data
        print("Adapting normalization layer...")
        start_adapt_time = time.time()
        norm_layer = self.model.get_layer('Normalization')
        if norm_layer:
            norm_layer.adapt(X_train)
            print(f"Normalization layer adapted in {time.time() - start_adapt_time:.2f}s")
        else:
             print("Warning: Normalization layer not found in the model.")

        # Configure Callbacks
        epochs = cfg.ML_MODEL_PARAMS.get("epochs", 100)
        batch_size = cfg.ML_MODEL_PARAMS.get("batch_size", 32)

        # Early Stopping: Stop training if validation loss doesn't improve
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,         # Increase patience slightly
            restore_best_weights=True,
            verbose=1
        )

        # Reduce Learning Rate on Plateau: Adjust LR if progress stalls
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,         # Reduce LR by factor of 5
            patience=10,         # Wait 10 epochs before reducing
            min_lr=1e-6,        # Minimum learning rate
            verbose=1
        )

        callbacks = [early_stopping, reduce_lr]

        # Train the model
        print(f"Training for up to {epochs} epochs with batch size {batch_size}...")
        start_train_time = time.time()
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=2 # 0=silent, 1=progress bar, 2=one line per epoch
        )
        print(f"--- Model Training Finished ({time.time() - start_train_time:.2f}s) ---")

        # Find the epoch with the best validation loss
        if history and 'val_loss' in history.history:
             best_epoch = np.argmin(history.history['val_loss'])
             best_val_loss = history.history['val_loss'][best_epoch]
             print(f"Best validation loss ({best_val_loss:.6f}) achieved at epoch {best_epoch + 1}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the trained model."""
        if self.model is None:
            raise RuntimeError("Model has not been built or loaded.")
        return self.model.predict(X)

    def save_model(self, filepath: str):
        """Saves the trained Keras model."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath) # Saves in Keras native format (recommended)
            print(f"Model saved successfully to: {filepath}")
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")

    def load_model(self, filepath: str) -> bool:
        """Loads a pre-trained Keras model."""
        if not os.path.exists(filepath):
            print(f"Model file not found at: {filepath}")
            return False
        try:
            self.model = keras.models.load_model(filepath)
            # Update input_dim based on loaded model if necessary (optional check)
            loaded_input_shape = self.model.input_shape
            if loaded_input_shape and len(loaded_input_shape) > 1 and loaded_input_shape[1] is not None:
                 loaded_input_dim = loaded_input_shape[1]
                 if loaded_input_dim != self.input_dim:
                      print(f"Warning: Loaded model input dim ({loaded_input_dim}) differs from initialized dim ({self.input_dim}). Using loaded dim.")
                      self.input_dim = loaded_input_dim
            else:
                 print(f"Warning: Could not verify input dimension from loaded model shape {loaded_input_shape}")

            print(f"Model loaded successfully from: {filepath}")
            # self.model.summary() # Optional: Show summary after loading
            return True
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            return False