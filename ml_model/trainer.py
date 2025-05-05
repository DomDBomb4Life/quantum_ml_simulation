# quantum_ml_simulation/ml_model/trainer.py
# Defines and trains the DNN model using Keras (Handles Vector Output)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import numpy as np
import os
import time

# Use relative import
from ..config import simulation_params as cfg

class ModelTrainer:
    """Defines, trains, and handles the DNN model using Keras."""

    # --- MODIFIED: Accepts output_dim explicitly ---
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initializes the ModelTrainer.

        Args:
            input_dim: The number of input features.
            output_dim: The number of output neurons (size of the target vector).
        """
        if input_dim <= 0: raise ValueError("Input dimension must be positive.")
        if output_dim <= 0: raise ValueError("Output dimension must be positive.")

        self.input_dim = input_dim
        self.output_dim = output_dim # Store the passed output dimension

        # Read other hyperparameters from config (can be overridden if needed)
        self.ml_config = cfg.DEFAULT_ML_PARAMS.copy()

        self.model = self._build_model()
        print(f"Initialized ModelTrainer:")
        print(f"  Input Dim: {self.input_dim}, Output Dim: {self.output_dim}")
        self.model.summary(print_fn=lambda x: print(f"  {x}")) # Print summary with indentation


    # --- MODIFIED: Uses self.output_dim and reads config ---
    def _build_model(self) -> keras.Sequential:
        """Defines the DNN architecture using Keras Sequential API."""
        model = keras.Sequential(name="QuantumSim_DNN_Vector")
        model.add(layers.InputLayer(input_shape=(self.input_dim,), name="Input"))
        model.add(layers.Normalization(axis=-1, name="Normalization"))

        hidden_layers = self.ml_config.get("hidden_layers", [64, 32])
        activation = self.ml_config.get("activation", "leaky_relu")
        for i, units in enumerate(hidden_layers):
            if units <= 0: continue
            model.add(layers.Dense(units, activation=activation, name=f"Dense_{i+1}"))

        # Output Layer uses self.output_dim
        model.add(layers.Dense(self.output_dim, name="Output")) # Linear activation for regression

        # Compile Model using config parameters
        optimizer_name = self.ml_config.get("optimizer", "adam")
        learning_rate = self.ml_config.get("learning_rate", 0.001)
        loss_function = self.ml_config.get("loss", "mean_squared_error")

        optimizer_instance = None
        if optimizer_name.lower() == "adam":
             optimizer_instance = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == "sgd":
             optimizer_instance = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
             print(f"Warning: Unknown optimizer '{optimizer_name}'. Defaulting to Adam.")
             optimizer_instance = keras.optimizers.Adam(learning_rate=learning_rate)

        # MSE loss works for multi-output regression
        model.compile(optimizer=optimizer_instance, loss=loss_function, metrics=['mae', 'mse'])
        print(f"Model compiled with Optimizer: {optimizer_name}, LR: {learning_rate}, Loss: {loss_function}")
        return model

    # --- train method signature is fine, handles 2D y_train/y_val ---
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> keras.callbacks.History:
        """
        Trains the Keras model. y_train/y_val are now 2D arrays.
        """
        if self.model is None: raise RuntimeError("Model not built.")
        if X_train is None or y_train is None or X_val is None or y_val is None: raise ValueError("Data cannot be None.")
        if len(X_train) == 0 or len(X_val) == 0: raise ValueError("Training/Validation sets empty.")
        if y_train.shape[1] != self.output_dim or y_val.shape[1] != self.output_dim:
             raise ValueError(f"y_train/y_val shape mismatch. Expected {self.output_dim} outputs, "
                              f"got y_train: {y_train.shape}, y_val: {y_val.shape}")


        print(f"\n--- Starting Model Training ---")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")

        # Adapt Normalization Layer
        print("Adapting normalization layer...")
        start_adapt_time = time.time()
        norm_layer = self.model.get_layer('Normalization')
        if norm_layer: norm_layer.adapt(X_train)
        else: print("Warning: Normalization layer not found.")
        print(f"Normalization layer adapted in {time.time() - start_adapt_time:.2f}s")

        # Callbacks from config
        epochs = self.ml_config.get("epochs", 100)
        batch_size = self.ml_config.get("batch_size", 32)
        es_patience = self.ml_config.get("early_stopping_patience", 15)
        lr_patience = self.ml_config.get("reduce_lr_patience", 7)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=es_patience, restore_best_weights=True, verbose=1)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=lr_patience, min_lr=1e-6, verbose=1)
        callbacks = [early_stopping, reduce_lr]

        # Train the model
        print(f"Training for up to {epochs} epochs with batch size {batch_size}...")
        start_train_time = time.time()
        history = self.model.fit(
            X_train,
            y_train, # Should be 2D numpy array
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val), # y_val should be 2D numpy array
            callbacks=callbacks,
            verbose=2
        )
        print(f"--- Model Training Finished ({time.time() - start_train_time:.2f}s) ---")

        if history and 'val_loss' in history.history:
             best_epoch = np.argmin(history.history['val_loss'])
             best_val_loss = history.history['val_loss'][best_epoch]
             print(f"Best validation loss ({best_val_loss:.6f}) achieved at epoch {best_epoch + 1}")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions (returns 2D array)."""
        if self.model is None: raise RuntimeError("Model not built.")
        return self.model.predict(X) # Output will be shape (num_samples, output_dim)

    def save_model(self, filepath: str):
        """Saves the trained Keras model."""
        # (Method unchanged)
        if self.model is None: raise RuntimeError("No model to save.")
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            print(f"Model saved successfully to: {filepath}")
        except Exception as e: print(f"Error saving model to {filepath}: {e}")

    def load_model(self, filepath: str) -> bool:
        """Loads a pre-trained Keras model."""
        # (Method largely unchanged, but verify output dim if possible)
        if not os.path.exists(filepath): print(f"Model file not found at: {filepath}"); return False
        try:
            self.model = keras.models.load_model(filepath)
            loaded_input_shape = self.model.input_shape
            loaded_output_shape = self.model.output_shape # Check output shape

            # Verify input dimension
            if loaded_input_shape and len(loaded_input_shape) > 1 and loaded_input_shape[1] is not None:
                 loaded_input_dim = loaded_input_shape[1]
                 if loaded_input_dim != self.input_dim:
                      print(f"Warning: Loaded model input dim ({loaded_input_dim}) differs from expected ({self.input_dim}). Using loaded.")
                      self.input_dim = loaded_input_dim
            else: print(f"Warning: Could not verify input dimension from loaded model shape {loaded_input_shape}")

            # Verify output dimension
            if loaded_output_shape and len(loaded_output_shape) > 1 and loaded_output_shape[1] is not None:
                 loaded_output_dim = loaded_output_shape[1]
                 if loaded_output_dim != self.output_dim:
                      print(f"Warning: Loaded model output dim ({loaded_output_dim}) differs from expected ({self.output_dim}). Using loaded.")
                      self.output_dim = loaded_output_dim
            else: print(f"Warning: Could not verify output dimension from loaded model shape {loaded_output_shape}")


            print(f"Model loaded successfully from: {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            return False