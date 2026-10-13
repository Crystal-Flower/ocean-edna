# import os
# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # Deep Learning Libraries
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import (
#     MobileNetV2, DenseNet121, EfficientNetB0,
#     VGG16, ResNet50V2
# )
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.utils.class_weight import compute_class_weight
#
# # Vision Transformer (requires tensorflow-addons)
# try:
#     import tensorflow_addons as tfa
#
#     ViT_AVAILABLE = True
# except ImportError:
#     print("Warning: tensorflow-addons not available. ViT model will be skipped.")
#     ViT_AVAILABLE = False
#
#
# class FishClassifier:
#     def __init__(self, data_dir, results_dir="classification_results", img_size=(224, 224), batch_size=32):
#         """
#         Initialize the Fish Classifier
#
#         Args:
#             data_dir (str): Path to the dataset directory
#             results_dir (str): Directory to save results
#             img_size (tuple): Input image size
#             batch_size (int): Batch size for training
#         """
#         self.data_dir = data_dir
#         self.results_dir = results_dir
#         self.img_size = img_size
#         self.batch_size = batch_size
#         self.models = {}
#         self.histories = {}
#         self.results = {}
#
#         # Create results directory
#         os.makedirs(self.results_dir, exist_ok=True)
#
#         # Setup GPU if available
#         self.setup_gpu()
#
#     def setup_gpu(self):
#         """Configure GPU settings"""
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         if gpus:
#             try:
#                 for gpu in gpus:
#                     tf.config.experimental.set_memory_growth(gpu, True)
#                 print(f"GPU available: {len(gpus)} GPU(s)")
#             except RuntimeError as e:
#                 print(f"GPU setup error: {e}")
#         else:
#             print("No GPU available, using CPU")
#
#     def prepare_data(self, validation_split=0.2, test_split=0.1):
#         """
#         Prepare data generators with augmentation
#         """
#         print("Preparing data generators...")
#
#         # Data augmentation for training
#         train_datagen = ImageDataGenerator(
#             rescale=1. / 255,
#             rotation_range=20,
#             width_shift_range=0.2,
#             height_shift_range=0.2,
#             horizontal_flip=True,
#             zoom_range=0.2,
#             fill_mode='nearest',
#             validation_split=validation_split + test_split
#         )
#
#         # No augmentation for validation/test
#         val_test_datagen = ImageDataGenerator(
#             rescale=1. / 255,
#             validation_split=validation_split / (validation_split + test_split)
#         )
#
#         # Training data
#         self.train_generator = train_datagen.flow_from_directory(
#             self.data_dir,
#             target_size=self.img_size,
#             batch_size=self.batch_size,
#             class_mode='categorical',
#             subset='training'
#         )
#
#         # Validation data
#         self.val_generator = train_datagen.flow_from_directory(
#             self.data_dir,
#             target_size=self.img_size,
#             batch_size=self.batch_size,
#             class_mode='categorical',
#             subset='validation'
#         )
#
#         # Test data (using a separate generator)
#         self.test_generator = val_test_datagen.flow_from_directory(
#             self.data_dir,
#             target_size=self.img_size,
#             batch_size=self.batch_size,
#             class_mode='categorical',
#             shuffle=False
#         )
#
#         self.num_classes = self.train_generator.num_classes
#         self.class_names = list(self.train_generator.class_indices.keys())
#
#         print(f"Number of classes: {self.num_classes}")
#         print(f"Class names: {self.class_names}")
#         print(f"Training samples: {self.train_generator.samples}")
#         print(f"Validation samples: {self.val_generator.samples}")
#         print(f"Test samples: {self.test_generator.samples}")
#
#         # Compute class weights for imbalanced datasets
#         self.compute_class_weights()
#
#     def compute_class_weights(self):
#         """Compute class weights to handle imbalanced datasets"""
#         labels = self.train_generator.labels
#         class_weights = compute_class_weight(
#             'balanced',
#             classes=np.unique(labels),
#             y=labels
#         )
#         self.class_weight = dict(enumerate(class_weights))
#         print("Class weights computed for balanced training")
#
#     def create_model(self, model_name, input_shape=(224, 224, 3)):
#         """
#         Create transfer learning model
#
#         Args:
#             model_name (str): Name of the model architecture
#             input_shape (tuple): Input shape
#
#         Returns:
#             keras.Model: Compiled model
#         """
#         print(f"Creating {model_name} model...")
#
#         if model_name == 'MobileNetV2':
#             base_model = MobileNetV2(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=input_shape
#             )
#         elif model_name == 'DenseNet121':
#             base_model = DenseNet121(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=input_shape
#             )
#         elif model_name == 'EfficientNetB0':
#             base_model = EfficientNetB0(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=input_shape
#             )
#         elif model_name == 'VGG16':
#             base_model = VGG16(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=input_shape
#             )
#         elif model_name == 'ResNet50V2':
#             base_model = ResNet50V2(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=input_shape
#             )
#         elif model_name == 'ViT' and ViT_AVAILABLE:
#             # Vision Transformer implementation
#             return self.create_vit_model(input_shape)
#         else:
#             raise ValueError(f"Model {model_name} not supported or not available")
#
#         # Freeze base model initially
#         base_model.trainable = False
#
#         # Add custom classification head
#         inputs = keras.Input(shape=input_shape)
#         x = base_model(inputs, training=False)
#         x = layers.GlobalAveragePooling2D()(x)
#         x = layers.Dropout(0.3)(x)
#         x = layers.Dense(512, activation='relu')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Dropout(0.5)(x)
#         outputs = layers.Dense(self.num_classes, activation='softmax')(x)
#
#         model = keras.Model(inputs, outputs)
#
#         # Compile model
#         model.compile(
#             optimizer=keras.optimizers.Adam(learning_rate=0.001),
#             loss='categorical_crossentropy',
#             metrics=['accuracy', 'top_3_accuracy']
#         )
#
#         return model
#
#     def create_vit_model(self, input_shape):
#         """Create Vision Transformer model (simplified version)"""
#         if not ViT_AVAILABLE:
#             raise ImportError("tensorflow-addons required for ViT model")
#
#         # Simplified ViT implementation
#         inputs = keras.Input(shape=input_shape)
#
#         # Patch extraction and embedding
#         patch_size = 16
#         num_patches = (input_shape[0] // patch_size) ** 2
#         projection_dim = 768
#
#         # Extract patches
#         patches = layers.Conv2D(projection_dim, patch_size, strides=patch_size)(inputs)
#         patches = layers.Reshape((num_patches, projection_dim))(patches)
#
#         # Position embedding
#         positions = tf.range(start=0, limit=num_patches, delta=1)
#         pos_embed = layers.Embedding(num_patches, projection_dim)(positions)
#         encoded_patches = patches + pos_embed
#
#         # Transformer blocks
#         for _ in range(8):
#             # Multi-head attention
#             attention = layers.MultiHeadAttention(
#                 num_heads=8, key_dim=projection_dim // 8
#             )(encoded_patches, encoded_patches)
#             attention = layers.Dropout(0.1)(attention)
#
#             # Skip connection
#             x1 = layers.Add()([attention, encoded_patches])
#             x1 = layers.LayerNormalization()(x1)
#
#             # MLP
#             mlp = layers.Dense(projection_dim * 2, activation='gelu')(x1)
#             mlp = layers.Dropout(0.1)(mlp)
#             mlp = layers.Dense(projection_dim)(mlp)
#
#             # Skip connection
#             encoded_patches = layers.Add()([mlp, x1])
#             encoded_patches = layers.LayerNormalization()(encoded_patches)
#
#         # Classification head
#         x = layers.GlobalAveragePooling1D()(encoded_patches)
#         x = layers.Dropout(0.3)(x)
#         outputs = layers.Dense(self.num_classes, activation='softmax')(x)
#
#         model = keras.Model(inputs, outputs)
#         model.compile(
#             optimizer=keras.optimizers.Adam(learning_rate=0.001),
#             loss='categorical_crossentropy',
#             metrics=['accuracy', 'top_3_accuracy']
#         )
#
#         return model
#
#     def train_model(self, model_name, epochs=50, fine_tune=True, fine_tune_epochs=20):
#         """
#         Train a model with transfer learning
#
#         Args:
#             model_name (str): Model architecture name
#             epochs (int): Initial training epochs
#             fine_tune (bool): Whether to fine-tune the base model
#             fine_tune_epochs (int): Fine-tuning epochs
#         """
#         print(f"\n{'=' * 60}")
#         print(f"Training {model_name}")
#         print(f"{'=' * 60}")
#
#         # Create model
#         model = self.create_model(model_name, (*self.img_size, 3))
#
#         # Callbacks
#         callbacks = [
#             EarlyStopping(
#                 monitor='val_loss',
#                 patience=10,
#                 restore_best_weights=True,
#                 verbose=1
#             ),
#             ReduceLROnPlateau(
#                 monitor='val_loss',
#                 factor=0.2,
#                 patience=5,
#                 min_lr=1e-7,
#                 verbose=1
#             ),
#             ModelCheckpoint(
#                 filepath=os.path.join(self.results_dir, f'{model_name}_best.h5'),
#                 monitor='val_accuracy',
#                 save_best_only=True,
#                 verbose=1
#             )
#         ]
#
#         # Initial training
#         print(f"Phase 1: Training with frozen base model ({epochs} epochs)")
#         history1 = model.fit(
#             self.train_generator,
#             epochs=epochs,
#             validation_data=self.val_generator,
#             callbacks=callbacks,
#             class_weight=self.class_weight,
#             verbose=1
#         )
#
#         # Fine-tuning
#         history2 = None
#         if fine_tune and model_name != 'ViT':
#             print(f"Phase 2: Fine-tuning ({fine_tune_epochs} epochs)")
#
#             # Unfreeze base model
#             base_model = model.layers[1]  # Assuming base model is the second layer
#             base_model.trainable = True
#
#             # Use lower learning rate for fine-tuning
#             model.compile(
#                 optimizer=keras.optimizers.Adam(learning_rate=0.0001),
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy', 'top_3_accuracy']
#             )
#
#             # Update callbacks for fine-tuning
#             callbacks[2] = ModelCheckpoint(
#                 filepath=os.path.join(self.results_dir, f'{model_name}_finetuned.h5'),
#                 monitor='val_accuracy',
#                 save_best_only=True,
#                 verbose=1
#             )
#
#             history2 = model.fit(
#                 self.train_generator,
#                 epochs=fine_tune_epochs,
#                 validation_data=self.val_generator,
#                 callbacks=callbacks,
#                 class_weight=self.class_weight,
#                 verbose=1
#             )
#
#         # Combine histories
#         if history2:
#             combined_history = {}
#             for key in history1.history.keys():
#                 combined_history[key] = history1.history[key] + history2.history[key]
#         else:
#             combined_history = history1.history
#
#         # Store results
#         self.models[model_name] = model
#         self.histories[model_name] = combined_history
#
#         # Evaluate on test set
#         self.evaluate_model(model_name)
#
#         print(f"Completed training {model_name}")
#
#     def evaluate_model(self, model_name):
#         """Evaluate model on test set and store results"""
#         print(f"Evaluating {model_name}...")
#
#         model = self.models[model_name]
#
#         # Predictions
#         predictions = model.predict(self.test_generator)
#         predicted_classes = np.argmax(predictions, axis=1)
#         true_classes = self.test_generator.classes
#
#         # Metrics
#         test_loss, test_accuracy, test_top3 = model.evaluate(
#             self.test_generator, verbose=0
#         )
#
#         # Classification report
#         class_report = classification_report(
#             true_classes,
#             predicted_classes,
#             target_names=self.class_names,
#             output_dict=True
#         )
#
#         # Confusion matrix
#         cm = confusion_matrix(true_classes, predicted_classes)
#
#         # Store results
#         self.results[model_name] = {
#             'test_loss': test_loss,
#             'test_accuracy': test_accuracy,
#             'test_top3_accuracy': test_top3,
#             'classification_report': class_report,
#             'confusion_matrix': cm,
#             'predictions': predictions,
#             'predicted_classes': predicted_classes,
#             'true_classes': true_classes
#         }
#
#         print(f"{model_name} - Test Accuracy: {test_accuracy:.4f}, Top-3 Accuracy: {test_top3:.4f}")
#
#     def train_all_models(self, models_to_train=None, epochs=50, fine_tune=True):
#         """
#         Train multiple models
#
#         Args:
#             models_to_train (list): List of model names to train
#             epochs (int): Training epochs
#             fine_tune (bool): Whether to fine-tune
#         """
#         if models_to_train is None:
#             models_to_train = ['MobileNetV2', 'DenseNet121', 'EfficientNetB0', 'VGG16', 'ResNet50V2']
#             if ViT_AVAILABLE:
#                 models_to_train.append('ViT')
#
#         print(f"Training models: {models_to_train}")
#
#         for model_name in models_to_train:
#             try:
#                 self.train_model(model_name, epochs=epochs, fine_tune=fine_tune)
#             except Exception as e:
#                 print(f"Error training {model_name}: {e}")
#                 continue
#
#     def plot_training_history(self, save=True):
#         """Plot training history for all models"""
#         if not self.histories:
#             print("No training histories available")
#             return
#
#         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#         fig.suptitle('Training History Comparison', fontsize=16)
#
#         metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
#         titles = ['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss']
#
#         for i, (metric, title) in enumerate(zip(metrics, titles)):
#             ax = axes[i // 2, i % 2]
#
#             for model_name, history in self.histories.items():
#                 if metric in history:
#                     ax.plot(history[metric], label=f'{model_name}')
#
#             ax.set_title(title)
#             ax.set_xlabel('Epochs')
#             ax.set_ylabel(metric.replace('_', ' ').title())
#             ax.legend()
#             ax.grid(True)
#
#         plt.tight_layout()
#
#         if save:
#             plt.savefig(os.path.join(self.results_dir, 'training_history.png'),
#                         dpi=300, bbox_inches='tight')
#         plt.show()
#
#     def plot_confusion_matrices(self, save=True):
#         """Plot confusion matrices for all models"""
#         if not self.results:
#             print("No results available")
#             return
#
#         n_models = len(self.results)
#         cols = min(3, n_models)
#         rows = (n_models + cols - 1) // cols
#
#         fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
#         if n_models == 1:
#             axes = [axes]
#         elif rows == 1:
#             axes = [axes]
#         else:
#             axes = axes.flatten()
#
#         for i, (model_name, results) in enumerate(self.results.items()):
#             cm = results['confusion_matrix']
#
#             # Normalize confusion matrix
#             cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#             ax = axes[i] if len(axes) > 1 else axes
#             sns.heatmap(
#                 cm_norm,
#                 annot=True,
#                 fmt='.2f',
#                 cmap='Blues',
#                 xticklabels=self.class_names[:cm.shape[1]],
#                 yticklabels=self.class_names[:cm.shape[0]],
#                 ax=ax
#             )
#             ax.set_title(f'{model_name} - Confusion Matrix')
#             ax.set_xlabel('Predicted')
#             ax.set_ylabel('True')
#
#         # Hide empty subplots
#         for j in range(i + 1, len(axes)):
#             axes[j].set_visible(False)
#
#         plt.tight_layout()
#
#         if save:
#             plt.savefig(os.path.join(self.results_dir, 'confusion_matrices.png'),
#                         dpi=300, bbox_inches='tight')
#         plt.show()
#
#     def create_model_comparison(self, save=True):
#         """Create comprehensive model comparison"""
#         if not self.results:
#             print("No results available")
#             return
#
#         # Prepare comparison data
#         comparison_data = []
#         for model_name, results in self.results.items():
#             comparison_data.append({
#                 'Model': model_name,
#                 'Test Accuracy': results['test_accuracy'],
#                 'Test Top-3 Accuracy': results['test_top3_accuracy'],
#                 'Test Loss': results['test_loss'],
#                 'Precision (Macro)': results['classification_report']['macro avg']['precision'],
#                 'Recall (Macro)': results['classification_report']['macro avg']['recall'],
#                 'F1-Score (Macro)': results['classification_report']['macro avg']['f1-score']
#             })
#
#         comparison_df = pd.DataFrame(comparison_data)
#         comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
#
#         print("\n" + "=" * 80)
#         print("MODEL COMPARISON RESULTS")
#         print("=" * 80)
#         print(comparison_df.round(4).to_string(index=False))
#
#         # Save comparison
#         if save:
#             comparison_df.to_csv(
#                 os.path.join(self.results_dir, 'model_comparison.csv'),
#                 index=False
#             )
#
#         # Visualization
#         fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#         fig.suptitle('Model Performance Comparison', fontsize=16)
#
#         # Accuracy comparison
#         ax1 = axes[0, 0]
#         bars1 = ax1.bar(comparison_df['Model'], comparison_df['Test Accuracy'])
#         ax1.set_title('Test Accuracy Comparison')
#         ax1.set_ylabel('Accuracy')
#         ax1.tick_params(axis='x', rotation=45)
#         for bar, acc in zip(bars1, comparison_df['Test Accuracy']):
#             ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                      f'{acc:.3f}', ha='center', va='bottom')
#
#         # Loss comparison
#         ax2 = axes[0, 1]
#         bars2 = ax2.bar(comparison_df['Model'], comparison_df['Test Loss'])
#         ax2.set_title('Test Loss Comparison')
#         ax2.set_ylabel('Loss')
#         ax2.tick_params(axis='x', rotation=45)
#         for bar, loss in zip(bars2, comparison_df['Test Loss']):
#             ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                      f'{loss:.3f}', ha='center', va='bottom')
#
#         # F1-Score comparison
#         ax3 = axes[1, 0]
#         bars3 = ax3.bar(comparison_df['Model'], comparison_df['F1-Score (Macro)'])
#         ax3.set_title('F1-Score Comparison (Macro Average)')
#         ax3.set_ylabel('F1-Score')
#         ax3.tick_params(axis='x', rotation=45)
#         for bar, f1 in zip(bars3, comparison_df['F1-Score (Macro)']):
#             ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                      f'{f1:.3f}', ha='center', va='bottom')
#
#         # Comprehensive metrics radar chart (for top 3 models)
#         ax4 = axes[1, 1]
#         top_3_models = comparison_df.head(3)
#
#         metrics = ['Test Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
#         angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
#         angles += angles[:1]  # Complete the circle
#
#         for _, row in top_3_models.iterrows():
#             values = [row[metric] for metric in metrics]
#             values += values[:1]  # Complete the circle
#             ax4.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
#             ax4.fill(angles, values, alpha=0.25)
#
#         ax4.set_xticks(angles[:-1])
#         ax4.set_xticklabels(metrics)
#         ax4.set_ylim(0, 1)
#         ax4.set_title('Top 3 Models - Multi-metric Comparison')
#         ax4.legend()
#         ax4.grid(True)
#
#         plt.tight_layout()
#
#         if save:
#             plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'),
#                         dpi=300, bbox_inches='tight')
#         plt.show()
#
#         return comparison_df
#
#     def save_results(self):
#         """Save all results to files"""
#         print(f"Saving results to {self.results_dir}")
#
#         # Save model summaries
#         for model_name, model in self.models.items():
#             with open(os.path.join(self.results_dir, f'{model_name}_summary.txt'), 'w') as f:
#                 model.summary(print_fn=lambda x: f.write(x + '\n'))
#
#         # Save detailed results
#         detailed_results = {}
#         for model_name, results in self.results.items():
#             detailed_results[model_name] = {
#                 'test_accuracy': float(results['test_accuracy']),
#                 'test_loss': float(results['test_loss']),
#                 'test_top3_accuracy': float(results['test_top3_accuracy']),
#                 'classification_report': results['classification_report']
#             }
#
#         with open(os.path.join(self.results_dir, 'detailed_results.json'), 'w') as f:
#             json.dump(detailed_results, f, indent=2)
#
#         # Save training histories
#         with open(os.path.join(self.results_dir, 'training_histories.json'), 'w') as f:
#             histories_serializable = {}
#             for model_name, history in self.histories.items():
#                 histories_serializable[model_name] = {
#                     key: [float(val) for val in values]
#                     for key, values in history.items()
#                 }
#             json.dump(histories_serializable, f, indent=2)
#
#         print("Results saved successfully!")
#
#
# def main():
#     """Main execution function"""
#     # Configuration
#     DATA_DIR = r"C:\Users\hp\OneDrive\Documents\THUN\archive\Fish_Dataset\Fish_Dataset"  # Update this path
#     RESULTS_DIR = "fish_classification_results"
#     IMG_SIZE = (224, 224)
#     BATCH_SIZE = 32
#     EPOCHS = 30
#     MODELS_TO_TRAIN = ['MobileNetV2', 'DenseNet121', 'EfficientNetB0', 'VGG16']
#
#     # Initialize classifier
#     classifier = FishClassifier(
#         data_dir=DATA_DIR,
#         results_dir=RESULTS_DIR,
#         img_size=IMG_SIZE,
#         batch_size=BATCH_SIZE
#     )
#
#     # Prepare data
#     classifier.prepare_data(validation_split=0.2, test_split=0.1)
#
#     # Train models
#     classifier.train_all_models(
#         models_to_train=MODELS_TO_TRAIN,
#         epochs=EPOCHS,
#         fine_tune=True
#     )
#
#     # Generate visualizations and comparisons
#     classifier.plot_training_history()
#     classifier.plot_confusion_matrices()
#     comparison_results = classifier.create_model_comparison()
#
#     # Save all results
#     classifier.save_results()
#
#     print(f"\n{'=' * 80}")
#     print("CLASSIFICATION COMPLETE!")
#     print(f"Results saved in: {RESULTS_DIR}")
#     print("Files generated:")
#     print("- Model comparison CSV and visualizations")
#     print("- Training history plots")
#     print("- Confusion matrices")
#     print("- Detailed results JSON")
#     print("- Model summaries")
#     print("- Trained model weights (.h5 files)")
#     print(f"{'=' * 80}")
#
#
# if __name__ == "__main__":
#     main()
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    MobileNetV2, DenseNet121, EfficientNetB0,
    VGG16, ResNet50V2
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Vision Transformer (requires tensorflow-addons)
try:
    import tensorflow_addons as tfa

    ViT_AVAILABLE = True
except ImportError:
    print("Warning: tensorflow-addons not available. ViT model will be skipped.")
    ViT_AVAILABLE = False


class FishClassifier:
    def __init__(self, data_dir, results_dir="classification_results", img_size=(224, 224), batch_size=32):
        """
        Initialize the Fish Classifier

        Args:
            data_dir (str): Path to the dataset directory
            results_dir (str): Directory to save results
            img_size (tuple): Input image size
            batch_size (int): Batch size for training
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.models = {}
        self.histories = {}
        self.results = {}

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

        # Setup GPU if available
        self.setup_gpu()

    def setup_gpu(self):
        """Configure GPU settings"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU available: {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("No GPU available, using CPU")

    def prepare_data(self, validation_split=0.2, test_split=0.1):
        """
        Prepare data generators with augmentation
        """
        print("Preparing data generators...")

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest',
            validation_split=validation_split + test_split
        )

        # No augmentation for validation/test
        val_test_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=validation_split / (validation_split + test_split)
        )

        # Training data
        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        # Validation data
        self.val_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

        # Test data (using a separate generator)
        self.test_generator = val_test_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        self.num_classes = self.train_generator.num_classes
        self.class_names = list(self.train_generator.class_indices.keys())

        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")

        # Compute class weights for imbalanced datasets
        self.compute_class_weights()

    def compute_class_weights(self):
        """Compute class weights to handle imbalanced datasets"""
        labels = self.train_generator.labels
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        self.class_weight = dict(enumerate(class_weights))
        print("Class weights computed for balanced training")

    def create_model(self, model_name, input_shape=(224, 224, 3)):
        """
        Create transfer learning model

        Args:
            model_name (str): Name of the model architecture
            input_shape (tuple): Input shape

        Returns:
            keras.Model: Compiled model
        """
        print(f"Creating {model_name} model...")

        if model_name == 'MobileNetV2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif model_name == 'DenseNet121':
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif model_name == 'VGG16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif model_name == 'ResNet50V2':
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif model_name == 'ViT' and ViT_AVAILABLE:
            # Vision Transformer implementation
            return self.create_vit_model(input_shape)
        else:
            raise ValueError(f"Model {model_name} not supported or not available")

        # Freeze base model initially
        base_model.trainable = False

        # Add custom classification head
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
        )

        return model

    def create_vit_model(self, input_shape):
        """Create Vision Transformer model (simplified version)"""
        if not ViT_AVAILABLE:
            raise ImportError("tensorflow-addons required for ViT model")

        # Simplified ViT implementation
        inputs = keras.Input(shape=input_shape)

        # Patch extraction and embedding
        patch_size = 16
        num_patches = (input_shape[0] // patch_size) ** 2
        projection_dim = 768

        # Extract patches
        patches = layers.Conv2D(projection_dim, patch_size, strides=patch_size)(inputs)
        patches = layers.Reshape((num_patches, projection_dim))(patches)

        # Position embedding
        positions = tf.range(start=0, limit=num_patches, delta=1)
        pos_embed = layers.Embedding(num_patches, projection_dim)(positions)
        encoded_patches = patches + pos_embed

        # Transformer blocks
        for _ in range(8):
            # Multi-head attention
            attention = layers.MultiHeadAttention(
                num_heads=8, key_dim=projection_dim // 8
            )(encoded_patches, encoded_patches)
            attention = layers.Dropout(0.1)(attention)

            # Skip connection
            x1 = layers.Add()([attention, encoded_patches])
            x1 = layers.LayerNormalization()(x1)

            # MLP
            mlp = layers.Dense(projection_dim * 2, activation='gelu')(x1)
            mlp = layers.Dropout(0.1)(mlp)
            mlp = layers.Dense(projection_dim)(mlp)

            # Skip connection
            encoded_patches = layers.Add()([mlp, x1])
            encoded_patches = layers.LayerNormalization()(encoded_patches)

        # Classification head
        x = layers.GlobalAveragePooling1D()(encoded_patches)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )

        return model

    def train_model(self, model_name, epochs=50, fine_tune=True, fine_tune_epochs=20):
        """
        Train a model with transfer learning

        Args:
            model_name (str): Model architecture name
            epochs (int): Initial training epochs
            fine_tune (bool): Whether to fine-tune the base model
            fine_tune_epochs (int): Fine-tuning epochs
        """
        print(f"\n{'=' * 60}")
        print(f"Training {model_name}")
        print(f"{'=' * 60}")

        # Create model
        model = self.create_model(model_name, (*self.img_size, 3))

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.results_dir, f'{model_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Initial training
        print(f"Phase 1: Training with frozen base model ({epochs} epochs)")
        history1 = model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            class_weight=self.class_weight,
            verbose=1
        )

        # Fine-tuning
        history2 = None
        if fine_tune and model_name != 'ViT':
            print(f"Phase 2: Fine-tuning ({fine_tune_epochs} epochs)")

            # Unfreeze base model
            base_model = model.layers[1]  # Assuming base model is the second layer
            base_model.trainable = True

            # Use lower learning rate for fine-tuning
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
            )

            # Update callbacks for fine-tuning
            callbacks[2] = ModelCheckpoint(
                filepath=os.path.join(self.results_dir, f'{model_name}_finetuned.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )

            history2 = model.fit(
                self.train_generator,
                epochs=fine_tune_epochs,
                validation_data=self.val_generator,
                callbacks=callbacks,
                class_weight=self.class_weight,
                verbose=1
            )

        # Combine histories
        if history2:
            combined_history = {}
            for key in history1.history.keys():
                combined_history[key] = history1.history[key] + history2.history[key]
        else:
            combined_history = history1.history

        # Store results
        self.models[model_name] = model
        self.histories[model_name] = combined_history

        # Evaluate on test set
        self.evaluate_model(model_name)

        print(f"Completed training {model_name}")

    def evaluate_model(self, model_name):
        """Evaluate model on test set and store results"""
        print(f"Evaluating {model_name}...")

        model = self.models[model_name]

        # Predictions
        predictions = model.predict(self.test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes

        # Metrics
        test_loss, test_accuracy, test_top3 = model.evaluate(
            self.test_generator, verbose=0
        )

        # Classification report
        class_report = classification_report(
            true_classes,
            predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)

        # Store results
        self.results[model_name] = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top3_accuracy': test_top3,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes
        }

        print(f"{model_name} - Test Accuracy: {test_accuracy:.4f}, Top-3 Accuracy: {test_top3:.4f}")

    def train_all_models(self, models_to_train=None, epochs=50, fine_tune=True):
        """
        Train multiple models

        Args:
            models_to_train (list): List of model names to train
            epochs (int): Training epochs
            fine_tune (bool): Whether to fine-tune
        """
        if models_to_train is None:
            models_to_train = ['MobileNetV2', 'DenseNet121', 'EfficientNetB0', 'VGG16', 'ResNet50V2']
            if ViT_AVAILABLE:
                models_to_train.append('ViT')

        print(f"Training models: {models_to_train}")

        for model_name in models_to_train:
            try:
                self.train_model(model_name, epochs=epochs, fine_tune=fine_tune)
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue

    def plot_training_history(self, save=True):
        """Plot training history for all models"""
        if not self.histories:
            print("No training histories available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training History Comparison', fontsize=16)

        metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
        titles = ['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]

            for model_name, history in self.histories.items():
                if metric in history:
                    ax.plot(history[metric], label=f'{model_name}')

            ax.set_title(title)
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True)

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.results_dir, 'training_history.png'),
                        dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrices(self, save=True):
        """Plot confusion matrices for all models"""
        if not self.results:
            print("No results available")
            return

        n_models = len(self.results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']

            # Normalize confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            ax = axes[i] if len(axes) > 1 else axes
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=self.class_names[:cm.shape[1]],
                yticklabels=self.class_names[:cm.shape[0]],
                ax=ax
            )
            ax.set_title(f'{model_name} - Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.results_dir, 'confusion_matrices.png'),
                        dpi=300, bbox_inches='tight')
        plt.show()

    def create_model_comparison(self, save=True):
        """Create comprehensive model comparison"""
        if not self.results:
            print("No results available")
            return

        # Prepare comparison data
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Test Accuracy': results['test_accuracy'],
                'Test Top-3 Accuracy': results['test_top3_accuracy'],
                'Test Loss': results['test_loss'],
                'Precision (Macro)': results['classification_report']['macro avg']['precision'],
                'Recall (Macro)': results['classification_report']['macro avg']['recall'],
                'F1-Score (Macro)': results['classification_report']['macro avg']['f1-score']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)

        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        print(comparison_df.round(4).to_string(index=False))

        # Save comparison
        if save:
            comparison_df.to_csv(
                os.path.join(self.results_dir, 'model_comparison.csv'),
                index=False
            )

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)

        # Accuracy comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(comparison_df['Model'], comparison_df['Test Accuracy'])
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        for bar, acc in zip(bars1, comparison_df['Test Accuracy']):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom')

        # Loss comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(comparison_df['Model'], comparison_df['Test Loss'])
        ax2.set_title('Test Loss Comparison')
        ax2.set_ylabel('Loss')
        ax2.tick_params(axis='x', rotation=45)
        for bar, loss in zip(bars2, comparison_df['Test Loss']):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{loss:.3f}', ha='center', va='bottom')

        # F1-Score comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(comparison_df['Model'], comparison_df['F1-Score (Macro)'])
        ax3.set_title('F1-Score Comparison (Macro Average)')
        ax3.set_ylabel('F1-Score')
        ax3.tick_params(axis='x', rotation=45)
        for bar, f1 in zip(bars3, comparison_df['F1-Score (Macro)']):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{f1:.3f}', ha='center', va='bottom')

        # Comprehensive metrics radar chart (for top 3 models)
        ax4 = axes[1, 1]
        top_3_models = comparison_df.head(3)

        metrics = ['Test Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for _, row in top_3_models.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            ax4.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
            ax4.fill(angles, values, alpha=0.25)

        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics)
        ax4.set_ylim(0, 1)
        ax4.set_title('Top 3 Models - Multi-metric Comparison')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'),
                        dpi=300, bbox_inches='tight')
        plt.show()

        return comparison_df

    def save_results(self):
        """Save all results to files"""
        print(f"Saving results to {self.results_dir}")

        # Save model summaries
        for model_name, model in self.models.items():
            with open(os.path.join(self.results_dir, f'{model_name}_summary.txt'), 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Save detailed results
        detailed_results = {}
        for model_name, results in self.results.items():
            model_results = {
                'test_accuracy': float(results['test_accuracy']),
                'test_loss': float(results['test_loss']),
                'classification_report': results['classification_report']
            }

            # Add top-3 accuracy if available
            if results.get('test_top3_accuracy') is not None:
                model_results['test_top3_accuracy'] = float(results['test_top3_accuracy'])

            detailed_results[model_name] = model_results

        with open(os.path.join(self.results_dir, 'detailed_results.json'), 'w') as f:
            json.dump(detailed_results, f, indent=2)

        # Save training histories
        with open(os.path.join(self.results_dir, 'training_histories.json'), 'w') as f:
            histories_serializable = {}
            for model_name, history in self.histories.items():
                histories_serializable[model_name] = {
                    key: [float(val) for val in values]
                    for key, values in history.items()
                }
            json.dump(histories_serializable, f, indent=2)

        print("Results saved successfully!")


def main():
    """Main execution function"""
    # Configuration
    DATA_DIR = r"C:\Users\hp\OneDrive\Documents\THUN\archive\Fish_Dataset\Fish_Dataset"  # Update this path
    RESULTS_DIR = "fish_classification_results"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    MODELS_TO_TRAIN = ['MobileNetV2', 'DenseNet121', 'EfficientNetB0', 'VGG16']

    # Initialize classifier
    classifier = FishClassifier(
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Prepare data
    classifier.prepare_data(validation_split=0.2, test_split=0.1)

    # Train models
    classifier.train_all_models(
        models_to_train=MODELS_TO_TRAIN,
        epochs=EPOCHS,
        fine_tune=True
    )

    # Generate visualizations and comparisons
    classifier.plot_training_history()
    classifier.plot_confusion_matrices()
    comparison_results = classifier.create_model_comparison()

    # Save all results
    classifier.save_results()

    print(f"\n{'=' * 80}")
    print("CLASSIFICATION COMPLETE!")
    print(f"Results saved in: {RESULTS_DIR}")
    print("Files generated:")
    print("- Model comparison CSV and visualizations")
    print("- Training history plots")
    print("- Confusion matrices")
    print("- Detailed results JSON")
    print("- Model summaries")
    print("- Trained model weights (.h5 files)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()