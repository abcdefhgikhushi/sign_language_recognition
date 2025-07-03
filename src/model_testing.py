import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg')
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import PROCESSED_DATA_DIR, SAVED_MODELS_DIR, IMG_SIZE


class ModelTester:
    def __init__(self, model_path=None):
        self.model = None
        self.label_encoder = None
        self.history = None
        self.model_path = model_path or os.path.join(SAVED_MODELS_DIR, 'best_model.h5')

    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        print(f"Loading model from {self.model_path}...")
        self.model = load_model(self.model_path)
        print("✅ Model loaded successfully!")
        return self.model

    def load_label_encoder(self):
        """Load the label encoder"""
        encoder_path = os.path.join(PROCESSED_DATA_DIR, 'label_encoder.pkl')
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found at {encoder_path}")

        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print("✅ Label encoder loaded successfully!")
        return self.label_encoder

    def load_training_history(self, history_path=None):
        """Load training history if available"""
        if history_path is None:
            history_path = os.path.join(SAVED_MODELS_DIR, 'training_history.pkl')

        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.history = pickle.load(f)
            print("✅ Training history loaded successfully!")
        else:
            print("⚠️ Training history not found. Skipping history plots.")
        return self.history

    def _load_all_data(self, x_prefix, y_prefix):
        """Load all batches into memory"""
        batch_num = 0
        X_batches = []
        y_batches = []

        while True:
            try:
                X_batch = np.load(os.path.join(PROCESSED_DATA_DIR, f'{x_prefix}_batch_{batch_num}.npy'))
                y_batch = np.load(os.path.join(PROCESSED_DATA_DIR, f'{y_prefix}_batch_{batch_num}.npy'))
                X_batches.append(X_batch)
                y_batches.append(y_batch)
                batch_num += 1
            except FileNotFoundError:
                break

        if X_batches:
            return np.concatenate(X_batches), np.concatenate(y_batches)
        else:
            return np.array([]), np.array([])

    def evaluate(self, dataset='test', save_results=True):
        """Comprehensive evaluation method"""
        print(f"Loading {dataset} data...")

        # Load test data
        X_test, y_test = self._load_all_data(f'X_{dataset}', f'y_{dataset}')

        if len(X_test) == 0:
            print(f"No {dataset} data found!")
            return None, None

        print(f"{dataset.capitalize()} data shape: {X_test.shape}")
        print(f"{dataset.capitalize()} labels shape: {y_test.shape}")

        # Evaluate model
        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        print(f"\n{dataset.capitalize()} Accuracy: {accuracy:.4f}")
        print(f"{dataset.capitalize()} Loss: {loss:.4f}")

        # Generate predictions
        print("Generating predictions...")
        y_pred_proba = self.model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Get class names if label encoder is available
        if self.label_encoder is not None:
            class_names = self.label_encoder.classes_
            target_names = class_names
        else:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y_test)))]
            target_names = class_names

        # Classification report
        print(f"\n{dataset.capitalize()} Classification Report:")
        print("=" * 60)
        report = classification_report(y_test, y_pred, target_names=target_names)
        print(report)

        # Detailed accuracy per class
        print(f"\nPer-class Accuracy:")
        print("-" * 30)
        for i, class_name in enumerate(class_names):
            class_mask = (y_test == i)
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                print(f"{class_name}: {class_acc:.4f} ({np.sum(class_mask)} samples)")

        # Confusion Matrix
        print(f"\n{dataset.capitalize()} Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'{dataset.capitalize()} Set - Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_results:
            plt.savefig(os.path.join(SAVED_MODELS_DIR, f'{dataset}_confusion_matrix.png'),
                        dpi=300, bbox_inches='tight')
        plt.close()

        # Save detailed results
        if save_results:
            results = {
                'accuracy': accuracy,
                'loss': loss,
                'classification_report': report,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'true_labels': y_test,
                'prediction_probabilities': y_pred_proba
            }

            results_path = os.path.join(SAVED_MODELS_DIR, f'{dataset}_results.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"✅ Results saved to {results_path}")

        return accuracy, loss

    def plot_training_history(self, save_plot=True):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Load history first using load_training_history()")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot learning rate (if available)
        if 'lr' in self.history.history:
            ax3.plot(self.history.history['lr'], linewidth=2, color='orange')
            ax3.set_title('Learning Rate', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Learning Rate\nHistory Not Available',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_xticks([])
            ax3.set_yticks([])

        # Plot accuracy difference (overfitting indicator)
        if 'accuracy' in self.history.history and 'val_accuracy' in self.history.history:
            acc_diff = np.array(self.history.history['accuracy']) - np.array(self.history.history['val_accuracy'])
            ax4.plot(acc_diff, linewidth=2, color='red')
            ax4.set_title('Overfitting Indicator\n(Train Acc - Val Acc)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy Difference')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Overfitting Indicator\nNot Available',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_xticks([])
            ax4.set_yticks([])

        plt.tight_layout()

        if save_plot:
            plt.savefig(os.path.join(SAVED_MODELS_DIR, 'training_history.png'),
                        dpi=300, bbox_inches='tight')
        plt.close()

        # Print training summary
        if 'accuracy' in self.history.history and 'val_accuracy' in self.history.history:
            final_train_acc = self.history.history['accuracy'][-1]
            final_val_acc = self.history.history['val_accuracy'][-1]
            print(f"\nTraining Summary:")
            print(f"Final Training Accuracy: {final_train_acc:.4f}")
            print(f"Final Validation Accuracy: {final_val_acc:.4f}")
            print(f"Accuracy Gap: {final_train_acc - final_val_acc:.4f}")

            if final_train_acc - final_val_acc > 0.1:
                print("⚠️ Warning: Large accuracy gap suggests overfitting")
            else:
                print("✅ Good: Small accuracy gap suggests good generalization")

    def predict_single_image(self, image_path):
        """Predict a single image"""
        import cv2

        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, IMG_SIZE)
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # Make prediction
        pred_proba = self.model.predict(img_batch, verbose=0)[0]
        pred_class = np.argmax(pred_proba)
        confidence = pred_proba[pred_class]

        # Get class name
        if self.label_encoder is not None:
            class_name = self.label_encoder.inverse_transform([pred_class])[0]
        else:
            class_name = f"Class_{pred_class}"

        return class_name, confidence, pred_proba

    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("=" * 60)
        print("SIGN LANGUAGE MODEL EVALUATION")
        print("=" * 60)

        # Load components
        self.load_model()
        self.load_label_encoder()
        self.load_training_history()

        # Plot training history if available
        if self.history is not None:
            print("\nPlotting training history...")
            self.plot_training_history()

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_acc, test_loss = self.evaluate('test')

        # Optional: Evaluate on validation set for comparison
        print("\nEvaluating on validation set...")
        val_acc, val_loss = self.evaluate('val')

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print("Check the saved plots and results in the models directory.")


if __name__ == "__main__":
    # Create tester instance
    tester = ModelTester()

    # Run full evaluation
    tester.run_full_evaluation()

    # Example of single image prediction (uncomment to use)
    # image_path = "path/to/your/test/image.jpg"
    # class_name, confidence, probabilities = tester.predict_single_image(image_path)
    # print(f"Predicted: {class_name} (Confidence: {confidence:.4f})")
