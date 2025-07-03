import cv2
import numpy as np
import pickle
import os
from collections import deque, Counter
from tensorflow.keras.models import load_model
from utils.config import SAVED_MODELS_DIR, PROCESSED_DATA_DIR, IMG_SIZE


class SignLanguageTranslator:
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode

        # Load the trained model
        model_path = os.path.join(SAVED_MODELS_DIR, 'best_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = load_model(model_path)
        print("✅ Model loaded successfully!")

        # Load the label encoder
        encoder_path = os.path.join(PROCESSED_DATA_DIR, 'label_encoder.pkl')
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found at {encoder_path}")

        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print("✅ Label encoder loaded successfully!")
        print(f"Available classes: {list(self.label_encoder.classes_)}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")

        # Debug: Check model output shape
        print(f"Model output shape: {self.model.output_shape}")
        print(f"Expected input shape: {self.model.input_shape}")

        # Verify model and encoder compatibility
        if self.model.output_shape[-1] != len(self.label_encoder.classes_):
            print(
                f"⚠️ WARNING: Model outputs {self.model.output_shape[-1]} classes but encoder has {len(self.label_encoder.classes_)} classes!")

        # Prediction buffers - SIMPLIFIED for debugging
        self.prediction_history = deque(maxlen=20)  # Store recent predictions for analysis
        self.sentence = ""
        self.current_word = ""

        # Debug counters
        self.frame_count = 0
        self.prediction_counts = Counter()

    def preprocess_frame(self, frame):
        """Preprocess frame with debugging"""
        # Extract hand region - LARGER ROI for better detection
        height, width = frame.shape[:2]

        # Use a larger, fixed region for consistency
        roi_x1, roi_y1 = 100, 100
        roi_x2, roi_y2 = min(width - 100, roi_x1 + 400), min(height - 100, roi_y1 + 400)

        hand_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        if self.debug_mode and self.frame_count % 30 == 0:  # Debug every 30 frames
            print(f"Original ROI shape: {hand_roi.shape}")
            print(f"ROI coordinates: ({roi_x1}, {roi_y1}) to ({roi_x2}, {roi_y2})")

        # Convert BGR to RGB
        hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)

        # Resize to training dimensions
        hand_roi_resized = cv2.resize(hand_roi_rgb, IMG_SIZE)

        if self.debug_mode and self.frame_count % 30 == 0:
            print(f"Resized shape: {hand_roi_resized.shape}")
            print(f"Target IMG_SIZE: {IMG_SIZE}")
            print(f"Pixel range after resize: {hand_roi_resized.min()} to {hand_roi_resized.max()}")

        # Normalize to [0, 1] range
        hand_roi_normalized = hand_roi_resized.astype('float32') / 255.0

        if self.debug_mode and self.frame_count % 30 == 0:
            print(f"Normalized pixel range: {hand_roi_normalized.min():.3f} to {hand_roi_normalized.max():.3f}")

        # Add batch dimension
        hand_roi_batch = np.expand_dims(hand_roi_normalized, axis=0)

        return hand_roi_batch, (roi_x1, roi_y1, roi_x2, roi_y2)

    def analyze_prediction_distribution(self, predictions, confidences):
        """Analyze prediction patterns to identify issues"""
        if len(predictions) < 10:
            return

        print("\n" + "=" * 50)
        print("PREDICTION ANALYSIS")
        print("=" * 50)

        # Count predictions
        pred_counter = Counter(predictions)
        print("Most common predictions:")
        for pred, count in pred_counter.most_common(5):
            percentage = (count / len(predictions)) * 100
            avg_conf = np.mean([conf for p, conf in zip(predictions, confidences) if p == pred])
            print(f"  {pred}: {count} times ({percentage:.1f}%) - Avg confidence: {avg_conf:.3f}")

        # Check if model is stuck on few classes
        unique_preds = len(pred_counter)
        total_classes = len(self.label_encoder.classes_)
        print(f"\nPrediction diversity: {unique_preds}/{total_classes} classes seen")

        if unique_preds < total_classes * 0.3:  # Less than 30% of classes seen
            print("⚠️ WARNING: Model is predicting very few classes - possible overfitting!")

        # Check confidence distribution
        avg_confidence = np.mean(confidences)
        print(f"Average confidence: {avg_confidence:.3f}")

        if avg_confidence > 0.95:
            print("⚠️ WARNING: Very high confidence - model might be overconfident/overfitted!")
        elif avg_confidence < 0.5:
            print("⚠️ WARNING: Low confidence - model might need more training!")

    def translate_frame(self, frame):
        """Process frame with extensive debugging"""
        self.frame_count += 1

        # Preprocess frame
        processed_frame, roi_coords = self.preprocess_frame(frame)

        # Make prediction
        try:
            pred_proba = self.model.predict(processed_frame, verbose=0)[0]
            pred_class_idx = np.argmax(pred_proba)
            confidence = pred_proba[pred_class_idx]

            # Get class name
            pred_class = self.label_encoder.inverse_transform([pred_class_idx])[0]

            # Store for analysis
            self.prediction_history.append(pred_class)
            self.prediction_counts[pred_class] += 1

            # Debug output every 30 frames
            if self.debug_mode and self.frame_count % 30 == 0:
                print(f"\nFrame {self.frame_count}:")
                print(f"Raw prediction probabilities (top 5):")
                top_5_indices = np.argsort(pred_proba)[-5:][::-1]
                for idx in top_5_indices:
                    class_name = self.label_encoder.inverse_transform([idx])[0]
                    prob = pred_proba[idx]
                    print(f"  {class_name}: {prob:.4f}")

                print(f"Predicted: {pred_class} (confidence: {confidence:.4f})")

                # Show prediction distribution every 150 frames
                if self.frame_count % 150 == 0:
                    recent_preds = list(self.prediction_history)
                    recent_confs = [np.max(self.model.predict(processed_frame, verbose=0)[0])
                                    for _ in range(min(10, len(recent_preds)))]  # Sample recent confidences
                    self.analyze_prediction_distribution(recent_preds, recent_confs)

        except Exception as e:
            print(f"Prediction error: {e}")
            pred_class = "ERROR"
            confidence = 0.0

        # SIMPLIFIED sentence building - just add letters with high confidence
        if confidence > 0.8 and pred_class not in ['NOTHING', 'BLANK', 'SPACE', 'DELETE']:
            # Add to current word only if different from last prediction
            if len(self.current_word) == 0 or self.current_word[-1] != pred_class:
                if self.frame_count % 15 == 0:  # Add letter every 15 frames to slow down
                    self.current_word += pred_class
                    print(f"Added '{pred_class}' to word: '{self.current_word}'")

        # Handle special commands
        elif pred_class == 'SPACE' and confidence > 0.8:
            if self.current_word:
                self.sentence += self.current_word + " "
                self.current_word = ""
                print(f"Added word to sentence: '{self.sentence.strip()}'")

        elif pred_class == 'DELETE' and confidence > 0.8:
            if self.current_word:
                self.current_word = self.current_word[:-1]
                print(f"Deleted character: '{self.current_word}'")

        # Visual annotations
        x1, y1, x2, y2 = roi_coords

        # Draw ROI rectangle - GREEN for high confidence, RED for low
        color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Display information with better formatting
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Current prediction with confidence
        text = f"Prediction: {pred_class} ({confidence:.3f})"
        cv2.putText(frame, text, (10, y_offset), font, 0.7, (255, 255, 0), 2)
        y_offset += 35

        # Current word
        cv2.putText(frame, f"Current Word: {self.current_word}", (10, y_offset), font, 0.8, (0, 255, 255), 2)
        y_offset += 35

        # Sentence
        sentence_display = self.sentence + self.current_word
        if len(sentence_display) > 40:
            sentence_display = "..." + sentence_display[-37:]
        cv2.putText(frame, f"Sentence: {sentence_display}", (10, y_offset), font, 0.8, (0, 255, 0), 2)
        y_offset += 35

        # Frame counter and most common predictions
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, y_offset), font, 0.6, (255, 255, 255), 1)
        y_offset += 25

        # Show top 3 most common predictions
        if len(self.prediction_counts) > 0:
            top_3 = self.prediction_counts.most_common(3)
            top_3_text = " | ".join([f"{pred}:{count}" for pred, count in top_3])
            cv2.putText(frame, f"Most frequent: {top_3_text}", (10, y_offset), font, 0.5, (200, 200, 200), 1)

        # Instructions
        cv2.putText(frame, "Controls: 'q'=quit, 'c'=clear, 's'=add space, 'd'=delete",
                    (10, frame.shape[0] - 40), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Position hand in the rectangle for detection",
                    (10, frame.shape[0] - 20), font, 0.5, (255, 255, 255), 1)

        return frame

    def clear_all(self):
        """Clear everything"""
        self.sentence = ""
        self.current_word = ""
        self.prediction_history.clear()
        self.prediction_counts.clear()
        print("Cleared all data")

    def save_debug_info(self):
        """Save debug information to file"""
        debug_info = {
            'prediction_counts': dict(self.prediction_counts),
            'recent_predictions': list(self.prediction_history),
            'total_frames': self.frame_count,
            'available_classes': list(self.label_encoder.classes_)
        }

        debug_path = os.path.join(SAVED_MODELS_DIR, 'debug_info.pkl')
        with open(debug_path, 'wb') as f:
            pickle.dump(debug_info, f)
        print(f"Debug info saved to {debug_path}")


def main():
    try:
        translator = SignLanguageTranslator(debug_mode=True)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("\n" + "=" * 60)
        print("SIGN LANGUAGE DETECTION - DEBUG MODE")
        print("=" * 60)
        print("Camera opened successfully!")
        print("Controls:")
        print("- 'q': Quit and save debug info")
        print("- 'c': Clear sentence")
        print("- 's': Add space manually")
        print("- 'd': Delete last character")
        print("- Position your hand in the green/red rectangle")
        print("- Check console for detailed prediction analysis")
        print("=" * 60)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)

            # Process frame
            output_frame = translator.translate_frame(frame)

            # Display
            cv2.imshow("Sign Language Detection - Debug Mode", output_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                translator.save_debug_info()
                break
            elif key == ord('c'):
                translator.clear_all()
            elif key == ord('s'):
                if translator.current_word:
                    translator.sentence += translator.current_word + " "
                    translator.current_word = ""
                    print(f"Manually added space. Sentence: '{translator.sentence.strip()}'")
            elif key == ord('d'):
                if translator.current_word:
                    translator.current_word = translator.current_word[:-1]
                    print(f"Manually deleted character: '{translator.current_word}'")

        cap.release()
        cv2.destroyAllWindows()

        # Final analysis
        print("\n" + "=" * 60)
        print("FINAL ANALYSIS")
        print("=" * 60)

        if translator.prediction_counts:
            print("Most common predictions during session:")
            for pred, count in translator.prediction_counts.most_common(10):
                percentage = (count / translator.frame_count) * 100
                print(f"  {pred}: {count} times ({percentage:.1f}%)")

        final_sentence = translator.sentence + translator.current_word
        if final_sentence.strip():
            print(f"\nFinal sentence: '{final_sentence.strip()}'")
        else:
            print("\nNo sentence was formed.")

        print(f"Total frames processed: {translator.frame_count}")
        print("Debug info saved for analysis.")

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()