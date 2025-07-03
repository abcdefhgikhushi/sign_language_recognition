import cv2
import os
import numpy as np
from utils.config import RAW_IMAGES_DIR, CLASSES, CAMERA_WIDTH, CAMERA_HEIGHT


class DataCollector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, CAMERA_WIDTH)
        self.cap.set(4, CAMERA_HEIGHT)

    def collect_data(self, class_name, num_images=500):
        """Collect images for a specific sign language letter"""
        class_dir = os.path.join(RAW_IMAGES_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        print(f"Collecting data for class: {class_name}")
        print("Press 's' to start capturing, 'q' to quit")

        count = 0
        capturing = False

        while count < num_images:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Draw rectangle for hand region
            cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

            # Display instructions
            cv2.putText(frame, f"Class: {class_name}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Images: {count}/{num_images}", (50, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if capturing:
                # Extract hand region
                hand_region = frame[100:400, 100:400]

                # Save image
                filename = os.path.join(class_dir, f"{class_name}_{count}.jpg")
                cv2.imwrite(filename, hand_region)
                count += 1

                # Small delay between captures
                cv2.waitKey(100)

            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(30) & 0xFF  # Reduced delay for responsiveness

            if key == ord('s') and not capturing:
                capturing = True
                print("Started capturing...")
            elif key == ord('q'):
                break

        print(f"Collected {count} images for class {class_name}")

    def collect_all_data(self):
        """Collect data for all classes"""
        for class_name in CLASSES[:26]:  # Only alphabets for now
            self.collect_data(class_name)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    collector = DataCollector()

    # Collect data for specific letter
    letter = input("Enter letter to collect data for (A-Z): ").upper()
    if letter in CLASSES:
        collector.collect_data(letter)
    else:
        print("Invalid letter!")