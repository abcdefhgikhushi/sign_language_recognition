import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from utils.config import TRAIN_IMAGES_DIR, PROCESSED_DATA_DIR, CLASSES, IMG_SIZE, MAX_RAM_USAGE, SAFE_MEMORY_RATIO, \
    BATCH_SIZE


class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def preprocess_pipeline(self):
        """Memory-safe preprocessing with label-image alignment in batches"""

        # 1. Collect all image paths and labels
        print("Collecting image paths and labels...")
        image_paths, labels = [], []

        for class_name in CLASSES:
            class_dir = os.path.join(TRAIN_IMAGES_DIR, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} not found!")
                continue

            class_images = 0
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(class_name)
                    class_images += 1

            print(f"Found {class_images} images for class '{class_name}'")

        print(f"Total images found: {len(image_paths)}")

        if len(image_paths) == 0:
            raise ValueError("No images found! Check your TRAIN_IMAGES_DIR and CLASSES configuration.")

        # 2. Encode labels
        print("Encoding labels...")
        labels_encoded = self.label_encoder.fit_transform(labels)

        # Print class distribution
        unique_labels, counts = np.unique(labels_encoded, return_counts=True)
        print("\nClass distribution:")
        for i, (label_id, count) in enumerate(zip(unique_labels, counts)):
            class_name = self.label_encoder.inverse_transform([label_id])[0]
            print(f"  {class_name} (ID: {label_id}): {count} images")

        # 3. Shuffle together
        print("Shuffling data...")
        data = list(zip(image_paths, labels_encoded))
        np.random.seed(42)
        np.random.shuffle(data)
        image_paths, labels_encoded = zip(*data)

        # 4. Split data (70% train, 15% val, 15% test)
        print("Splitting data...")
        paths_train, paths_temp, y_train, y_temp = train_test_split(
            image_paths, labels_encoded,
            test_size=0.3,  # 30% for val+test
            stratify=labels_encoded,
            random_state=42
        )

        paths_val, paths_test, y_val, y_test = train_test_split(
            paths_temp, y_temp,
            test_size=0.5,  # Split the 30% equally: 15% val, 15% test
            stratify=y_temp,
            random_state=42
        )

        print(f"Train set: {len(paths_train)} images")
        print(f"Validation set: {len(paths_val)} images")
        print(f"Test set: {len(paths_test)} images")

        # 5. Save batches safely
        print("Saving training batches...")
        self._save_batches(paths_train, y_train, 'X_train', 'y_train')

        print("Saving validation batches...")
        self._save_batches(paths_val, y_val, 'X_val', 'y_val')

        print("Saving test batches...")
        self._save_batches(paths_test, y_test, 'X_test', 'y_test')

        # 6. Save label encoder
        print("Saving label encoder...")
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        with open(os.path.join(PROCESSED_DATA_DIR, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print("✅ Preprocessing complete. Batches and labels saved.")
        self._verify_batches()

    def _save_batches(self, paths, labels, x_prefix, y_prefix):
        """Save image batches and corresponding labels"""
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        total_batches = (len(paths) + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division

        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(paths))
            current_batch_size = end_idx - start_idx

            # Initialize batch arrays
            X_batch = np.zeros((current_batch_size, *IMG_SIZE, 3), dtype='float32')
            y_batch = np.array(labels[start_idx:end_idx])

            # Load and process images
            successful_loads = 0
            for j, path in enumerate(paths[start_idx:end_idx]):
                try:
                    # Read and process image
                    img = cv2.imread(path)
                    if img is None:
                        raise ValueError(f"Could not read image: {path}")

                    # Convert color space and resize
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img, IMG_SIZE)

                    # Normalize to [0, 1]
                    X_batch[j] = img_resized.astype('float32') / 255.0
                    successful_loads += 1

                except Exception as e:
                    print(f"Warning: Failed to load {path}: {str(e)}")
                    # Fill with zeros for failed images
                    X_batch[j] = np.zeros((*IMG_SIZE, 3), dtype='float32')

            # Save batch
            np.save(os.path.join(PROCESSED_DATA_DIR, f'{x_prefix}_batch_{batch_idx}.npy'), X_batch)
            np.save(os.path.join(PROCESSED_DATA_DIR, f'{y_prefix}_batch_{batch_idx}.npy'), y_batch)

            print(
                f"  Saved batch {batch_idx + 1}/{total_batches} - {successful_loads}/{current_batch_size} images loaded successfully")

    def _verify_batches(self):
        """Verify that all batches were saved correctly"""
        print("\nVerifying saved batches...")

        for split in ['train', 'val', 'test']:
            batch_count = 0
            total_samples = 0

            while True:
                x_file = os.path.join(PROCESSED_DATA_DIR, f'X_{split}_batch_{batch_count}.npy')
                y_file = os.path.join(PROCESSED_DATA_DIR, f'y_{split}_batch_{batch_count}.npy')

                if os.path.exists(x_file) and os.path.exists(y_file):
                    X_batch = np.load(x_file)
                    y_batch = np.load(y_file)
                    total_samples += len(X_batch)

                    # Verify batch integrity
                    assert len(X_batch) == len(y_batch), f"Mismatch in batch {batch_count} for {split}"
                    assert X_batch.shape[1:] == (*IMG_SIZE, 3), f"Wrong image shape in {split} batch {batch_count}"

                    batch_count += 1
                else:
                    break

            print(f"  {split.capitalize()}: {batch_count} batches, {total_samples} total samples")

    def verify_data_distribution(self):
        """Check class distribution across all splits"""
        print("\n" + "=" * 50)
        print("DATA DISTRIBUTION VERIFICATION")
        print("=" * 50)

        for split in ['train', 'val', 'test']:
            print(f"\n{split.upper()} SET:")

            y_batches = []
            batch_num = 0

            while True:
                try:
                    y_batch = np.load(os.path.join(PROCESSED_DATA_DIR, f'y_{split}_batch_{batch_num}.npy'))
                    y_batches.append(y_batch)
                    batch_num += 1
                except FileNotFoundError:
                    break

            if y_batches:
                y_all = np.concatenate(y_batches)
                unique, counts = np.unique(y_all, return_counts=True)

                print(f"  Total samples: {len(y_all)}")
                print("  Class distribution:")
                for class_id, count in zip(unique, counts):
                    class_name = self.label_encoder.inverse_transform([class_id])[0]
                    percentage = (count / len(y_all)) * 100
                    print(f"    {class_name} (ID: {class_id}): {count} samples ({percentage:.1f}%)")
            else:
                print(f"  No {split} data found!")


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_pipeline()

    # Optional: Verify data distribution
    preprocessor.verify_data_distribution()