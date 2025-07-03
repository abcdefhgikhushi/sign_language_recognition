import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils.config import PROCESSED_DATA_DIR, SAVED_MODELS_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_SIZE


class SignLanguageModel:
    def __init__(self):
        self.model = None
        self.history = None
        self.input_shape = (*IMG_SIZE, 3)

    def _load_all_data(self, x_prefix, y_prefix):
        """Load all batches into memory - use only if you have enough RAM"""
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

    def _create_batch_generator(self, x_prefix, y_prefix, batch_size, shuffle=True, augment=False):
        """Create a proper batch generator"""
        # Get all batch files
        batch_files = []
        batch_num = 0
        while True:
            x_file = os.path.join(PROCESSED_DATA_DIR, f'{x_prefix}_batch_{batch_num}.npy')
            y_file = os.path.join(PROCESSED_DATA_DIR, f'{y_prefix}_batch_{batch_num}.npy')
            if os.path.exists(x_file) and os.path.exists(y_file):
                batch_files.append((x_file, y_file))
                batch_num += 1
            else:
                break

        # Create data augmentation if needed
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=10,  # Reduced from 15
                width_shift_range=0.05,  # Reduced from 0.1
                height_shift_range=0.05,  # Reduced from 0.1
                zoom_range=0.05,  # Reduced from 0.1
                horizontal_flip=True,
                fill_mode='nearest'
            )

        while True:
            if shuffle:
                np.random.shuffle(batch_files)

            for x_file, y_file in batch_files:
                X_batch = np.load(x_file)
                y_batch = np.load(y_file)

                if augment:
                    # Apply augmentation
                    for i in range(0, len(X_batch), batch_size):
                        batch_x = X_batch[i:i + batch_size]
                        batch_y = y_batch[i:i + batch_size]

                        if len(batch_x) > 0:
                            # Generate augmented batch
                            aug_gen = datagen.flow(batch_x, batch_y, batch_size=len(batch_x), shuffle=False)
                            yield next(aug_gen)
                else:
                    # No augmentation, just yield batches
                    for i in range(0, len(X_batch), batch_size):
                        batch_x = X_batch[i:i + batch_size]
                        batch_y = y_batch[i:i + batch_size]

                        if len(batch_x) > 0:
                            yield batch_x, batch_y

    def load_data(self):
        print("Verifying batch files exist...")
        if not os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'X_train_batch_0.npy')):
            raise FileNotFoundError("Batch files not found. Please rerun preprocessing.")
        return

    def build_model(self, num_classes):
        """Improved model architecture with better regularization"""
        model = Sequential([
            # First block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Second block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Third block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Dense layers
            Flatten(),
            Dense(256, activation='relu'),  # Reduced from 512
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model

    def train(self, num_classes, epochs):
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                os.path.join(SAVED_MODELS_DIR, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # More aggressive reduction
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Count total batches for steps calculation
        train_batch_files = len([f for f in os.listdir(PROCESSED_DATA_DIR) if f.startswith('X_train_batch_')])
        val_batch_files = len([f for f in os.listdir(PROCESSED_DATA_DIR) if f.startswith('X_val_batch_')])

        # Calculate steps per epoch
        train_steps = max(1, (train_batch_files * BATCH_SIZE) // BATCH_SIZE)
        val_steps = max(1, (val_batch_files * BATCH_SIZE) // BATCH_SIZE)

        # Create generators
        train_gen = self._create_batch_generator('X_train', 'y_train', BATCH_SIZE, shuffle=True, augment=True)
        val_gen = self._create_batch_generator('X_val', 'y_val', BATCH_SIZE, shuffle=False, augment=False)

        print(f"Training steps per epoch: {train_steps}")
        print(f"Validation steps per epoch: {val_steps}")

        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    def run_training(self, epochs=EPOCHS):
        self.load_data()

        # Load a sample to get number of classes
        y_sample = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train_batch_0.npy'))

        # Get all y_train batches to determine number of classes
        y_train_batches = []
        batch_num = 0
        while True:
            try:
                y_batch = np.load(os.path.join(PROCESSED_DATA_DIR, f'y_train_batch_{batch_num}.npy'))
                y_train_batches.append(y_batch)
                batch_num += 1
            except FileNotFoundError:
                break

        y_train = np.concatenate(y_train_batches)
        num_classes = len(np.unique(y_train))

        print(f"\nDataset info:")
        print(f"Number of classes: {num_classes}")
        print(f"Training samples: {len(y_train)}")

        print(f"\nBuilding model for {num_classes} classes...")
        self.build_model(num_classes)
        self.model.summary()

        print("\nTraining model...")
        self.train(num_classes, epochs)


if __name__ == "__main__":
    trainer = SignLanguageModel()
    trainer.run_training(epochs=20)