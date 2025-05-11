import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# EarlyStopping and ReduceLROnPlateau are no longer imported as they are not used
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt

# --- Configuration for Data Files ---
TRAIN_CSV_FILE = 'train_data.csv'
VALIDATION_CSV_FILE = 'validation_data.csv'
TEST_CSV_FILE = 'test_data.csv'

# --- 1. Load Pre-split Data ---
try:
    train_data = pd.read_csv(TRAIN_CSV_FILE)
    validation_data = pd.read_csv(VALIDATION_CSV_FILE)
    test_data = pd.read_csv(TEST_CSV_FILE)
    print("Successfully loaded train, validation, and test CSV files.")
except FileNotFoundError as e:
    print(f"Error: One or more CSV files not found. Please ensure '{TRAIN_CSV_FILE}', '{VALIDATION_CSV_FILE}', and '{TEST_CSV_FILE}' exist.")
    print(f"Details: {e}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV files: {e}")
    exit()

# Display basic information about the loaded data
print("\n--- Training Data Info ---")
train_data.info()
print(f"Shape: {train_data.shape}")

# --- 2. Define Features (X) and Target (y) from Loaded DataFrames ---
target_column = 'Failure_Status'
feature_columns = ['Temperature', 'Pressure', 'Vibration_Level', 'Humidity', 'Power_Consumption']

for df_name, df in [("Training Data", train_data), ("Validation Data", validation_data), ("Test Data", test_data)]:
    missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
    if missing_cols:
        print(f"\nError: The following columns are missing from '{df_name}': {missing_cols}")
        exit()

X_train_orig = train_data[feature_columns]
y_train = train_data[target_column]
X_val_orig = validation_data[feature_columns]
y_val = validation_data[target_column]
X_test_orig = test_data[feature_columns]
y_test = test_data[target_column]

print(f"\nShape of X_train_orig: {X_train_orig.shape}")
print(f"Class distribution in y_train:\n{y_train.value_counts(normalize=True)}")

class_distribution_train = y_train.value_counts(normalize=True)
if class_distribution_train.min() < 0.2: # Warning if minority class is less than 20%
    print("\nWarning: Class imbalance detected in training data. Class weights will be used.")

# --- 3. Preprocess Data: Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_orig.copy())
X_val_scaled = scaler.transform(X_val_orig.copy())
X_test_scaled = scaler.transform(X_test_orig.copy())

print("\n--- Scaled Training Data Head ---")
print(pd.DataFrame(X_train_scaled, columns=feature_columns).head())

# --- 4. Calculate Class Weights ---
classes = np.unique(y_train)
weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, weights))
print(f"\nCalculated Class Weights: {class_weights_dict}")

# --- 5. Define the Neural Network Model (Keras) - Modified Architecture ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

print("\n--- Model Summary ---")
model.summary()

# --- 6. Compile the Model ---
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

# --- 7. Train the Model ---
print("\n--- Training Model ---")
# Callbacks for EarlyStopping and ReduceLROnPlateau have been removed.
# The model will train for the full number of epochs specified.
FIXED_EPOCHS = 200 # <<<<------ CHECK AND SET YOUR DESIRED FIXED EPOCHS HERE
print(f"Training for a fixed {FIXED_EPOCHS} epochs.")

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=FIXED_EPOCHS, # Use the fixed number of epochs
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    class_weight=class_weights_dict,
    callbacks=[], # Empty list for callbacks ensures no early stopping
    verbose=1
)

# --- 8. Evaluate the Model on the Validation Set ---
# Note: Since EarlyStopping with restore_best_weights is removed,
# the model will have the weights from the final epoch of training.
print("\n--- Evaluating Model on Validation Set (Weights from Final Epoch) ---")
val_results = model.evaluate(X_val_scaled, y_val, verbose=0)
print(f"Validation Loss: {val_results[0]:.4f}")
print(f"Validation Accuracy: {val_results[1]:.4f} ({(val_results[1] * 100):.2f}%)")
if len(val_results) > 2:
    print(f"Validation Precision: {val_results[2]:.4f}")
    print(f"Validation Recall: {val_results[3]:.4f}")

# --- 9. Evaluate the Model on the Test Set ---
print("\n--- Evaluating Model on Test Set (Weights from Final Epoch) ---")
test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f} ({(test_results[1] * 100):.2f}%)")
if len(test_results) > 2:
    print(f"Test Precision: {test_results[2]:.4f}")
    print(f"Test Recall: {test_results[3]:.4f}")

# --- 10. Identify and Display Misclassified Rows from the Test Set ---
print("\n--- Identifying Misclassified Rows from Test Set ---")
y_pred_proba_test = model.predict(X_test_scaled)
y_pred_test = (y_pred_proba_test > 0.5).astype(int).flatten()
y_test_array = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test
misclassified_indices = np.where(y_pred_test != y_test_array)[0]

if len(misclassified_indices) > 0:
    print(f"\nFound {len(misclassified_indices)} misclassified rows in the test set (out of {len(y_test_array)} total test samples).")
    misclassified_data = X_test_orig.iloc[misclassified_indices].copy()
    misclassified_data['True_Failure_Status'] = y_test_array[misclassified_indices]
    misclassified_data['Predicted_Failure_Status'] = y_pred_test[misclassified_indices]
    misclassified_data['Predicted_Probability_Failure'] = y_pred_proba_test[misclassified_indices].round(6)
    print("\n--- Misclassified Rows (Original Values) ---")
    print(misclassified_data)
else:
    print("\nNo misclassified rows found in the test set. The model is 100% accurate on this test split!")

# --- 11. Optional: Plot training history ---
if all(m in history.history for m in ['accuracy', 'val_accuracy', 'loss', 'val_loss', 'precision', 'val_precision', 'recall', 'val_recall']):
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(2, 2, 3)
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(2, 2, 4)
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot (matplotlib.pyplot.show() error: {e}). Saving to file instead.")
        plt.savefig("training_history_full_presplit_fixed_epochs.png")
        print("Plot saved as training_history_full_presplit_fixed_epochs.png")
else:
    print("\nCould not plot full history: Some metrics might not be available.")
    if 'accuracy' in history.history and 'val_accuracy' in history.history and 'loss' in history.history and 'val_loss' in history.history:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.tight_layout()
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot (matplotlib.pyplot.show() error: {e}). Saving to file instead.")
            plt.savefig("training_history_acc_loss_presplit_fixed_epochs.png")
            print("Plot saved as training_history_acc_loss_presplit_fixed_epochs.png")

print("\n--- AI Model Script Finished ---")
