import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall  # Critical fix
import matplotlib.pyplot as plt

# Rest of your original code remains the same...

# --- Configuration ---
TRAIN_CSV_FILE = 'train_data.csv'
VALIDATION_CSV_FILE = 'validation_data.csv'
TEST_CSV_FILE = 'test_data.csv'

# --- Load Data ---
try:
    train_data = pd.read_csv(TRAIN_CSV_FILE)
    validation_data = pd.read_csv(VALIDATION_CSV_FILE)
    test_data = pd.read_csv(TEST_CSV_FILE)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

# --- Data Preparation ---
target_column = 'Failure_Status'
feature_columns = ['Temperature', 'Pressure', 'Vibration_Level', 'Humidity', 'Power_Consumption']

# Extract features and targets
X_train = train_data[feature_columns]
y_train = train_data[target_column]
X_val = validation_data[feature_columns]
y_val = validation_data[target_column]
X_test = test_data[feature_columns]
y_test = test_data[target_column]

# --- Handle Class Imbalance with SMOTE ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- Enhanced Model Architecture ---
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 Precision(name='precision'),
                 Recall(name='recall')]
    )
    return model

model = create_model()

# --- Training Configuration ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# --- Training ---
history = model.fit(
    X_train_scaled,
    y_train_res,
    epochs=100,
    batch_size=64,
    validation_data=(X_val_scaled, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# --- Evaluation ---
def evaluate_model(model, X, y, dataset_name):
    results = model.evaluate(X, y, verbose=0)
    y_pred = (model.predict(X) > 0.5).astype(int)
    
    print(f"\n--- {dataset_name} Evaluation ---")
    print(f"Loss: {results[0]:.4f} | Accuracy: {results[1]:.4f}")
    print(f"Precision: {results[2]:.4f} | Recall: {results[3]:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

evaluate_model(model, X_val_scaled, y_val, "Validation")
evaluate_model(model, X_test_scaled, y_test, "Test")

# --- Feature Importance Analysis ---
def check_feature_importance(X, y):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=feature_columns)
    importance.sort_values().plot.barh(title='Feature Importance')
    plt.show()

check_feature_importance(X_train_scaled, y_train_res)

# --- Plot Training History ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training History - Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training History - Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()

# --- Hyperparameter Tuning (Optional) ---
# Uncomment to use Keras Tuner for systematic parameter search
import keras_tuner as kt

def build_tunable_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units1', 32, 256, step=32),
                   activation='relu',
                   input_shape=(X_train_scaled.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5)))
    
    model.add(Dense(units=hp.Int('units2', 16, 128, step=32), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout2', 0.1, 0.5)))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('lr', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    build_tunable_model,
    objective='val_accuracy',
    max_trials=15,
    executions_per_trial=2,
    directory='tuning',
    project_name='failure_prediction'
)

tuner.search(X_train_scaled, y_train_res,
             epochs=50,
             validation_data=(X_val_scaled, y_val),
             callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

# After completing the hyperparameter tuning (uncomment this section)

# 1. Get the best model from the tuner
best_model = tuner.get_best_models(num_models=1)[0]

# 2. Evaluate on test set
print("\n--- Final Evaluation with Best Model ---")
test_loss, test_acc = best_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f} ({(test_acc * 100):.2f}%)")

# 3. Generate detailed classification report
y_pred = best_model.predict(X_test_scaled)
y_pred_classes = (y_pred > 0.5).astype(int)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_classes))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred_classes))

# 4. Optional: Save the best model for later use
best_model.save('best_failure_prediction_model.keras')
print("\nBest model saved as 'best_failure_prediction_model.keras'")