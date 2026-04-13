import numpy as np
import os
import cv2
import random
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import keras
import matplotlib.pyplot as plt
from joblib import dump, load

def load_images(X, y, folder, value):
    files = os.listdir(f'data/{folder}')
    for file_name in files:
        img = cv2.imread(f'data/{folder}/{file_name}')
        X.append(img)
        y.append(value)

def get_augmented_imgs(img):
    augmented_imgs = []

    # Gaussian noise
    gauss_noise = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.randn(gauss_noise, 0, random.randint(25, 75))
    noise_img = cv2.merge([gauss_noise, gauss_noise, gauss_noise])
    augmented_imgs.append(cv2.add(img, noise_img))

    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), random.uniform(1.5, 4))
    augmented_imgs.append(blurred)

    # Brightness/Contrast
    alpha = random.uniform(1.1, 1.5)     # Contrast
    beta = random.randint(-30, 30)          # Brightness
    contrast_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    augmented_imgs.append(contrast_img)

    # Hue/Saturation
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    offset = random.randint(-15, 15)
    h_changed = ((h.astype(int) + offset) % 180).astype(np.uint8)   # type: ignore
    s_changed = s * random.uniform(0.3, 2)
    s_changed = np.clip(s_changed, 0, 255).astype(np.uint8)

    hue_sat_img = cv2.cvtColor(cv2.merge([h_changed, s_changed, v]), cv2.COLOR_HSV2BGR)
    augmented_imgs.append(hue_sat_img)

    # cv2.imshow('notaugmented', img)
    # for i in range(len(augmented_imgs)):
    #     cv2.imshow(f'augmented{i}', augmented_imgs[i])
    #
    # cv2.waitKey(0)

    return augmented_imgs

def build_model(params, epochs, X_train, y_train):

    model = keras.models.Sequential([
        keras.layers.Input(X_train[0].shape),
        keras.layers.Conv2D(params['num_filters'], (5, 5), activation='relu'),
        keras.layers.MaxPooling2D((4, 4)),
        keras.layers.Dropout(params['dropout']),
        keras.layers.Conv2D(params['num_filters'], (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(params['dropout']),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='mean_squared_error',
        metrics=['mae', 'mse']
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=params['batch_size'],
        validation_split=0.2
    )

    return model, history

def train_new_model(X_train, y_train, filename):

    # Augment the training dataset
    X_train_aug, y_train_aug = [], []
    for img, label in zip(X_train, y_train):
        for augmented in get_augmented_imgs(img):
            X_train_aug.append(augmented)
            y_train_aug.append(label)

    X_train_aug = np.concatenate((X_train, np.array(X_train_aug)))
    y_train_aug = np.concatenate((y_train, np.array(y_train_aug)))

    params = {'num_filters': 32, 'dropout': 0.2, 'learning_rate': 0.01, 'batch_size': 16}

    print(f"Hyperparameter search complete. Params found: {params}")

    # model, history = build_model(params, 10, X_train, y_train)
    model, history = build_model(params, 20, X_train_aug, y_train_aug)

    dump(model, filename=filename)
    print(f"ML Model was saved as '{filename}'.")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend(loc='upper right')
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='MAE')
    plt.plot(history.history['val_mae'], label='val_MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend(loc='upper right')
    plt.title('Mean Absolute Error')

    plt.tight_layout()
    plt.show()

def run_saved_model(X_test, y_test, filename):

    model = load(filename)

    print("Test Metrics:")

    # Print metrics
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=2)
    print(f"Loss (MSE): {test_loss}")
    print(f"MAE: {test_mae}")
    print(f"MSE: {test_mse}")

    y_pred = model.predict(X_test).flatten()

    # Calculate additional regression metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nDetailed Regression Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}")

    # Plot predictions vs actual
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Actual (R² = {r2:.3f})')

    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.axvline(x=0, color='r', linestyle='--')

    plt.tight_layout()
    plt.show()


def main():
    X = []
    y = []

    load_images(X, y, 'A', 1.0)
    load_images(X, y, 'B', 0.33)
    load_images(X, y, 'C', -0.33)
    load_images(X, y, 'D', -1.0)
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    filename = "regression_model.joblib"

    # Un-comment line below to train new model
    train_new_model(X_train, y_train, filename)
    run_saved_model(X_test, y_test, filename)

if __name__ == '__main__':
    main()