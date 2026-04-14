import numpy as np
import os
import cv2
import random
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import dump, load

def load_images(X, y, folder):
    files = os.listdir(f'data/{folder}')
    for file_name in files:
        img = cv2.imread(f'data/{folder}/{file_name}', flags=0)
        img = np.expand_dims(img, axis=2)
        X.append(cv2.resize(img, (64, 64)))
        y.append(folder)

def get_augmented_imgs(img):
    augmented_imgs = []

    # Gaussian noise
    gauss_noise = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.randn(gauss_noise, 0, random.randint(25, 75))
    noise_img = np.expand_dims(gauss_noise, axis=2)
    # noise_img = cv2.merge([gauss_noise, gauss_noise, gauss_noise])
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
    # h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    # offset = random.randint(-15, 15)
    # h_changed = ((h.astype(int) + offset) % 180).astype(np.uint8)   # type: ignore
    # s_changed = s * random.uniform(0.3, 2)
    # s_changed = np.clip(s_changed, 0, 255).astype(np.uint8)
    #
    # hue_sat_img = cv2.cvtColor(cv2.merge([h_changed, s_changed, v]), cv2.COLOR_HSV2BGR)
    # augmented_imgs.append(hue_sat_img)

    # Flip each augmented
    tmp = []
    for img in augmented_imgs:
        tmp.append(cv2.flip(img, 1))

    augmented_imgs += tmp

    # Flip
    augmented_imgs.append(cv2.flip(img, 1))

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
        keras.layers.Dense(3, activation='softmax')
    ])

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
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

    tmp = []
    for img in X_train_aug:
        tmp.append(np.expand_dims(img, axis=2))

    X_train_aug = np.array(tmp)

    params = {'num_filters': 64, 'dropout': 0.4, 'learning_rate': 0.001, 'batch_size': 16}

    print(f"Hyperparameter search complete. Params found: {params}")

    # model, history = build_model(params, 10, X_train, y_train)
    model, history = build_model(params, 40, X_train_aug, y_train_aug)

    dump(model, filename=filename)
    print(f"ML Model was saved as '{filename}'.")

    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    return model

def run_saved_model(X_test, y_test, label_encoder, filename):

    model = load(filename)

    print("Test Metrics:")

    # Print metrics
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Loss: {test_loss} Accuracy: {test_acc}")

    y_pred = model.predict(X_test).argmax(axis=-1)

    report = classification_report(y_test, y_pred, output_dict=True)

    classes = [0, 1, 2]
    # classes = [0, 1, 2, 3]
    class_names = label_encoder.inverse_transform(classes)

    for _class in classes:

        class_str = str(_class)

        print(f"Class '{class_names[_class]}'")
        print(f"Precision: {report[class_str]['precision']}")
        print(f"Recall: {report[class_str]['recall']}")
        print(f"F1 Score: {report[class_str]['f1-score']}\n")

    # Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['A', 'B', 'C', 'D'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['A', 'B', 'C'])
    cm_display.plot()
    plt.show()


def main():
    X = []
    y = []

    load_images(X, y, 'A')
    load_images(X, y, 'B')
    load_images(X, y, 'C')
    # load_images(X, y, 'D')
    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=.2, random_state=11)

    filename = "classification_model.joblib"

    # Un-comment line below to train new model
    model = train_new_model(X_train, y_train, filename)
    run_saved_model(X_test, y_test, label_encoder, filename)

    # Convert the model to TensorFlow Lite format
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()

if __name__ == '__main__':
    main()