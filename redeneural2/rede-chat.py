import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import (
    TimeDistributed,
    LSTM,
    Dense,
    GlobalAveragePooling2D,
    Dropout
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================
# CONFIGURAÇÕES
# =========================

IMG_HEIGHT = 224
IMG_WIDTH = 224

SEQUENCE_LENGTH = 20   # quantidade de frames por vídeo
BATCH_SIZE = 4
EPOCHS = 10

DATASET_PATH = "dataset/train"

# =========================
# CARREGAR SEQUÊNCIAS
# =========================

X = []
y = []

classes = os.listdir(DATASET_PATH)

for classe in classes:

    classe_path = os.path.join(DATASET_PATH, classe)

    if not os.path.isdir(classe_path):
        continue

    for video_folder in os.listdir(classe_path):

        video_path = os.path.join(classe_path, video_folder)

        frames = []

        frame_files = sorted(os.listdir(video_path))

        # pega apenas os primeiros N frames
        frame_files = frame_files[:SEQUENCE_LENGTH]

        for frame_name in frame_files:

            frame_path = os.path.join(video_path, frame_name)

            img = cv2.imread(frame_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = preprocess_input(img)

            frames.append(img)

        # completa com frames vazios se faltar
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)))

        X.append(frames)
        y.append(classe)

X = np.array(X)
y = np.array(y)

print("Formato X:", X.shape)

# =========================
# ENCODER DAS CLASSES
# =========================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

y_encoded = to_categorical(y_encoded)

# =========================
# TREINO / TESTE
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# RESNET50
# =========================

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

base_model.trainable = False

# =========================
# MODELO CNN + LSTM
# =========================

model = Sequential()

# Extrai características frame a frame
model.add(
    TimeDistributed(
        base_model,
        input_shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3)
    )
)

# Reduz dimensões
model.add(TimeDistributed(GlobalAveragePooling2D()))

# Aprende sequência temporal
model.add(LSTM(128))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dense(len(classes), activation='softmax'))

# =========================
# COMPILAR
# =========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# TREINAR
# =========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# =========================
# SALVAR MODELO
# =========================

model.save("modelo_libras_lstm.h5")

# =========================
# TESTE
# =========================

loss, acc = model.evaluate(X_test, y_test)

print(f"\nAcurácia: {acc:.4f}")