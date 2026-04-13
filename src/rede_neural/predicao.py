import torch
import os
import cv2
from src.rede_neural.modelo import ModeloLibras

# pega lista de classes
def carregar_classes(base_path):
    return sorted([
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ])

# carrega frames de uma pasta
def carregar_frames(pasta):

    frames = []

    for arq in sorted(os.listdir(pasta)):

        img = cv2.imread(os.path.join(pasta, arq))

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = img.permute(2, 0, 1)

        frames.append(img)

    return frames


def prever(pasta_frames, model, classes):

    frames = carregar_frames(pasta_frames)

    if len(frames) == 0:
        return "erro"

    outputs = []

    with torch.no_grad():
        for f in frames:
            out = model(f.unsqueeze(0))
            outputs.append(out)

    outputs = torch.stack(outputs).mean(dim=0)

    pred = outputs.argmax(1).item()

    return classes[pred]


def executar_previsao():

    base_path = "data/pre_processado/yolo11_resultado_testes"

    classes = carregar_classes(base_path)

    model = ModeloLibras(len(classes))
    model.load_state_dict(torch.load("models/modelo.pth", map_location="cpu"))
    model.eval()

    acertos = 0
    total = 0

    for classe in sorted(os.listdir(base_path)):

        pasta = os.path.join(base_path, classe)

        if not os.path.isdir(pasta):
            continue

        pred = prever(pasta, model, classes)

        print(f"Real: {classe} | Previsto: {pred}")

        if pred == classe:
            acertos += 1

        total += 1

    if total > 0:
        print(f"\nAcurácia: {acertos}/{total} = {acertos/total:.2f}")