import torch
import cv2
import os
from src.rede_neural.modelo import ModeloLibras


# 🔥 carrega frames da pasta (sequência)
def carregar_frames(pasta, max_frames=20):

    frames = []

    arquivos = sorted(os.listdir(pasta))

    for arq in arquivos[:max_frames]:

        img = cv2.imread(os.path.join(pasta, arq))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = img.permute(2, 0, 1)

        frames.append(img)

    # padding
    while len(frames) < max_frames:
        frames.append(torch.zeros(3, 224, 224))

    return torch.stack(frames)


# 🔥 previsão de UMA pasta
def prever(pasta, model, classes):

    frames = carregar_frames(pasta).unsqueeze(0)

    with torch.no_grad():
        out = model(frames)

    pred = out.argmax(1).item()

    return classes[pred]


# 🔥 função principal (AGORA EXISTE)
def executar_previsao(base_path, classes):

    model = ModeloLibras(len(classes))
    model.load_state_dict(torch.load("models/modelo_lstm.pth"))
    model.eval()

    acertos = 0
    total = 0

    print("\n===== PREVISÃO =====\n")

    # percorre TODAS as pastas (cada vídeo)
    for pasta in sorted(os.listdir(base_path)):

        caminho = os.path.join(base_path, pasta)

        if not os.path.isdir(caminho):
            continue

        # classe real vem do nome da pasta (acontecer_1 → acontecer)
        real = pasta.split("_")[0]

        pred = prever(caminho, model, classes)

        print(f"Real: {real} | Previsto: {pred}")

        if real == pred:
            acertos += 1

        total += 1

    print("\n===== RESULTADO FINAL =====")
    print(f"Acurácia: {acertos}/{total} = {acertos/total:.2f}")