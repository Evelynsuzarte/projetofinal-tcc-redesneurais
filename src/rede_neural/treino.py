import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.rede_neural.dataset import LibrasDataset
from src.rede_neural.modelo import ModeloLibras

def treinar():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path = "data/pre_processado/yolo11_resultado_testes"

    dataset = LibrasDataset(base_path)        # carrega dataset
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ModeloLibras(len(dataset.classes)).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()     # erro
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoca in range(5):

        total_loss = 0
        acertos = 0

        for imgs, labels in tqdm(loader, desc=f"Epoca {epoca+1}"):

            imgs, labels = imgs.to(device), labels.to(device)

            opt.zero_grad()                  # zera gradiente

            out = model(imgs)                # forward
            loss = loss_fn(out, labels)      # calcula erro

            loss.backward()                  # backprop
            opt.step()                      # atualiza pesos

            total_loss += loss.item()

            preds = out.argmax(1)           # classe prevista
            acertos += (preds == labels).sum().item()

        acc = acertos / len(dataset)

        print(f"Epoca {epoca+1} | Loss: {total_loss:.2f} | Acc: {acc:.2f}")

    os.makedirs("models", exist_ok=True) 
    torch.save(model.state_dict(), "models/modelo.pth")  # salva modelo
    print("Modelo salvo!")