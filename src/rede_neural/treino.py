import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.rede_neural.dataset import LibrasDatasetSequencia
from src.rede_neural.modelo import ModeloLibras


def treinar():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset (SEQUÊNCIA)
    dataset = LibrasDatasetSequencia(
        "data/pre_processado/yolo11_resultado_testes"
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = ModeloLibras(len(dataset.classes)).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5

    for ep in range(epochs):

        total_loss = 0
        acertos = 0

        loop = tqdm(loader, desc=f"Epoch {ep+1}")

        for x, y in loop:

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out = model(x)

            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = out.argmax(1)
            acertos += (preds == y).sum().item()

            loop.set_postfix(loss=loss.item())

        acc = acertos / len(dataset)

        print(f"\nEpoch {ep+1}")
        print(f"Loss: {total_loss:.4f}")
        print(f"Acurácia: {acc:.2f}\n")

    torch.save(model.state_dict(), "models/modelo_lstm.pth")