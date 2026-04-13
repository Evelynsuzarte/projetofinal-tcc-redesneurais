import os
import cv2
import torch
from torch.utils.data import Dataset

class LibrasDataset(Dataset):
    def __init__(self, base_path):

        self.imagens = []   # caminhos das imagens
        self.labels = []    # labels numéricos

        # pega nomes das classes (pastas)
        self.classes = sorted([
            d for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        ])

        # mapeia classe → número
        self.mapa = {c: i for i, c in enumerate(self.classes)}

        # percorre todas as imagens
        for c in self.classes:
            pasta = os.path.join(base_path, c)

            for arq in sorted(os.listdir(pasta)):
                self.imagens.append(os.path.join(pasta, arq))
                self.labels.append(self.mapa[c])

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, idx):

        img = cv2.imread(self.imagens[idx])              # lê imagem
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # BGR → RGB
        img = cv2.resize(img, (224, 224))                # padrão ResNet

        img = torch.tensor(img, dtype=torch.float32) / 255.0  # normaliza
        img = img.permute(2, 0, 1)                       # (H,W,C) → (C,H,W)

        label = torch.tensor(self.labels[idx])           # label da imagem

        return img, label