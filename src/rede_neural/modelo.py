import torch
import torch.nn as nn
import torchvision.models as models


class ModeloLibras(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 🔥 CNN (extrai features de cada frame)
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.cnn = nn.Sequential(*list(base.children())[:-1])
        # saída: 2048 features por frame

        # 🔥 LSTM (aprende sequência temporal)
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=256,
            batch_first=True
        )

        # 🔥 classificador final
        self.fc = nn.Linear(256, num_classes)


    def forward(self, x):
        """
        x: (batch, seq, C, H, W)
        """

        b, t, c, h, w = x.shape

        # junta batch + tempo
        x = x.view(b * t, c, h, w)

        # extrai features CNN
        x = self.cnn(x)
        x = x.view(b, t, -1)

        # sequência temporal
        out, _ = self.lstm(x)

        # último estado da sequência
        out = out[:, -1, :]

        # classificação
        out = self.fc(out)

        return out