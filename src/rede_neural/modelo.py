import torch.nn as nn
import torchvision.models as models

class ModeloLibras(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # carrega ResNet50 pré-treinada
        self.base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # substitui última camada (1000 → num_classes)
        self.base.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.base(x)  # forward padrão