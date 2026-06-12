# Reconhecimento de LIBRAS com Redes Neurais

**Trabalho de Conclusão de Curso (TCC)** — Reconhecimento de sinais da Língua Brasileira de Sinais (LIBRAS) a partir de vídeos, utilizando uma arquitetura CNN-LSTM com transferência de aprendizado.

---

## Visão Geral

Este projeto implementa um pipeline completo para reconhecimento de 20 sinais de LIBRAS em vídeo:

1. **Download automático** do dataset MINDS-LIBRAS via Kaggle
2. **Pré-processamento** dos vídeos com detecção de mãos via YOLOv11
3. **Treinamento** de uma rede neural TimeDistributed ResNet50 + LSTM no Google Colab
4. **Avaliação** com métricas por classe e matriz de confusão

---

## Arquitetura do Modelo

```
Entrada: (batch, 30 frames, 128×128, 3 canais RGB)
         ↓
TimeDistributed(ResNet50 — ImageNet, pesos congelados)  →  (batch, 30, 2048)
         ↓
TimeDistributed(Dense 512, ReLU)                        →  (batch, 30, 512)
         ↓
BatchNormalization
         ↓
LSTM(256 unidades)                                      →  (batch, 256)
         ↓
Dropout(0.5)
         ↓
Dense(20, Softmax)                                      →  (batch, 20 classes)
```

| Parâmetros         | Quantidade        |
|--------------------|-------------------|
| Total              | 25.431.444        |
| Treináveis         | 1.842.708 (7 MB)  |
| Não treináveis     | 23.588.736 (90 MB)|

**Compilação:**
- Otimizador: Adam (`lr=0.0005`, `weight_decay=0.0005`)
- Loss: `CategoricalCrossentropy` com `label_smoothing=0.1`
- Métrica: Acurácia

---

## Dataset — MINDS-LIBRAS

| Item                  | Detalhe                                    |
|-----------------------|--------------------------------------------|
| Fonte                 | Kaggle: `j0aopsantos/minds-libras`         |
| Total de vídeos       | 800                                        |
| Treino / Teste        | 640 (80%) / 160 (20%)                      |
| Classes               | 20 sinais de LIBRAS                        |
| Frames por vídeo      | 30 (fixo, após pré-processamento)          |
| Resolução dos frames  | 128 × 128 pixels                           |

**Os 20 sinais reconhecidos:**

| # | Sinal       | # | Sinal     |
|---|-------------|---|-----------|
| 1 | acontecer   | 11| conhecer  |
| 2 | aluno       | 12| espelho   |
| 3 | amarelo     | 13| esquina   |
| 4 | america     | 14| filho     |
| 5 | aproveitar  | 15| maca      |
| 6 | bala        | 16| medo      |
| 7 | banco       | 17| ruim      |
| 8 | banheiro    | 18| sapo      |
| 9 | barulho     | 19| vacina    |
|10 | cinco       | 20| vontade   |

---

## Pipeline de Pré-processamento

Para cada vídeo do dataset:

1. Detecta o intervalo de movimento real (optical flow + `find_peaks`)
2. Roda YOLOv11 nano para detectar as mãos quadro a quadro
3. Seleciona a maior bounding box (mão principal)
4. Recorta a ROI com margens de 35% (horizontal) e 45% (vertical)
5. Aplica padding para manter proporção sem distorcer
6. Redimensiona para 128×128
7. Normaliza o número de frames para exatamente 30:
   - Amostra a cada 4 frames (se resultado > 8, para)
   - Se < 8 frames, amostragem uniforme em toda a sequência
   - Se ainda < 30, replica o último frame até completar

---

## Estrutura do Projeto

```
projetofinal-tcc-redesneurais/
├── main.py                                         # Entry point do pipeline
├── requirements.txt                                # Dependências
├── best.pt                                         # Pesos YOLO customizados
├── yolo11n.pt                                      # YOLOv11 nano (detecção de mãos)
├── src/
│   ├── download_dataset.py                         # Download via Kaggle API
│   ├── ajuste_dataset.py                           # Pré-processamento dos vídeos
│   └── rede_neural_resnet_lstm.ipynb               # Notebook de treino (versão final)
├── data/
│   ├── origin/                                     # Vídeos originais (.mp4)
│   └── pre_processado/
│       └── dataset_minds/                          # Frames processados (800 pastas)
└── reports/                                        # CSVs de estatísticas gerados
```

---

## Requisitos

```
python==13.13.13
tensorflow==2.20.0
opencv-python>=4.8.0
ultralytics>=8.0.0
numpy>=1.26.0
scipy>=1.11.0
pandas>=2.1.0
scikit-learn>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
tqdm>=4.66.0
kagglehub>=0.2.2
```

---

## Como Usar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o pré-processamento

```bash
python main.py
```

Isso irá:
- Baixar o dataset MINDS-LIBRAS do Kaggle para `data/origin/`
- Processar os vídeos com YOLO e salvar os frames em `data/pre_processado/dataset_minds/`

### 3. Treinar o modelo

```bash
jupyter notebook src/lstm_gemini_com_batches_melhor_de_todos.ipynb
```

Execute todas as células em ordem. O modelo treinado será salvo em `saved_model/modelo_lstm_final.keras`.

---

## Treinamento

| Parâmetro        | Valor                               |
|------------------|-------------------------------------|
| Épocas máximas   | 30                                  |
| Batch size       | 16                                  |
| EarlyStopping    | paciência 15 épocas (val_accuracy)  |
| ReduceLROnPlateau| fator 0.5, paciência 5 épocas       |
| LR mínimo        | 1e-6                                |
| Semente aleatória| 42 (split reproduzível)             |

O split treino/teste é salvo em CSVs no Drive para garantir reprodutibilidade entre sessões.

---

## Avaliação

O notebook gera automaticamente:

- Curvas de acurácia e loss (treino vs validação)
- Matriz de confusão
- Métricas globais: Acurácia, Precisão, Recall, F1-Score
- Métricas por classe para cada um dos 20 sinais
- CSV com resultados detalhados

---

## Tecnologias

| Categoria              | Tecnologia                        |
|------------------------|-----------------------------------|
| Framework DL           | TensorFlow / Keras 2.20           |
| Visão Computacional    | OpenCV, YOLOv11 (Ultralytics)     |
| Processamento          | NumPy, SciPy                      |
| Dados / Métricas       | Pandas, Scikit-learn              |
| Visualização           | Matplotlib, Seaborn               |
| Dataset                | Kaggle (kagglehub)                |
| Ambiente de Treino     | Google Colab (GPU T4/A100)        |

---

## Autora

**Evelyn Suzarte Fernandes**  
TCC — 2026.1
