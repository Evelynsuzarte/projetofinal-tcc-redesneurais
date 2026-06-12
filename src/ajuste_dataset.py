import os  
import cv2 
import csv  
import numpy as np  
from ultralytics import YOLO  
from tqdm import tqdm  
from datetime import datetime  
from scipy.signal import find_peaks

model = YOLO("yolo11n.pt") 

INPUT = "data/origin"  
OUTPUT = "data/pre_processado/dataset_minds" 


# organiza o nome dos arquivos para poder juntar em um só
def extrair_palavra(nome):
    nome = nome.replace(".mp4", "")                                 # remove a extensão do arquivo
    while len(nome) > 0 and nome[0].isdigit():                      # enquanto o primeiro char for número
        nome = nome[1:]                                             # remove o dígito inicial

    if "Sinalizador" in nome:                                       # se o nome tiver Sinalizador
        nome = nome.split("Sinalizador")[0]                         # mantém só a parte antes dessa palavra

    return nome.lower()                                             # retorna o nome em minusculo


#detectar trecho com movimento no vídeo
def detectar_intervalo_movimento(video_path):
    cap = cv2.VideoCapture(video_path)                              # abre o vídeo para leitura
    movimentos = []                                                 # lista com a magnitude de movimento por frame
    frames_gray = []                                                # lista de frames em escala de cinza

    while True:
        ret, frame = cap.read()                                     # lê o próximo frame
        
        if not ret:                                                 # se não há mais frames, encerra o loop
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)              # converte para escala de cinza
        gray = cv2.GaussianBlur(gray, (5, 5), 0)                    # suaviza para reduzir ruído
        frames_gray.append(gray)                                    # guarda o frame suavizado
        
        if len(frames_gray) > 1:                                    # a partir do segundo frame...
            diff = cv2.absdiff(frames_gray[-2], frames_gray[-1])    # diferença absoluta entre frames consecutivos
            movimentos.append(diff.mean())                          # média da diferença como medida de movimento
        else:
            movimentos.append(0)                                    # primeiro frame não tem frame anterior, movimento = 0

    cap.release()                                                   # libera o recurso de vídeo

    if len(movimentos) == 0:                                        # vídeo vazio ou ilegível
        return 0, 0

    movimentos = np.array(movimentos, dtype=np.float32)                         # converte para array numpy
    movimentos = np.convolve(movimentos, np.ones(15) / 15, mode='same')         # suaviza o sinal com média móvel de 15 frames
    limiar = np.percentile(movimentos, 60) + np.std(movimentos) * 0.8           # limiar adaptativo: percentil 60 + desvio padrão ponderado
    picos, _ = find_peaks(movimentos, height=limiar, distance=10)               # acha todos os picos acima do limiar

    if len(picos) == 0:                                                         # nenhum pico encontrado considera o vídeo inteiro
        return 0, len(movimentos)

    primeiro_pico = picos[0]                                        # índice do primeiro pico de movimento
    ultimo_pico = picos[-1]                                         # índice do último pico de movimento

    margem_inicio = max(5, int(len(movimentos) * 0.03))             # margem de segurança no início (mínimo 5 frames)
    margem_fim = max(5, int(len(movimentos) * 0.03))                # margem de segurança no fim (mínimo 5 frames)

    inicio = max(0, primeiro_pico - margem_inicio)                  # frame de início com margem, sem sair do vídeo
    fim = min(len(movimentos), ultimo_pico + margem_fim)            # frame de fim com margem, sem ultrapassar o total

    return inicio, fim                                              # retorna o intervalo de movimento relevante


#faz o processamento dos vídeos, recortes, redimensionamentos e etc
def processar_video(video_path, output_folder, label):
    max_frames = 30
    img_size = 128
    frame_idx = 0                                                   # contador de frames lidos
    frames_validos = []                                             # lista de frames que passaram pela YOLO com sucesso
    frames_finais = []                                              # lista final de frames a serem salvos
    
    cap = cv2.VideoCapture(video_path)                              # abre o vídeo
    inicio_real, fim_real = detectar_intervalo_movimento(video_path)            # detecta intervalo com movimento

    os.makedirs(os.path.join(output_folder, label), exist_ok=True)              # cria a pasta de saída da classe

    while True:
        ret, frame = cap.read()                                     # lê o próximo frame
        if not ret:                                                 # fim do vídeo
            break
        frame_idx += 1                                              # incrementa o contador de frames

        if frame_idx < inicio_real or frame_idx > fim_real:         # frame fora do intervalo de movimento
            continue                                                # pula o frame

        try:
            resultados = model(frame, conf=0.3, verbose=False)      # roda a YOLO com confiança mínima de 30%
            boxes = resultados[0].boxes                             # bounding boxes detectadas
            if boxes is None or len(boxes) == 0:                    # nenhuma detecção
                continue                                            # pula o frame

            # maior bbox
            areas = []                                              # lista de áreas das bboxes detectadas
            for box in boxes.xyxy:                                  # itera sobre cada bbox no formato (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box)                      # converte coordenadas para inteiro
                area = (x2 - x1) * (y2 - y1)                        # calcula a área da bbox
                areas.append(area)                                  # adiciona à lista

            idx = areas.index(max(areas))                           # índice da bbox com maior área (mão mais próxima)
            x1, y1, x2, y2 = map(int, boxes.xyxy[idx])              # coordenadas da bbox principal
            h_img, w_img, _ = frame.shape                           # dimensões do frame original

            # margens maiores
            margin_x = int((x2 - x1) * 0.35)                        # margem horizontal: 35% da largura da bbox
            margin_y = int((y2 - y1) * 0.45)                        # margem vertical: 45% da altura da bbox

            x1 = max(0, x1 - margin_x)                              # expande à esquerda sem sair da imagem
            y1 = max(0, y1 - margin_y)                              # expande para cima sem sair da imagem
            x2 = min(w_img, x2 + margin_x)                          # expande à direita sem ultrapassar a largura
            y2 = min(h_img, y2 + margin_y)                          # expande para baixo sem ultrapassar a altura

            frame = frame[y1:y2, x1:x2]                             # recorta a região de interesse do frame


            # padding -  torna o frame quadrado sem distorcer
            h, w, _ = frame.shape                                   # dimensões do recorte
            size = max(h, w)                                        # lado do quadrado que envolve o recorte
            frame = cv2.copyMakeBorder(
                frame,
                (size - h) // 2, size - h - (size - h) // 2,        # padding superior e inferior
                (size - w) // 2, size - w - (size - w) // 2,        # padding esquerdo e direito
                cv2.BORDER_CONSTANT, value=[0, 0, 0]                # preenche com preto
            ) 

            frame = cv2.resize(frame, (img_size, img_size))         # redimensiona para o tamanho padrão
            frames_validos.append(frame)                            # armazena o frame válido processado

        except Exception as e:
            print(f"Erro ao processar frame {frame_idx}: {e}")
            continue

    cap.release()                                                   # libera o recurso de vídeo

    #após válidar os frames, agora faz o recorte pra usar de 4 em 4 frames
    if len(frames_validos) == 0:
        frames_finais = [np.zeros((img_size, img_size, 3), dtype=np.uint8)] * max_frames    # se a YOLO falhou no vídeo todo, gera 30 frames pretos
    else:
        if len(frames_validos) <= 8:                                # Se o vídeo original tiver 8 frames ou menos, usa todos diretamente
            frames_finais = frames_validos.copy()
        else:
            frames_finais = frames_validos[::4]                     # Para vídeos maiores, primeiro tenta a amostragem 1 a cada 4
            if len(frames_finais) < 8:                              # Se a amostragem resultou em menos de 8, redistribui 8 frames uniformemente
                indices = np.linspace(0, len(frames_validos) - 1, 8).astype(int)  # 8 índices uniformemente espaçados
                for i in indices:                                   # seleciona os frames nos índices calculados
                    idx = int(i)
                    if idx < 0:
                        idx = 0
                    if idx >= len(frames_validos):
                        idx = len(frames_validos) - 1
                    frames_finais.append(frames_validos[idx].copy())

        frames_finais = frames_finais[:max_frames]                  # garante no máximo 30 frames

        ultimo_frame = frames_finais[-1]                            # Se não deu 30 frames, repete o último frame até preencher, referência ao último frame válido
        while len(frames_finais) < max_frames:                      # enquanto não tiver 30 frames
            frames_finais.append(ultimo_frame)                      # replica o último frame 


    for i, frame_final in enumerate(frames_finais):                         # itera sobre os 30 frames finais
        output_path = os.path.join(output_folder, label, f"frame_{i}.jpg")  # caminho de saída do frame
        cv2.imwrite(output_path, frame_final)                               # salva o frame como JPEG


#agora faz o processo com todos os vídeos
def processar_dataset():
    input_folder = INPUT                                            # pasta de entrada com os vídeos originais
    output_folder = OUTPUT                                          # pasta de saída com os frames processados
    video_files = []
    resultados = []                                                 # lista que acumulará as estatísticas de cada vídeo
    contador_palavras = {}                                          # dicionário para contar ocorrências de cada palavra/sinal
    
    for f in os.listdir(input_folder):                              # lista todos os arquivos na pasta de entrada
        if f.lower().endswith(".mp4") and os.path.isfile(os.path.join(input_folder, f)):
            video_files.append(f)
    print(f"Total de vídeos: {len(video_files)}\n")             
    
    inicio_total = datetime.now()                                   # marca o início 
    for file in tqdm(video_files, desc="Processando vídeos"):       # barra de progresso
        video_path = os.path.join(input_folder, file)               # caminho completo do vídeo
        palavra = extrair_palavra(file)                             # extrai a classe a partir do nome do arquivo
        
        if palavra not in contador_palavras:                        # primeira vez que essa palavra aparece
            contador_palavras[palavra] = 0                          # inicializa o contador

        contador_palavras[palavra] += 1                             # incrementa o contador da palavra
        label = f"{palavra}_{contador_palavras[palavra]}"           # label único: ex. "oi_1", "oi_2"
        inicio_video = datetime.now()                               # marca início do processamento deste vídeo
        processar_video(video_path, output_folder, label)           # processa o vídeo
        
        tempo_video = (datetime.now() - inicio_video).total_seconds()  # calcula tempo de processamento do vídeo em segundos
        resultados.append({
            "video": file,                                          # nome do arquivo de vídeo
            "classe": label,                                        # label gerado para este vídeo
            "tempo_processamento_s": tempo_video                    # tempo gasto em segundos
        })
    tempo_total = (datetime.now() - inicio_total).total_seconds()  # tempo total de execução em segundos

    print("\n===== FINAL =====")
    print(f"Tempo total: {(tempo_total/60):.2f} minutos")  # exibe o tempo total em minutos
    os.makedirs("reports", exist_ok=True)  # cria a pasta de relatórios se não existir
    csv_path = (f'reports/estatisticas_'f'{datetime.now().strftime("%d_%m_%H_%M")}.csv')  # nome do CSV com timestamp para evitar sobrescrita

    with open(csv_path, "w", newline="", encoding="utf-8") as f:  # abre o CSV para escrita
        writer = csv.DictWriter(f, fieldnames=resultados[0].keys())  # usa as chaves do primeiro resultado como cabeçalho
        writer.writeheader()  # escreve a linha de cabeçalho
        writer.writerows(resultados)  # escreve todas as linhas de dados

    print(f"\nCSV salvo em: {csv_path}")  # informa onde o relatório foi salvo
