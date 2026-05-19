import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime

model = YOLO("yolo11n.pt")


def extrair_palavra(nome):
    nome = nome.replace(".mp4", "")
    while len(nome) > 0 and nome[0].isdigit():
        nome = nome[1:]

    if "Sinalizador" in nome:
        nome = nome.split("Sinalizador")[0]

    return nome.lower()


def detectar_intervalo_movimento(video_path):
    cap = cv2.VideoCapture(video_path)
    movimentos = []
    frames_gray = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        frames_gray.append(gray)
        if len(frames_gray) > 1:
            diff = cv2.absdiff(frames_gray[-2], frames_gray[-1])
            movimentos.append(diff.mean())
        else:
            movimentos.append(0)

    cap.release()

    if len(movimentos) == 0:
        return 0, 0

    movimentos = np.array(movimentos, dtype=np.float32)
    movimentos = np.convolve(movimentos, np.ones(15) / 15, mode='same')

    limiar = np.percentile(movimentos, 60) + np.std(movimentos) * 0.8

    # acha todos os picos acima do limiar
    from scipy.signal import find_peaks
    picos, _ = find_peaks(movimentos, height=limiar, distance=10)

    if len(picos) == 0:
        return 0, len(movimentos)

    primeiro_pico = picos[0]
    ultimo_pico   = picos[-1]

    margem_inicio = max(5, int(len(movimentos) * 0.03))
    margem_fim    = max(5, int(len(movimentos) * 0.03))

    inicio = max(0, primeiro_pico - margem_inicio)
    fim    = min(len(movimentos), ultimo_pico + margem_fim)

    return inicio, fim


def processar_video(video_path, output_folder, label):
    cap = cv2.VideoCapture(video_path)

    inicio_real, fim_real = detectar_intervalo_movimento(video_path)

    frame_idx = 0
    processado_frames = 0
    descartado_frames = 0
    yolo_falhas = 0
    sem_movimento = 0

    prev_gray = None
    last_saved_frame = None

    os.makedirs(os.path.join(output_folder, label), exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # corta início/fim sem movimento
        if frame_idx < inicio_real or frame_idx > fim_real:
            continue
        try:
            resultados = model(frame, conf=0.3, verbose=False)
            boxes = resultados[0].boxes
            if boxes is None or len(boxes) == 0:
                descartado_frames += 1
                yolo_falhas += 1
                continue
            # maior bbox
            areas = []
            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                areas.append(area)

            idx = areas.index(max(areas))
            x1, y1, x2, y2 = map(int, boxes.xyxy[idx])
            h_img, w_img, _ = frame.shape

            # margens maiores
            margin_x = int((x2 - x1) * 0.35)
            margin_y = int((y2 - y1) * 0.45)

            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w_img, x2 + margin_x)
            y2 = min(h_img, y2 + margin_y)

            frame = frame[y1:y2, x1:x2]

            if frame is None or frame.size == 0:
                descartado_frames += 1
                continue

            # grayscale SOMENTE para análise
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # filtro de movimento
            # if prev_gray is not None:
            #     h, w = gray.shape
            #     regiao = gray[
            #         int(h*0.25):int(h*0.55),
            #         int(w*0.3):int(w*0.7)
            #     ]
            #     prev_regiao = prev_gray[
            #         int(h*0.25):int(h*0.55),
            #         int(w*0.3):int(w*0.7)
            #     ]
            #     diff = cv2.absdiff(prev_regiao, regiao)
            #     _, thresh = cv2.threshold(
            #         diff,
            #         20,
            #         255,
            #         cv2.THRESH_BINARY
            #     )
            #     movimento = cv2.countNonZero(thresh)
            #     area = thresh.shape[0] * thresh.shape[1]
            #     if (movimento / area) < 0.003:
            #         descartado_frames += 1
            #         sem_movimento += 1
            #         continue

            if frame_idx % 5 == 0:
                prev_gray = gray

            # padding
            h, w, _ = frame.shape
            size = max(h, w)
            frame = cv2.copyMakeBorder(
                frame,
                (size - h) // 2,
                size - h - (size - h) // 2,
                (size - w) // 2,
                size - w - (size - w) // 2,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            # mantém colorido
            frame = cv2.resize(frame, (224, 224))

            # remover duplicados
            # if last_saved_frame is not None:
            #     gray_last = cv2.cvtColor(
            #         last_saved_frame,
            #         cv2.COLOR_BGR2GRAY
            #     )
            #     gray_current = cv2.cvtColor(
            #         frame,
            #         cv2.COLOR_BGR2GRAY
            #     )
            #     diff = cv2.absdiff(
            #         gray_last,
            #         gray_current
            #     )
            #     diff = cv2.GaussianBlur(diff, (5, 5), 0)
            #     diff_mean = diff.mean()
            #     _, thresh = cv2.threshold(
            #         diff,
            #         15,
            #         255,
            #         cv2.THRESH_BINARY
            #     )
            #     pixels_diff = cv2.countNonZero(thresh)
            #     total_pixels = thresh.shape[0] * thresh.shape[1]
            #     ratio = pixels_diff / total_pixels
            #     if diff_mean < 3 and ratio < 0.01:
            #         continue

            output_path = os.path.join(
                output_folder,
                label,
                f"frame_{processado_frames}.jpg"
            )
            cv2.imwrite(output_path, frame)
            last_saved_frame = frame.copy()
            processado_frames += 1

        except Exception:
            descartado_frames += 1
            continue

    cap.release()

    return {
        "total_frames": frame_idx,
        "processado": processado_frames,
        "descartado": descartado_frames,
        "yolo_falhas": yolo_falhas,
        "sem_movimento": sem_movimento
    }


def processar_dataset():

    input_folder = "data/origin"
    output_folder = "data/pre_processado/teste_rede_neural_15_05_com_melhoria"

    video_files = [
        f for f in os.listdir(input_folder)
        if f.endswith(".mp4")
    ]

    resultados = []
    inicio_total = datetime.now()
    print(f"Total de vídeos: {len(video_files)}\n")
    contador_palavras = {}

    for file in tqdm(video_files, desc="Processando vídeos"):
        video_path = os.path.join(input_folder, file)
        palavra = extrair_palavra(file)
        if palavra not in contador_palavras:
            contador_palavras[palavra] = 0

        contador_palavras[palavra] += 1
        label = f"{palavra}_{contador_palavras[palavra]}"
        inicio_video = datetime.now()
        estatistica = processar_video(
            video_path,
            output_folder,
            label
        )
        tempo_video = (
            datetime.now() - inicio_video
        ).total_seconds()
        resultados.append({
            "video": file,
            "classe": label,
            "tempo_processamento_s": tempo_video
        })

    tempo_total = (
        datetime.now() - inicio_total
    ).total_seconds()

    print("\n===== FINAL =====")
    print(f"Tempo total: {(tempo_total/60):.2f} minutos")

    os.makedirs("reports", exist_ok=True)

    csv_path = (
        f'reports/estatisticas_'
        f'{datetime.now().strftime("%d_%m_%H_%M")}.csv'
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=resultados[0].keys()
        )
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\nCSV salvo em: {csv_path}")