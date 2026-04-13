import os
import cv2
import csv
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime

model = YOLO("yolo11n.pt")


def extrair_palavra(nome):
    nome = nome.replace(".mp4", "")

    # remove números do início
    while len(nome) > 0 and nome[0].isdigit():
        nome = nome[1:]

    # corta "Sinalizador"
    if "Sinalizador" in nome:
        nome = nome.split("Sinalizador")[0]

    return nome.lower()


def processar_video(video_path, output_folder, label):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30  # fallback

    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_pular = int(fps * 1)  # 1 segundo

    frame_quant = 0
    processado_frames = 0
    descartado_frames = 0
    yolo_falhas = 0
    sem_movimento = 0

    prev_frame = None
    last_saved_frame = None
    frames_com_movimento = 0

    os.makedirs(os.path.join(output_folder, label), exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_quant += 1

        # corta início
        if frame_quant < frames_pular:
            continue

        # corta final
        if frame_quant > (total_frames_video - frames_pular):
            break

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
                areas.append((x2 - x1) * (y2 - y1))

            idx = areas.index(max(areas))
            x1, y1, x2, y2 = map(int, boxes.xyxy[idx])

            h_img, w_img, _ = frame.shape

            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.3)

            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w_img, x2 + margin_x)
            y2 = min(h_img, y2 + margin_y)

            frame = frame[y1:y2, x1:x2]

            if frame is None or frame.size == 0:
                descartado_frames += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_frame is not None:
                h, w = gray.shape

                regiao = gray[int(h*0.25):int(h*0.55), int(w*0.3):int(w*0.7)]
                prev_regiao = prev_frame[int(h*0.25):int(h*0.55), int(w*0.3):int(w*0.7)]

                diff = cv2.absdiff(prev_regiao, regiao)
                _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)

                areas_validas = [
                    stats[i, cv2.CC_STAT_AREA]
                    for i in range(1, num_labels)
                    if stats[i, cv2.CC_STAT_AREA] > 300
                ]

                if not areas_validas:
                    frames_com_movimento = max(0, frames_com_movimento - 1)
                    descartado_frames += 1
                    sem_movimento += 1
                    continue

                maior_area = max(areas_validas)
                area_total = thresh.shape[0] * thresh.shape[1]

                if (maior_area / area_total) < 0.004:
                    frames_com_movimento = max(0, frames_com_movimento - 1)
                    descartado_frames += 1
                    sem_movimento += 1
                    continue

                # remove braço baixo
                h_m = thresh.shape[0]
                cima = cv2.countNonZero(thresh[0:int(h_m*0.6), :])
                baixo = cv2.countNonZero(thresh[int(h_m*0.6):, :])

                if baixo > cima:
                    frames_com_movimento = max(0, frames_com_movimento - 1)
                    descartado_frames += 1
                    sem_movimento += 1
                    continue

                frames_com_movimento += 1

                if frames_com_movimento < 2:
                    continue

            if frame_quant % 5 == 0:
                prev_frame = gray

            # pad quadrado
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

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (224, 224))

            # remove duplicados
            if last_saved_frame is not None:
                if cv2.absdiff(last_saved_frame, frame).mean() < 1.5:
                    continue

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
        "total_frames": frame_quant,
        "processado": processado_frames,
        "descartado": descartado_frames,
        "yolo_falhas": yolo_falhas,
        "sem_movimento": sem_movimento
    }


def processar_dataset():

    input_folder = "data/origin/testes_rapidos"
    output_folder = "data/pre_processado/yolo11_resultado_testes"

    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    resultados = []

    total_frames = total_processado = total_descartado = 0
    total_yolo_falhas = total_sem_movimento = 0

    inicio_total = datetime.now()

    print(f"Total de vídeos: {len(video_files)}\n")

    contador_palavras = {}

    for file in tqdm(video_files, desc="Processando vídeos"):

        video_path = os.path.join(input_folder, file)

        palavra = extrair_palavra(file)

        if palavra not in contador_palavras:
            contador_palavras[palavra] = 0

        contador_palavras[palavra] += 1

        # 🔥 cada vídeo vira uma pasta única
        label = f"{palavra}_{contador_palavras[palavra]}"

        inicio_video = datetime.now()

        estatistica = processar_video(video_path, output_folder, label)

        tempo_video = (datetime.now() - inicio_video).total_seconds()

        resultados.append({
            "video": file,
            "classe": label,
            "total_frames": estatistica["total_frames"],
            "processados": estatistica["processado"],
            "descartados": estatistica["descartado"],
            "falhas_yolo": estatistica["yolo_falhas"],
            "sem_movimento": estatistica["sem_movimento"],
            "tempo_processamento_s": tempo_video
        })

        total_frames += estatistica["total_frames"]
        total_processado += estatistica["processado"]
        total_descartado += estatistica["descartado"]
        total_yolo_falhas += estatistica["yolo_falhas"]
        total_sem_movimento += estatistica["sem_movimento"]

    tempo_total = (datetime.now() - inicio_total).total_seconds()

    print("\n===== ESTATÍSTICAS GERAIS =====")
    print(f"Tempo total: {tempo_total:.2f}s")

    os.makedirs("reports", exist_ok=True)

    csv_path = f'reports/estatisticas_{datetime.now().strftime("%d_%m_%H_%M")}.csv'

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\nCSV salvo em: {csv_path}")