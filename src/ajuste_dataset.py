import os
import cv2
import csv
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime

model = YOLO("yolo11n.pt")

def processar_video(video_path, output_folder, label):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_pular = int(fps * 1)  # 🔥 corta 1 segundo

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

        # 🔥 IGNORA INÍCIO
        if frame_quant < frames_pular:
            continue

        # 🔥 IGNORA FINAL
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

                areas_validas = []

                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]

                    if area > 300:
                        areas_validas.append(area)

                if len(areas_validas) == 0:
                    frames_com_movimento = max(0, frames_com_movimento - 1)
                    descartado_frames += 1
                    sem_movimento += 1
                    continue

                maior_area = max(areas_validas)

                h_m, w_m = thresh.shape
                area_total = h_m * w_m

                ratio = maior_area / area_total

                if ratio < 0.004:
                    frames_com_movimento = max(0, frames_com_movimento - 1)
                    descartado_frames += 1
                    sem_movimento += 1
                    continue

                parte_cima = thresh[0:int(h_m*0.6), :]
                parte_baixo = thresh[int(h_m*0.6):h_m, :]

                cima = cv2.countNonZero(parte_cima)
                baixo = cv2.countNonZero(parte_baixo)

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

            h, w, _ = frame.shape
            size = max(h, w)

            top = (size - h) // 2
            bottom = size - h - top
            left = (size - w) // 2
            right = size - w - left

            frame = cv2.copyMakeBorder(
                frame, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=[0,0,0]
            )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (224, 224))

            # 🔥 remove duplicados
            if last_saved_frame is not None:
                diff_final = cv2.absdiff(last_saved_frame, frame)
                if diff_final.mean() < 1.5:
                    continue

            output_path = os.path.join(output_folder, label, f"{label}_{processado_frames}.jpg")
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

    total_frames = 0
    total_processado = 0
    total_descartado = 0
    total_yolo_falhas = 0
    total_sem_movimento = 0

    inicio_total = datetime.now()

    print(f"Total de vídeos: {len(video_files)}\n")

    for file in tqdm(video_files, desc="Processando vídeos"):

        video_path = os.path.join(input_folder, file)
        label = file.split("_")[0]

        inicio_video = datetime.now()

        estatistica = processar_video(video_path, output_folder, label)

        fim_video = datetime.now()
        tempo_video = (fim_video - inicio_video).total_seconds()

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

    fim_total = datetime.now()
    tempo_total = (fim_total - inicio_total).total_seconds()

    minutos = int(tempo_total // 60)
    segundos = int(tempo_total % 60)

    print("\n===== ESTATÍSTICAS GERAIS =====")
    print(f"Total de frames: {total_frames}")
    print(f"Processados: {total_processado}")
    print(f"Descartados: {total_descartado}")
    print(f"Falhas YOLO: {total_yolo_falhas}")
    print(f"Sem movimento: {total_sem_movimento}")
    print(f"Tempo total: {minutos} min {segundos} s")

    if total_frames > 0:
        aproveitamento = (total_processado / total_frames) * 100
        print(f"Aproveitamento: {aproveitamento:.2f}%")

    resultados.append({
        "video": "TOTAL",
        "classe": "-",
        "total_frames": total_frames,
        "processados": total_processado,
        "descartados": total_descartado,
        "falhas_yolo": total_yolo_falhas,
        "sem_movimento": total_sem_movimento,
        "tempo_processamento_s": tempo_total
    })

    os.makedirs("reports", exist_ok=True)

    data_hora_atual = datetime.now()
    csv_path = f'reports/estatisticas_processamento_{data_hora_atual.strftime("%d_%m_%H_%M")}.csv'

    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "video",
            "classe",
            "total_frames",
            "processados",
            "descartados",
            "falhas_yolo",
            "sem_movimento",
            "tempo_processamento_s"
        ])

        writer.writeheader()
        writer.writerows(resultados)

    print(f"\nCSV salvo em: {csv_path}")