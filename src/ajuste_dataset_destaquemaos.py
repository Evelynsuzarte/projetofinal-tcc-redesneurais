import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime

model = YOLO("yolov8n.pt")
hand_model = YOLO("best.pt")

def processar_video(video_path, output_folder, label):

    cap = cv2.VideoCapture(video_path)

    frame_quant = 0
    processado_frames = 0
    descartado_frames = 0
    yolo_falhas = 0
    sem_movimento = 0

    prev_frame = None

    os.makedirs(os.path.join(output_folder, label), exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_quant += 1

        try:
            resultados = model(frame)
            detectado = False

            for resultado in resultados:
                for box, cls in zip(resultado.boxes.xyxy, resultado.boxes.cls):

                    if int(cls) == 0:  # pessoa

                        x1, y1, x2, y2 = map(int, box)

                        h_img, w_img, _ = frame.shape

                        margin_x = int((x2 - x1) * 0.2)
                        margin_y = int((y2 - y1) * 0.3)

                        x1_new = max(0, x1 - margin_x)
                        y1_new = max(0, y1 - margin_y)
                        x2_new = min(w_img, x2 + margin_x)
                        y2_new = min(h_img, y2 + margin_y)

                        pessoa = frame[y1_new:y2_new, x1_new:x2_new]

                        if pessoa.size == 0:
                            continue

                        frame = pessoa
                        detectado = True
                        break

                if detectado:
                    break

            if not detectado:
                descartado_frames += 1
                yolo_falhas += 1
                continue

            if frame is None or frame.size == 0:
                descartado_frames += 1
                continue

            # 🔥 DETECÇÃO DE MÃOS
            resultados_mao = hand_model(frame)
            detectou_mao = False

            maior_area = 0
            melhor_box = None

            for resultado in resultados_mao:
                for box in resultado.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)

                    area = (x2 - x1) * (y2 - y1)

                    if area > maior_area:
                        maior_area = area
                        melhor_box = (x1, y1, x2, y2)

            if melhor_box:
                x1, y1, x2, y2 = melhor_box

                h_img, w_img, _ = frame.shape

                margin_x = int((x2 - x1) * 0.3)
                margin_y = int((y2 - y1) * 0.3)

                x1_new = max(0, x1 - margin_x)
                y1_new = max(0, y1 - margin_y)
                x2_new = min(w_img, x2 + margin_x)
                y2_new = min(h_img, y2 + margin_y)

                # 🔥 MÁSCARA (AO INVÉS DE CORTAR)
                mask = np.zeros_like(frame)

                cv2.rectangle(
                    mask,
                    (x1_new, y1_new),
                    (x2_new, y2_new),
                    (255, 255, 255),
                    -1
                )

                # opcional: suavizar borda
                mask = cv2.GaussianBlur(mask, (35, 35), 0)

                frame = cv2.bitwise_and(frame, mask)

                detectou_mao = True

            if not detectou_mao:
                descartado_frames += 1
                yolo_falhas += 1
                continue

            # 🔥 FILTRO DE MOVIMENTO
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                h, w = gray.shape

                regiao_maos = gray[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]
                prev_maos = prev_frame[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]

                regiao_baixa = gray[int(h*0.7):h, int(w*0.2):int(w*0.8)]
                prev_baixa = prev_frame[int(h*0.7):h, int(w*0.2):int(w*0.8)]

                movimento_maos = cv2.absdiff(prev_maos, regiao_maos).mean()
                movimento_baixo = cv2.absdiff(prev_baixa, regiao_baixa).mean()

                if movimento_maos < 5 and movimento_baixo < 5:
                    descartado_frames += 1
                    sem_movimento += 1
                    continue

                if movimento_maos < 3 and movimento_baixo > movimento_maos:
                    descartado_frames += 1
                    sem_movimento += 1
                    continue

            prev_frame = gray

            # 🔥 PADRONIZAÇÃO
            h, w, _ = frame.shape
            size = max(h, w)

            top = (size - h) // 2
            bottom = size - h - top
            left = (size - w) // 2
            right = size - w - left

            frame = cv2.copyMakeBorder(
                frame, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

            frame = cv2.resize(frame, (224, 224))

            output_path = os.path.join(output_folder, label, f"{label}_{processado_frames}.jpg")
            cv2.imwrite(output_path, frame)

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
    output_folder = "data/pre_processado/000_resultado_testes"

    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    resultados = []

    total_frames = 0
    total_processado = 0
    total_descartado = 0
    total_yolo_falhas = 0
    total_sem_movimento = 0

    print(f"Total de vídeos: {len(video_files)}\n")

    for file in tqdm(video_files, desc="Processando vídeos"):

        video_path = os.path.join(input_folder, file)
        label = file.split("_")[0]

        estatistica = processar_video(video_path, output_folder, label)

        resultados.append({
            "video": file,
            "classe": label,
            "total_frames": estatistica["total_frames"],
            "processados": estatistica["processado"],
            "descartados": estatistica["descartado"],
            "falhas_yolo": estatistica["yolo_falhas"],
            "sem_movimento": estatistica["sem_movimento"]
        })

        total_frames += estatistica["total_frames"]
        total_processado += estatistica["processado"]
        total_descartado += estatistica["descartado"]
        total_yolo_falhas += estatistica["yolo_falhas"]
        total_sem_movimento += estatistica["sem_movimento"]

    print("\n===== ESTATÍSTICAS GERAIS =====")
    print(f"Total de frames: {total_frames}")
    print(f"Processados: {total_processado}")
    print(f"Descartados: {total_descartado}")
    print(f"Falhas YOLO: {total_yolo_falhas}")
    print(f"Sem movimento: {total_sem_movimento}")

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
        "sem_movimento": total_sem_movimento
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
            "sem_movimento"
        ])

        writer.writeheader()
        writer.writerows(resultados)

    print(f"\nCSV salvo em: {csv_path}")
