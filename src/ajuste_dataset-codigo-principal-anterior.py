import os
import cv2
import csv
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime

model = YOLO("yolo11n.pt")

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
            # 🔥 YOLO mais sensível
            resultados = model(frame, conf=0.3, verbose=False)

            boxes = resultados[0].boxes

            if boxes is None or len(boxes) == 0:
                descartado_frames += 1
                yolo_falhas += 1
                continue

            # 🔥 maior bounding box
            areas = []
            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                areas.append((x2 - x1) * (y2 - y1))

            idx = areas.index(max(areas))
            box = boxes.xyxy[idx]

            x1, y1, x2, y2 = map(int, box)

            h_img, w_img, _ = frame.shape

            # 🔥 expansão
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.3)

            x1_new = max(0, x1 - margin_x)
            y1_new = max(0, y1 - margin_y)
            x2_new = min(w_img, x2 + margin_x)
            y2_new = min(h_img, y2 + margin_y)

            frame = frame[y1_new:y2_new, x1_new:x2_new]

            if frame is None or frame.size == 0:
                descartado_frames += 1
                continue

            # 🔥 FILTRO DE MOVIMENTO MELHORADO
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 🔥 remove ruído fino
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_frame is not None:
                h, w = gray.shape

                regiao_maos = gray[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]
                prev_maos = prev_frame[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]

                regiao_baixa = gray[int(h*0.7):h, int(w*0.2):int(w*0.8)]
                prev_baixa = prev_frame[int(h*0.7):h, int(w*0.2):int(w*0.8)]

                # 🔥 diferença + threshold mais forte
                diff_maos = cv2.absdiff(prev_maos, regiao_maos)
                _, thresh_maos = cv2.threshold(diff_maos, 30, 255, cv2.THRESH_BINARY)

                diff_baixo = cv2.absdiff(prev_baixa, regiao_baixa)
                _, thresh_baixo = cv2.threshold(diff_baixo, 30, 255, cv2.THRESH_BINARY)

                # 🔥 CONTAGEM de pixels em movimento
                movimento_maos = cv2.countNonZero(thresh_maos)
                movimento_baixo = cv2.countNonZero(thresh_baixo)

                # 🔥 normalização (ratio)
                area_maos = thresh_maos.shape[0] * thresh_maos.shape[1]
                area_baixo = thresh_baixo.shape[0] * thresh_baixo.shape[1]

                movimento_maos_ratio = movimento_maos / area_maos
                movimento_baixo_ratio = movimento_baixo / area_baixo

                # 🔥 regra forte (remove frames parados)
                if movimento_maos_ratio < 0.01 and movimento_baixo_ratio < 0.01:
                    descartado_frames += 1
                    sem_movimento += 1
                    continue

                # 🔥 remove movimento irrelevante (baixo)
                if movimento_maos_ratio < 0.005 and movimento_baixo_ratio > movimento_maos_ratio:
                    descartado_frames += 1
                    sem_movimento += 1
                    continue

            # 🔥 atualiza mais devagar (ESSENCIAL)
            if frame_quant % 2 == 0:
                prev_frame = gray

            # 🔥 deixar quadrado
            h, w, _ = frame.shape
            size = max(h, w)

            top = (size - h) // 2
            bottom = size - h - top
            left = (size - w) // 2
            right = size - w - left

            frame = cv2.copyMakeBorder(
                frame,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            # 🔹 preprocess final
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (224, 224))

            # 🔹 salvar
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

    print("\n===== ESTATÍSTICAS GERAIS =====")
    print(f"Total de frames: {total_frames}")
    print(f"Processados: {total_processado}")
    print(f"Descartados: {total_descartado}")
    print(f"Falhas YOLO: {total_yolo_falhas}")
    print(f"Sem movimento: {total_sem_movimento}")
    print(f"Tempo total de processamento: {tempo_total:.2f}s")

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