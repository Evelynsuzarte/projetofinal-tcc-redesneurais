import os
import cv2
import torch
from torch.utils.data import Dataset


class LibrasDatasetSequencia(Dataset):
    def __init__(self, base_path, max_frames=20):
        """
        base_path = pasta com vídeos já processados
        cada subpasta = 1 vídeo (sequência)
        """
        self.base_path = base_path
        self.max_frames = max_frames

        self.samples = []   # lista de (frames, label)
        self.classes = []   # nomes das classes

        # pega todas as pastas (cada vídeo)
        folders = sorted([
            f for f in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, f))
        ])

        # classes = palavra base (acontecer_1 -> acontecer)
        self.classes = sorted(list(set([
            f.split("_")[0] for f in folders
        ])))

        # mapa classe → índice
        self.class_to_idx = {
            c: i for i, c in enumerate(self.classes)
        }

        # monta dataset
        for folder in folders:
            label_name = folder.split("_")[0]
            label = self.class_to_idx[label_name]

            folder_path = os.path.join(base_path, folder)

            frames = sorted(os.listdir(folder_path))
            frame_paths = [
                os.path.join(folder_path, f)
                for f in frames
            ]

            self.samples.append((frame_paths, label))


    def __len__(self):
        return len(self.samples)


    def load_frame(self, path):
        """
        carrega e normaliza frame
        """
        img = cv2.imread(path)

        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        img = torch.tensor(img, dtype=torch.float32) / 255.0

        # (H, W, C) → (C, H, W)
        img = img.permute(2, 0, 1)

        return img


    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        frames = []

        # carrega sequência de frames
        for p in frame_paths[:self.max_frames]:
            frame = self.load_frame(p)
            if frame is not None:
                frames.append(frame)

        # padding se tiver poucos frames
        while len(frames) < self.max_frames:
            frames.append(torch.zeros(3, 224, 224))

        # (T, C, H, W)
        frames = torch.stack(frames)

        return frames, torch.tensor(label)