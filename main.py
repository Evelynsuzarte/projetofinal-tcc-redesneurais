import kagglehub
import shutil
import os

def download_minds_libras():

    print("Baixando dataset do Kaggle...")
    path = kagglehub.dataset_download("j0aopsantos/minds-libras")
    print("Dataset baixado em:", path)

    destination = os.path.join("data", "origin")
    os.makedirs(destination, exist_ok=True)

    for file in os.listdir(path):
        src_file = os.path.join(path, file)
        dst_file = os.path.join(destination, file)

        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
    print("Dataset copiado para data/origin com sucesso!")

if __name__ == "__main__":
    download_minds_libras()