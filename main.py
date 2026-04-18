import os
import src.ajuste_dataset as ad
import src.download_dataset as dd
from src.rede_neural.treino import treinar
import src.ajuste_dataset_destaquemaos as teste_destaque
from src.rede_neural.predicao import executar_previsao
from src.rede_neural.predicao import executar_previsao
from src.rede_neural.dataset import LibrasDatasetSequencia



def main():

    # passo 1 - baixar o dataser
    print("1. Baixando dataset...")
    dd.download_minds_libras()


    # passo 2 - fazer o pré-processamento das imagens com o yolo
    caminho = "data/pre_processado/yolo11_resultado_testes"
    if os.path.exists(caminho) and os.path.isdir(caminho) and os.listdir(caminho):
        print("O dataset já foi pré processado!")
    elif os.path.exists(caminho) and os.path.isdir(caminho):
        print("\n2. Processando vídeos...")
        ad.processar_dataset()
    else:
        print("\n2. Processando vídeos...")
        ad.processar_dataset()

    # passo 3 - treinar rede neural
    treinar() 

    # passo 4 - execução do treino
    dataset = LibrasDatasetSequencia("data/pre_processado/yolo11_resultado_testes")
    executar_previsao(
        "data/pre_processado/yolo11_resultado_testes",
        dataset.classes
    )

    print("Pipeline concluído!")

if __name__ == "__main__":
    main()