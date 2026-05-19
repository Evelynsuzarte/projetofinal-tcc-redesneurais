import os
import src.ajuste_dataset as ad
import src.download_dataset as dd
import papermill as pm
import src.ajuste_dataset_destaquemaos as teste_destaque



def main():

    # passo 1 - baixar o dataser
    print("1. Baixando dataset...")
    dd.download_minds_libras()

    print("\n2. Processando vídeos...")
    ad.processar_dataset()

    # print("\n3. Processando rede neural...")


    # notebook_entrada = 'lstm_gemini.ipynb'
    # notebook_saida = 'lstm_gemini_resultado.ipynb' # Ele vai criar este arquivo com os gráficos salvos

    # pm.execute_notebook(notebook_entrada,notebook_saida)

    # print("✅ Execução finalizada!")



    # # passo 2 - fazer o pré-processamento das imagens com o yolo
    # caminho = "data/pre_processado/yolo11_resultado_testes"
    # if os.path.exists(caminho) and os.path.isdir(caminho) and os.listdir(caminho):
    #     print("O dataset já foi pré processado!")
    # elif os.path.exists(caminho) and os.path.isdir(caminho):
    #     print("\n2. Processando vídeos...")
    #     ad.processar_dataset()
    # else:
    #     print("\n2. Processando vídeos...")
    #     ad.processar_dataset()

    # passo 3 - treinar rede neural
    #treinar() 

    # passo 4 - execução do treino
    #dataset = LibrasDatasetSequencia("data/pre_processado/yolo11_resultado_testes")
    #executar_previsao(
    #    "data/pre_processado/yolo11_resultado_testes",
    #    dataset.classes
    #)

    print("Pipeline concluído!")

if __name__ == "__main__":
    main()