import src.ajuste_dataset as ad
import src.download_dataset as dd

import src.ajuste_dataset_destaquemaos as teste_destaque


def main():

    print("1. Baixando dataset...")
    dd.download_minds_libras()

    print("\n2. Processando vídeos...")
    ad.processar_dataset()
    #teste_destaque.processar_dataset()

    print("Pipeline concluído!")

if __name__ == "__main__":
    main()