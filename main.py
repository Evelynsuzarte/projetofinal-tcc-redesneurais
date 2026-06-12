import src.ajuste_dataset as ad
import src.download_dataset as dd
import subprocess
import sys

def executar_notebook():
    print("Executando o notebook via nbconvert...")
    
    comando = [
        sys.executable, "-m", "jupyter", "nbconvert", 
        "--to", "notebook", 
        "--execute", "src/teste.ipynb", 
        "--inplace"  
    ]
    
    try:
        subprocess.run (comando, check=True)
        print("Notebook da rede neural executado com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o notebook: {e}")


def main():

    # passo 1 - baixar o dataser
    print("1. Baixando dataset...")
    #dd.download_minds_libras()

    # passo 2 - pré processamento dos vídeos
    print("\n2. Processando vídeos...")
    #ad.processar_dataset()

    # passo 3 - processamento da rede neural
    print("\n3. Processando rede neural...")
    executar_notebook()

    print("Pipeline concluído!")



if __name__ == "__main__":
    main()