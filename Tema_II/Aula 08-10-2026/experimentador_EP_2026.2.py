import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit

# ==========================================
# 1. Função do Aluno (Mock)
# ==========================================
def segmenta_imagem(imagem_array, k):
    """
    Substitua pelo script real de segmentação do aluno.
    Recebe o array da imagem já em memória e o número de clusters K.
    """
    # Mock de processamento para testes (remova na versão final)
    # Simula um tempo de execução proporcional a O(n*K)
    n = imagem_array.shape[0] * imagem_array.shape[1]
    time.sleep((n * k) * 1e-8)
    pass

# ==========================================
# 2. Descoberta e Ordenação Ligeira (Sem estourar RAM)
# ==========================================
img_dir = "./minhas_imagens" # Substitua pelo diretório real
k_list = [2, 3, 4]
R = 30 # Número de repetições por combinação

print("Mapeando imagens e calculando 'n' (apenas cabeçalhos)...")
arquivos_n = []

# Varre os arquivos no diretório especificado
if os.path.exists(img_dir):
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            filepath = os.path.join(img_dir, filename)
            
            # Image.open lê apenas os metadados. Não aloca a matriz de pixels.
            with Image.open(filepath) as img:
                w, h = img.size
                n = w * h
                arquivos_n.append({'arquivo': filename, 'caminho': filepath, 'n': n})

# Ordena a lista de dicionários pelo valor de 'n' crescente
arquivos_n = sorted(arquivos_n, key=lambda x: x['n'])

print(f"{len(arquivos_n)} imagens ordenadas prontas para processamento.")

# ==========================================
# 3. Execução (Carregando sob demanda)
# ==========================================
resultados = []

for item in arquivos_n:
    n = item['n']
    filepath = item['caminho']
    nome_arquivo = item['arquivo']
    
    print(f"Carregando {nome_arquivo} para a RAM (n={n})...")
    
    # Carrega a imagem real para a RAM apenas no momento de usar
    with Image.open(filepath) as img:
        img_array = np.array(img)
    
    # Executa todas as repetições e Ks para esta imagem
    for k in k_list:
        for r in range(R):
            t0 = time.perf_counter()
            segmenta_imagem(img_array, k)  # A matriz é passada aqui
            t1 = time.perf_counter()
            
            resultados.append({
                'arquivo': nome_arquivo,
                'n': n, 
                'k': k, 
                'repeticao': r, 
                'tempo': (t1 - t0)
            })
            
    # Libera a RAM explicitamente antes de carregar a próxima imagem
    del img_array
    print(f"Análise de {nome_arquivo} concluída. Memória liberada.")

# Transforma os resultados em DataFrame
df_raw = pd.DataFrame(resultados)

# ==========================================
# 4. Pré-processamento e Validação F.O.C.A.
# ==========================================
if not df_raw.empty:
    df_filtered = pd.DataFrame()
    
    # Filtragem de Outliers (Percentis 5 e 95)
    for (n, k), group in df_raw.groupby(['n', 'k']):
        p05 = group['tempo'].quantile(0.05)
        p95 = group['tempo'].quantile(0.95)
        valid_mask = (group['tempo'] >= p05) & (group['tempo'] <= p95)
        df_filtered = pd.concat([df_filtered, group[valid_mask]])
    
    # Validação de Estabilidade (Média, Desvio e CV)
    resumo = df_filtered.groupby(['n', 'k']).agg(
        tempo_medio=('tempo', 'mean'),
        desvio=('tempo', 'std'),
        contagem_valida=('tempo', 'count')
    ).reset_index()
    
    resumo['CV'] = resumo['desvio'] / resumo['tempo_medio']
    resumo['estavel_CV_menor_15'] = resumo['CV'] < 0.15
    
    print("\n--- Resumo de Estabilidade ---")
    print(resumo)
    
    # ==========================================
    # 5. Ajuste de Curvas
    # ==========================================
    def model_n_k(X, alpha, beta, gamma, delta):
        n, k = X
        return alpha * (n * k) + beta * n + gamma * k + delta
    
    X_data = np.vstack((resumo['n'], resumo['k']))
    y_data = resumo['tempo_medio']
    
    # O p0 pode precisar de sintonia dependendo da escala de tempo real
    popt, _ = curve_fit(model_n_k, X_data, y_data, p0=[1e-6, 1e-6, 1e-6, 0.01], maxfev=10000)
    resumo['tempo_predito'] = model_n_k(X_data, *popt)
    
    # ==========================================
    # 6. Plotagens Bidimensionais
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot A: Tempo vs n
    colors_k = ['red', 'green', 'blue', 'orange', 'purple'] 
    for idx, k in enumerate(k_list):
        dados_k = resumo[resumo['k'] == k]
        if not dados_k.empty:
            c = colors_k[idx % len(colors_k)]
            ax1.plot(dados_k['n'], dados_k['tempo_medio'], marker='o', 
                     color=c, label=f'K={k} (Empírico)')
            ax1.plot(dados_k['n'], dados_k['tempo_predito'], linestyle='--', 
                     color=c, alpha=0.6)
    
    ax1.set_xlabel('Número de Pixels (n)')
    ax1.set_ylabel('Tempo Médio de Execução (s)')
    ax1.set_title('Tempo vs Tamanho da Imagem')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot B: Tempo vs K
    n_processados = sorted(resumo['n'].unique())
    colors_n = plt.cm.viridis(np.linspace(0, 1, len(n_processados)))
    
    for idx, n_val in enumerate(n_processados):
        dados_n = resumo[resumo['n'] == n_val]
        if not dados_n.empty:
            ax2.plot(dados_n['k'], dados_n['tempo_medio'], marker='s', 
                     color=colors_n[idx], label=f'n={n_val}')
            ax2.plot(dados_n['k'], dados_n['tempo_predito'], linestyle='--', 
                     color=colors_n[idx], alpha=0.6)
    
    ax2.set_xlabel('Número de Clusters (K)')
    ax2.set_title('Tempo vs Número de Clusters')
    ax2.set_xticks(k_list)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('relatorio_complexidade.png', dpi=300)
    plt.show()
    
    print("\nPipeline concluído. Gráficos gerados e salvos como 'relatorio_complexidade.png'.")
	
	# ==========================================
    # 7. Exportação dos Dados para Avaliação
    # ==========================================
    arquivo_csv = 'resultados_foca_aluno.csv'
    
    # Exporta os dados usando ponto e vírgula como separador e vírgula para decimais,
    # garantindo que o arquivo seja aberto corretamente no Excel em PT-BR.
    resumo.to_csv(arquivo_csv, index=False, float_format='%.6f', sep=';', decimal=',')
    
    print(f"Tabela de avaliação exportada com sucesso para: {arquivo_csv}")
	
else:
    print("\nNenhum dado processado. Verifique se o diretório contém imagens válidas.")
	
	