import os

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

from scipy.optimize import curve_fit



# ==========================================
# 1. Função de segmentação K-Means do aluno
# ==========================================
from fcm_missao4 import segmenta_imagem



# Define a semente para reproduzir os mesmos resultados aleatórios.
np.random.seed(42)



# ==========================================
# 2. Descoberta e ordenação das imagens
# ==========================================



img_dir = "./imagens"

k_list = [2, 3, 4]

R = 30  # Número de repetições por combinação

pasta_resultados = "./imagens_segmentadas"
os.makedirs(pasta_resultados, exist_ok=True)



print(
    "Mapeando imagens e calculando n "
    "(apenas cabeçalhos)..."
)



arquivos_n = []



# Verifica se o diretório existe.
if os.path.exists(img_dir):



    for filename in os.listdir(img_dir):



        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".webp")
        ):



            filepath = os.path.join(
                img_dir,
                filename
            )



            # Lê as dimensões sem carregar todos os pixels.
            with Image.open(filepath) as img:



                w, h = img.size



                n = w * h



                arquivos_n.append({
                    "arquivo": filename,
                    "caminho": filepath,
                    "n": n
                })



else:
    print(
        f"Diretório não encontrado: {img_dir}"
    )





# Ordena as imagens pelo número de pixels.
arquivos_n = sorted(
    arquivos_n,
    key=lambda x: x["n"]
)



print(
    f"{len(arquivos_n)} imagens ordenadas "
    "prontas para processamento."
)





# ==========================================
# 3. Execução dos experimentos
# ==========================================



resultados = []



for item in arquivos_n:



    n = item["n"]
    filepath = item["caminho"]
    nome_arquivo = item["arquivo"]



    print(
        f"\nCarregando {nome_arquivo} "
        f"para a memória (n={n})..."
    )



    # Carrega a imagem no momento do processamento.
    # RGB garante três canais de cor.
    with Image.open(filepath) as img:



        img_array = np.array(
            img.convert("RGB")
        )



    # Executa todos os valores de K.
    for k in k_list:



        for r in range(R):



            t0 = time.perf_counter()



            # Chamada da função real de segmentação.
            imagem_segmentada = segmenta_imagem(
                img_array,
                k
            )



            t1 = time.perf_counter()

            if r == 0:
                nome_base = os.path.splitext(nome_arquivo)[0]
                caminho_imagem = os.path.join(
                    pasta_resultados,
                    f"{nome_base}_K{k}.png"
                )
                Image.fromarray(
                    np.asarray(imagem_segmentada, dtype=np.uint8)
                ).save(caminho_imagem)



            tempo_execucao = t1 - t0



            resultados.append({
                "arquivo": nome_arquivo,
                "n": n,
                "k": k,
                "repeticao": r,
                "tempo": tempo_execucao
            })



        print(
            f"K={k}: {R} repetições concluídas."
        )



    # Libera a memória antes de carregar a próxima imagem.
    del img_array
    del imagem_segmentada



    print(
        f"Análise de {nome_arquivo} concluída. "
        "Memória liberada."
    )





# ==========================================
# 4. Pré-processamento e validação F.O.C.A.
# ==========================================



df_raw = pd.DataFrame(resultados)



if not df_raw.empty:



    # Filtragem de outliers pelos percentis 5 e 95.
    grupos_filtrados = []



    for (n, k), grupo in df_raw.groupby(["n", "k"]):



        p05 = grupo["tempo"].quantile(0.05)
        p95 = grupo["tempo"].quantile(0.95)



        mascara_valida = (
            (grupo["tempo"] >= p05)
            & (grupo["tempo"] <= p95)
        )



        grupos_filtrados.append(
            grupo[mascara_valida]
        )



    df_filtered = pd.concat(
        grupos_filtrados,
        ignore_index=True
    )



    # Calcula as estatísticas de cada combinação n e K.
    resumo = df_filtered.groupby(
        ["n", "k"]
    ).agg(
        tempo_medio=("tempo", "mean"),
        desvio=("tempo", "std"),
        contagem_valida=("tempo", "count")
    ).reset_index()



    # Coeficiente de variação.
    resumo["CV"] = (
        resumo["desvio"]
        / resumo["tempo_medio"]
    )



    resumo["estavel_CV_menor_15"] = (
        resumo["CV"] < 0.15
    )



    print("\n--- Resumo de Estabilidade ---")
    print(resumo)



    # ==========================================
    # 5. Ajuste de curvas
    # ==========================================



    def model_n_k(X, alpha, beta, gamma, delta):



        n, k = X



        return (
            alpha * (n * k)
            + beta * n
            + gamma * k
            + delta
        )



    X_data = np.vstack((
        resumo["n"].to_numpy(dtype=float),
        resumo["k"].to_numpy(dtype=float)
    ))



    y_data = resumo["tempo_medio"].to_numpy(
        dtype=float
    )



    # A curva precisa de pelo menos quatro pontos
    # para estimar os quatro parâmetros.
    if len(resumo) >= 4:



        try:



            popt, _ = curve_fit(
                model_n_k,
                X_data,
                y_data,
                p0=[1e-6, 1e-6, 1e-6, 0.01],
                maxfev=10000
            )



            resumo["tempo_predito"] = model_n_k(
                X_data,
                *popt
            )



            ajuste_realizado = True



        except (RuntimeError, ValueError) as erro:



            print(
                f"Não foi possível ajustar a curva: {erro}"
            )



            resumo["tempo_predito"] = np.nan



            ajuste_realizado = False



    else:



        print(
            "Não há pontos suficientes para ajustar "
            "a curva com quatro parâmetros."
        )



        resumo["tempo_predito"] = np.nan



        ajuste_realizado = False



    # ==========================================
    # 6. Gráficos bidimensionais
    # ==========================================



    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )



    colors_k = [
        "red",
        "green",
        "blue",
        "orange",
        "purple"
    ]



    # ------------------------------------------
    # Gráfico A: Tempo versus número de pixels
    # ------------------------------------------



    for idx, k in enumerate(k_list):



        dados_k = resumo[
            resumo["k"] == k
        ].sort_values("n")



        if not dados_k.empty:



            cor = colors_k[
                idx % len(colors_k)
            ]



            ax1.plot(
                dados_k["n"],
                dados_k["tempo_medio"],
                marker="o",
                color=cor,
                label=f"K={k} (Empírico)"
            )



            if ajuste_realizado:



                ax1.plot(
                    dados_k["n"],
                    dados_k["tempo_predito"],
                    linestyle="--",
                    color=cor,
                    alpha=0.6
                )



    ax1.set_xlabel(
        "Número de Pixels (n)"
    )



    ax1.set_ylabel(
        "Tempo Médio de Execução (s)"
    )



    ax1.set_title(
        "Tempo vs Tamanho da Imagem"
    )



    ax1.legend()
    ax1.grid(True, alpha=0.3)



    # ------------------------------------------
    # Gráfico B: Tempo versus número de clusters
    # ------------------------------------------



    n_processados = sorted(
        resumo["n"].unique()
    )



    colors_n = plt.cm.viridis(
        np.linspace(0, 1, len(n_processados))
    )



    for idx, n_val in enumerate(n_processados):



        dados_n = resumo[
            resumo["n"] == n_val
        ].sort_values("k")



        if not dados_n.empty:



            ax2.plot(
                dados_n["k"],
                dados_n["tempo_medio"],
                marker="s",
                color=colors_n[idx],
                label=f"n={n_val}"
            )



            if ajuste_realizado:



                ax2.plot(
                    dados_n["k"],
                    dados_n["tempo_predito"],
                    linestyle="--",
                    color=colors_n[idx],
                    alpha=0.6
                )



    ax2.set_xlabel(
        "Número de Clusters (K)"
    )



    ax2.set_ylabel(
        "Tempo Médio de Execução (s)"
    )



    ax2.set_title(
        "Tempo vs Número de Clusters"
    )



    ax2.set_xticks(k_list)
    ax2.legend()
    ax2.grid(True, alpha=0.3)



    plt.tight_layout()



    plt.savefig(
        "relatorio_complexidade.png",
        dpi=300
    )

    plt.close(fig)



    print(
        "\nPipeline concluído. Gráficos gerados e "
        "salvos como 'relatorio_complexidade.png'."
    )



    # ==========================================
    # 7. Exportação dos dados
    # ==========================================



    arquivo_csv = "resultados_foca_aluno.csv"



    resumo.to_csv(
        arquivo_csv,
        index=False,
        float_format="%.6f",
        sep=";",
        decimal=","
    )



    print(
        "\nTabela de avaliação exportada com sucesso para: "
        f"{arquivo_csv}"
    )



else:



    print(
        "\nNenhum dado processado. Verifique se o "
        "diretório contém imagens válidas."
    )







def exibir_equacao_ajustada(popt, X_data, y_data):
    """
    Exibe no terminal a equação ajustada com notação científica legível,
    os coeficientes individuais e o coeficiente de determinação (R²).
    """
    alpha, beta, gamma, delta = popt



    # Cálculo do R² (Coeficiente de Determinação)
    y_pred = model_n_k(X_data, *popt)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0



    print("\n" + "="*60)
    print("   MODELO EMPÍRICO AJUSTADO: T(n, K)")
    print("="*60)
    print("Formulação Geral:")
    print("  T(n, K) = α·(n·K) + β·n + γ·K + δ\n")
    print("Parâmetros Identificados:")
    print(f"  α (alpha) = {alpha:14.6e} s/(pixel·cluster)")
    print(f"  β (beta)  = {beta:14.6e} s/pixel")
    print(f"  γ (gamma) = {gamma:14.6e} s/cluster")
    print(f"  δ (delta) = {delta:14.6e} s (overhead base)")
    print("-" * 60)
    print("Equação Parametrizada:")
    print(f"  T(n, K) = ({alpha:.4e})·(n·K) + ({beta:.4e})·n + ({gamma:.4e})·K + ({delta:.4e})")
    print(f"Aderência Estatística (R²): {r2:.4f}")
    print("="*60 + "\n")



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



# ---> CHAMADA DA NOVA FUNÇÃO AQUI <---
exibir_equacao_ajustada(popt, X_data, y_data)



# ==========================================
# 6.5. Plotagem da Superfície 3D (Tempo vs n, k)
# ==========================================
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')



# Cria uma malha contínua para a superfície teórica baseada nos limites reais
n_linspace = np.linspace(resumo['n'].min(), resumo['n'].max(), 30)
k_linspace = np.linspace(resumo['k'].min(), resumo['k'].max(), 30)
N_mesh, K_mesh = np.meshgrid(n_linspace, k_linspace)



# Aplica a equação ajustada (popt) em todos os pontos da malha
T_mesh = model_n_k((N_mesh, K_mesh), *popt)



# Desenha a superfície teórica (Translúcida)
surf = ax_3d.plot_surface(N_mesh, K_mesh, T_mesh, cmap='viridis', alpha=0.6,
                              linewidth=0.1, antialiased=True)
fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10, pad=0.1, label='Tempo (s)')



# Sobrepõe os pontos empíricos medidos na bancada
ax_3d.scatter(resumo['n'], resumo['k'], resumo['tempo_medio'],
              color='red', edgecolor='black', s=60, label='Pontos Empíricos', depthshade=False)



# Configuração dos eixos e ângulos
ax_3d.set_xlabel('Tamanho da Imagem (n pixels)')
ax_3d.set_ylabel('Número de Clusters (K)')
ax_3d.set_zlabel('Tempo de Execução (s)')
ax_3d.set_title('Superfície F.O.C.A: Tempo(n, K)')
ax_3d.view_init(elev=25, azim=135) # Ajuste o ângulo de visão conforme necessário
ax_3d.legend()



plt.tight_layout()
plt.savefig('superficie_3d_foca.png', dpi=300)
plt.close(fig_3d)