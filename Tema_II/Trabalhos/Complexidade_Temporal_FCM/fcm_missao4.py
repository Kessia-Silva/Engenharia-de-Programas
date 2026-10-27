import random
from PIL import Image


def fcm_processar_imagem(img, K, m=2.0, max_iter=100, epsilon=0.01):
    """
    Executa o pipeline completo do FCM com Loop Unrolling nos canais RGB.
    """
    largura, altura = img.size

    # Extração segura dos pixels da imagem
    pixels_2d = list(img.getdata())

    X = []
    for p in pixels_2d:
        X.append([float(p[0]), float(p[1]), float(p[2])])

    N = len(X)
    canais = 3

    paleta_cores = [
        [255, 0, 0],      # Cluster 0: Vermelho
        [0, 255, 0],      # Cluster 1: Verde
        [0, 0, 255],      # Cluster 2: Azul
        [255, 255, 0],    # Cluster 3: Amarelo
        [255, 0, 255],    # Cluster 4: Magenta
        [0, 255, 255]     # Cluster 5: Ciano
    ]

    # Inicialização da matriz de pertinências U
    U = []
    for i in range(N):
        linha = []
        soma_linha = 0.0

        for j in range(K):
            val = random.random()
            linha.append(val)
            soma_linha += val

        for j in range(K):
            if soma_linha == 0.0:
                linha[j] = 1.0 / K
            else:
                linha[j] = linha[j] / soma_linha

        U.append(linha)

    # Laço principal do FCM
    iteracoes_realizadas = 0

    for iteracao in range(max_iter):
        iteracoes_realizadas = iteracao + 1

        C = []

        for j in range(K):
            numerador = [0.0] * canais
            denominador = 0.0

            for i in range(N):
                u_m = U[i][j] ** m
                denominador += u_m

                # MISSÃO 4: Loop Unrolling na acumulação dos canais RGB
                numerador[0] += u_m * X[i][0]
                numerador[1] += u_m * X[i][1]
                numerador[2] += u_m * X[i][2]

            centroide_j = []

            for c in range(canais):
                if denominador == 0.0:
                    centroide_j.append(0.0)
                else:
                    centroide_j.append(numerador[c] / denominador)

            C.append(centroide_j)

        U_nova = []
        max_diff = 0.0

        for i in range(N):
            linha_nova = [0.0] * K
            distancias = [0.0] * K

            for j in range(K):
                # MISSÃO 4: Loop Unrolling no cálculo da distância RGB
                diff_r = X[i][0] - C[j][0]
                diff_g = X[i][1] - C[j][1]
                diff_b = X[i][2] - C[j][2]

                soma_quadrados = (
                    diff_r * diff_r
                    + diff_g * diff_g
                    + diff_b * diff_b
                )

                distancias[j] = soma_quadrados ** 0.5

            for j in range(K):
                if distancias[j] == 0.0:
                    linha_nova[j] = 1.0

                    for k in range(K):
                        if k != j:
                            linha_nova[k] = 0.0

                    break

                soma_dist = 0.0

                for k in range(K):
                    d_k = distancias[k] if distancias[k] > 0.0 else 1e-10
                    razao = distancias[j] / d_k
                    soma_dist += razao ** (2.0 / (m - 1.0))

                pertinencia = (
                    1.0 / soma_dist if soma_dist > 0.0 else (1.0 / K)
                )
                linha_nova[j] = pertinencia

                diff_u = pertinencia - U[i][j]

                if diff_u < 0:
                    diff_u = -diff_u

                if diff_u > max_diff:
                    max_diff = diff_u

            U_nova.append(linha_nova)

        # Atualização da matriz de pertinências
        U = U_nova

        if max_diff < epsilon:
            print(f"-> Convergiu na iteração {iteracao + 1}!")
            break

    # Registra a quantidade de iterações da execução
    fcm_processar_imagem.ultima_iteracao = iteracoes_realizadas

    # Colorização por pertinência máxima
    pixels_segmentados_lista = []

    for i in range(N):
        maior_pertinencia = -1.0
        cluster_vencedor = 0

        for j in range(K):
            if U[i][j] > maior_pertinencia:
                maior_pertinencia = U[i][j]
                cluster_vencedor = j

        cor = paleta_cores[cluster_vencedor % len(paleta_cores)]
        pixels_segmentados_lista.append((cor[0], cor[1], cor[2]))

    # Reconstrução da imagem segmentada
    imagem_final = Image.new("RGB", (largura, altura))
    imagem_final.putdata(pixels_segmentados_lista)

    return C, U, imagem_final


def segmenta_imagem(imagem_array, k):
    """
    Adaptador para o experimentador da Missão 4.
    Recebe uma imagem como array e retorna a imagem segmentada.
    """
    imagem = Image.fromarray(imagem_array).convert("RGB")

    _, _, imagem_segmentada = fcm_processar_imagem(
        img=imagem,
        K=k,
        m=2.0,
        max_iter=100,
        epsilon=0.001
    )

    return imagem_segmentada