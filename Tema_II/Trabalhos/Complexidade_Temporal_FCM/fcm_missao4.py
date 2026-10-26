
import random
import numpy as np



def fcm_missao4(X, K, m=2.0, max_iter=100, epsilon=0.01):
    """
    FCM baseado no baseline oficial do professor.
    Missão 4: Loop Unrolling dos canais RGB.

    X: lista de pixels, cada um representado por [R, G, B].
    K: número de clusters.
    """

    N = len(X)
    canais = len(X[0])

    # FASE 1: INICIALIZAÇÃO DA MATRIZ DE PERTINÊNCIA U
    U = []

    for i in range(N):
        linha = []
        soma_linha = 0.0

        for j in range(K):
            val = random.random()
            linha.append(val)
            soma_linha += val

        for j in range(K):
            linha[j] = linha[j] / soma_linha

        U.append(linha)

    # LAÇO PRINCIPAL DO FCM
    for iteracao in range(max_iter):

        # FASE 2: ATUALIZAÇÃO DOS CENTROIDES
        C = []

        for j in range(K):
            numerador = [0.0] * canais
            denominador = 0.0

            for i in range(N):
                u_m = U[i][j] ** m
                denominador += u_m

                # MISSÃO 4: LOOP UNROLLING
                # Substitui o laço dos canais RGB por operações explícitas.
                numerador[0] += u_m * X[i][0]
                numerador[1] += u_m * X[i][1]
                numerador[2] += u_m * X[i][2]

            # MISSÃO 4: construção explícita do centróide RGB.
            centroide_j = [
                numerador[0] / denominador,
                numerador[1] / denominador,
                numerador[2] / denominador
            ]

            C.append(centroide_j)

        # FASE 3 E 4: DISTÂNCIAS, PERTINÊNCIAS E CONVERGÊNCIA
        U_nova = []
        max_diff = 0.0

        for i in range(N):
            linha_nova = [0.0] * K
            distancias = [0.0] * K

            for j in range(K):
                soma_quadrados = 0.0

                # MISSÃO 4: LOOP UNROLLING DAS DISTÂNCIAS RGB.
                diff_r = X[i][0] - C[j][0]
                soma_quadrados += diff_r * diff_r

                diff_g = X[i][1] - C[j][1]
                soma_quadrados += diff_g * diff_g

                diff_b = X[i][2] - C[j][2]
                soma_quadrados += diff_b * diff_b

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
                    razao = distancias[j] / distancias[k]
                    soma_dist += razao ** (2.0 / (m - 1.0))

                pertinencia = 1.0 / soma_dist
                linha_nova[j] = pertinencia

                diff_u = pertinencia - U[i][j]

                if diff_u < 0:
                    diff_u = -diff_u

                if diff_u > max_diff:
                    max_diff = diff_u

            U_nova.append(linha_nova)

        # Atualização da matriz de pertinências.
        U = U_nova

        if max_diff < epsilon:
            break

    return C, U


def segmenta_imagem(imagem_array, k):
    """
    Adaptador para o experimentador.py.

    Recebe uma imagem NumPy no formato H x W x 3
    e devolve a imagem segmentada no mesmo formato.
    """

    # Garante três canais RGB.
    if imagem_array.ndim == 2:
        imagem_array = np.stack(
            [imagem_array] * 3,
            axis=-1
        )
    elif imagem_array.shape[2] == 4:
        imagem_array = imagem_array[:, :, :3]

    img = np.asarray(imagem_array, dtype=np.uint8)

    H, W, C = img.shape

    # Converte a imagem em uma lista de pixels RGB.
    X = img.reshape(-1, 3).astype(float).tolist()

    # Executa o FCM da Missão 4.
    _, U = fcm_missao4(
        X,
        K=k,
        m=2.0,
        max_iter=100,
        epsilon=0.01
    )

    # Paleta de cores utilizada na visualização.
    cores = np.array([
        [255, 165, 0],    # Laranja
        [255, 105, 180],  # Rosa
        [0, 0, 0],        # Preto
        [255, 0, 0],      # Vermelho
        [0, 0, 255],      # Azul
        [255, 255, 255]   # Branco
    ], dtype=np.uint8)

    # Atribui cada pixel ao cluster de maior pertinência.
    clusters = np.argmax(np.asarray(U), axis=1)

    imagem_segmentada = cores[
        clusters % len(cores)
    ].reshape(H, W, 3)

    return imagem_segmentada
