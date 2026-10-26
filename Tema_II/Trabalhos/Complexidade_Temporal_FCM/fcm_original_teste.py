import numpy as np
from PIL import Image
from FCM_PURO_v3 import fcm_processar_imagem

def segmenta_imagem(imagem_array, k):
    imagem = Image.fromarray(
        np.asarray(imagem_array, dtype=np.uint8)
    ).convert("RGB")

    _, _, imagem_segmentada = fcm_processar_imagem(
        img=imagem,
        K=k,
        m=2.0,
        max_iter=100,
        epsilon=0.01
    )

    return np.array(imagem_segmentada)