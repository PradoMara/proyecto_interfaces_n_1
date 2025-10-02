import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funciones_comunes import (
    cargar_imagen,
    convertir_a_gris,
    guardar_imagen_color
)


def crear_coloracion_azul_oceano(img_gris: np.ndarray) -> np.ndarray:

    g = img_gris.astype(np.float32)

    # Factores de escala por canal (pueden ajustarse)
    f_r = 0.25   # rojo bajo evita magenta
    f_g = 0.55   # verde moderado controla cian
    f_b = 1.00   # azul pleno
    boost_b = 40 # refuerzo constante en azul (profundidad)

    r = g * f_r
    v = g * f_g
    b = g * f_b + boost_b

    img = np.stack([r, v, b], axis=-1)
    img = np.clip(img, 0, 255)

    # Espuma: intensidades muy altas se fuerzan a blanco (umbral ajustable)
    mask_espuma = g > 230
    if np.any(mask_espuma):
        blancos = np.stack([g[mask_espuma], g[mask_espuma], g[mask_espuma]], axis=-1)
        img[mask_espuma] = blancos

    return img.astype(np.uint8)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_imagen = os.path.join(script_dir, "sea.jpg")
    # Cargar la imagen
    img_original = cargar_imagen(ruta_imagen)
    
    # Convertir a escala de grises para aplicar colorización
    img_gris = convertir_a_gris(img_original)

    # Crear colorización azul océano (única)
    img_coloreada = crear_coloracion_azul_oceano(img_gris)
    # Guardar solo la imagen coloreada
    ruta_salida_img = os.path.join(script_dir, "oceano_coloreado.png")
    guardar_imagen_color(img_coloreada, ruta_salida_img)

if __name__ == "__main__":
    main()
