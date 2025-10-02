import os
import sys
import numpy as np
from PIL import Image

# Agregar directorio padre para importar funciones comunes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from funciones_comunes import (
    cargar_imagen_color,
    separar_planos_rgb,
    calcular_area_ocupada
)

def guardar_plano_png(canal: np.ndarray, ruta: str, color: str):

    h, w = canal.shape
    plano = np.zeros((h, w, 3), dtype=np.uint8)
    if color == 'r':
        plano[..., 0] = canal
    elif color == 'g':
        plano[..., 1] = canal
    elif color == 'b':
        plano[..., 2] = canal
    Image.fromarray(plano, 'RGB').save(ruta)


def imprimir_estadisticas_canales(stats_r, stats_g, stats_b, dimensiones):

    print(f"Total píxeles: {stats_r['area_total']:,}  (Imagen: {dimensiones[1]}x{dimensiones[0]}): \n")
    canales = [("ROJO", stats_r), ("VERDE", stats_g), ("AZUL", stats_b)]
    for nombre, st in canales:
        print(f"{nombre:<5} -> ocupados: {st['pixeles_ocupados']:,} ({st['porcentaje_ocupacion']:.2f}%)  |  vacíos: {st['pixeles_vacios']:,} ({st['porcentaje_vacio']:.2f}%)")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_imagen = os.path.join(script_dir, "fig_05.jpg")

    img_rgb = cargar_imagen_color(ruta_imagen)

    canal_r, canal_g, canal_b = separar_planos_rgb(img_rgb)

    stats_r = calcular_area_ocupada(canal_r)
    stats_g = calcular_area_ocupada(canal_g)
    stats_b = calcular_area_ocupada(canal_b)

    imprimir_estadisticas_canales(stats_r, stats_g, stats_b, img_rgb.shape)

    ruta_r = os.path.join(script_dir, "plano_rojo.png")
    ruta_g = os.path.join(script_dir, "plano_verde.png")
    ruta_b = os.path.join(script_dir, "plano_azul.png")
    guardar_plano_png(canal_r, ruta_r, 'r')
    guardar_plano_png(canal_g, ruta_g, 'g')
    guardar_plano_png(canal_b, ruta_b, 'b')

if __name__ == "__main__":
    main()