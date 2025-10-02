import os
import sys
import numpy as np
from typing import Tuple, Dict

# Agregar el directorio padre al path para importar global.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar funciones desde funciones_comunes.py
from funciones_comunes import (
    cargar_imagen_color,
    extraer_figura_color,
    calcular_area,
    calcular_momento_crudo,
    calcular_centroide_por_pixeles,
    calcular_centroide_por_momentos,
    calcular_momento_central,
    calcular_momento_central_normalizado,
    calcular_momentos_hu,
    crear_imagen_con_centroide
)

# ------------------------------------------------------------
# Funciones para procesar cada figura
# ------------------------------------------------------------

def procesar_figura_a(ruta: str) -> Dict:
    """
    Procesa la Figura A (roja): área, centroide por píxeles, centroide por momentos.
    """
    img_rgb = cargar_imagen_color(ruta)
    mascara = extraer_figura_color(img_rgb)
    
    area = calcular_area(mascara)
    centroide_pixeles = calcular_centroide_por_pixeles(mascara)
    centroide_momentos = calcular_centroide_por_momentos(mascara)
    
    # Crear imagen con centroide marcado
    dir_entrada = os.path.dirname(ruta)
    ruta_salida = os.path.join(dir_entrada, "fig_a_centroid.png")
    crear_imagen_con_centroide(ruta, centroide_momentos[0], centroide_momentos[1], ruta_salida)
    
    return {
        "area": area,
        "centroide_por_pixeles": centroide_pixeles,
        "centroide_por_momentos": centroide_momentos,
        "imagen_marcada": ruta_salida
    }

def procesar_figura_b(ruta: str) -> Dict:
    """
    Procesa la Figura B (verde): momento crudo M_2,3, momento central μ_2,3, 
    momento central normalizado η_2,3.
    """
    img_rgb = cargar_imagen_color(ruta)
    mascara = extraer_figura_color(img_rgb)
    
    area = calcular_area(mascara)
    cx, cy = calcular_centroide_por_momentos(mascara)
    
    # Momento crudo M_2,3
    M_2_3 = calcular_momento_crudo(mascara, 2, 3)
    
    # Momento central μ_2,3
    mu_2_3 = calcular_momento_central(mascara, cx, cy, 2, 3)
    
    # Momento central normalizado η_2,3
    mu_00 = area  # μ_00 = área
    eta_2_3 = calcular_momento_central_normalizado(mu_2_3, mu_00, 2, 3)
    
    return {
        "area_con_hueco": area,
        "centroide": (cx, cy),
        "M_2_3": M_2_3,
        "mu_2_3": mu_2_3,
        "eta_2_3": eta_2_3
    }

def procesar_figura_c(ruta: str) -> Dict:
    """
    Procesa la Figura C (azul): momentos de Hu H1, H2, H3.
    """
    img_rgb = cargar_imagen_color(ruta)
    mascara = extraer_figura_color(img_rgb)
    
    H1, H2, H3 = calcular_momentos_hu(mascara)
    
    return {
        "H1": H1,
        "H2": H2,
        "H3": H3
    }

# ------------------------------------------------------------
# Ejecución principal
# ------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    ruta_a = os.path.join(script_dir, "a.png")
    ruta_b = os.path.join(script_dir, "b.png")
    ruta_c = os.path.join(script_dir, "c.png")

    print("\n")
    print("FIGURA A")
    print("="*50)
    res_a = procesar_figura_a(ruta_a)
    print(f"Area: {res_a['area']}")
    print(f"Centroide por pixeles: {res_a['centroide_por_pixeles']}")
    print(f"Centroide por momentos: {res_a['centroide_por_momentos']}")


    print("\n")
    print("FIGURA B")
    print("="*50)
    res_b = procesar_figura_b(ruta_b)
    print(f"Momento crudo M_2,3: {res_b['M_2_3']}")
    print(f"Momento central mu_2,3: {res_b['mu_2_3']}")
    print(f"Momento central normalizado eta_2,3: {res_b['eta_2_3']}")

    print("\n" )
    print("FIGURA C")
    print("="*50)
    res_c = procesar_figura_c(ruta_c)
    print(f"Momento de Hu H1: {res_c['H1']}")
    print(f"Momento de Hu H2: {res_c['H2']}")
    print(f"Momento de Hu H3: {res_c['H3']}")
