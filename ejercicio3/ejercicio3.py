import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Agregar el directorio padre al path para importar funciones_comunes.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar funciones desde funciones_comunes.py
from funciones_comunes import (
    cargar_imagen_color,
    separar_planos_color,
    convertir_a_gris,
    analizar_estadisticas_canales,
    guardar_imagen_gris
)

from PIL import Image

def graficar_solo_histogramas_rgb(canal_r, canal_g, canal_b, guardar_como: str = None):
    """
    Crea una visualización con matplotlib mostrando SOLO:
    - Los histogramas de los 3 planos de color (R, G, B)
    SIN mostrar las imágenes de los planos
    """
    # Configurar la figura con subplots en 1 fila, 3 columnas (solo histogramas)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Histogramas de los Planos RGB', 
                 fontsize=16, fontweight='bold')
    
    # Histograma del canal Rojo
    axes[0].hist(canal_r.flatten(), bins=50, color='red', alpha=0.7, density=True)
    axes[0].set_title('Histograma Canal Rojo', fontweight='bold')
    axes[0].set_xlabel('Intensidad')
    axes[0].set_ylabel('Densidad')
    axes[0].grid(True, alpha=0.3)
    
    # Histograma del canal Verde
    axes[1].hist(canal_g.flatten(), bins=50, color='green', alpha=0.7, density=True)
    axes[1].set_title('Histograma Canal Verde', fontweight='bold')
    axes[1].set_xlabel('Intensidad')
    axes[1].set_ylabel('Densidad')
    axes[1].grid(True, alpha=0.3)
    
        # Histograma del canal Azul
    axes[2].hist(canal_b.flatten(), bins=50, color='blue', alpha=0.7, density=True)
    axes[2].set_title('Histograma Canal Azul', fontweight='bold')
    axes[2].set_xlabel('Intensidad')
    axes[2].set_ylabel('Densidad')
    axes[2].grid(True, alpha=0.3)
    
    # Ajustar el layout
    plt.tight_layout()
    
    # Guardar si se especifica una ruta
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado como: {guardar_como}")
    
    # Mostrar el gráfico
    plt.show()

def main():
    """
    Función principal que ejecuta todo el procesamiento.
    """
    # Obtener el directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_imagen = os.path.join(script_dir, "flores.png")
    
    print("="*60)
    print("EJERCICIO 3 - SEPARACIÓN DE PLANOS DE COLOR")
    print("="*60)
    
    # Cargar la imagen
    print(f"Cargando imagen: {ruta_imagen}")
    img_rgb = cargar_imagen_color(ruta_imagen)
    print(f"Dimensiones de la imagen: {img_rgb.shape}")
    
    # Separar planos de color
    print("Separando planos de color RGB...")
    canal_r, canal_g, canal_b = separar_planos_color(img_rgb)
    
    # Convertir a escala de grises
    print("Convirtiendo a escala de grises...")
    img_gris = convertir_a_gris(img_rgb)
    
    # Mostrar estadísticas
    analizar_estadisticas_canales(canal_r, canal_g, canal_b, img_gris)
    
    # Guardar imagen en escala de grises
    ruta_gris = os.path.join(script_dir, "flores_grises.png")
    guardar_imagen_gris(img_gris, ruta_gris)
    print(f"Imagen en gris guardada como: {ruta_gris}")
    
    # Crear y mostrar solo los histogramas de los planos RGB
    graficar_solo_histogramas_rgb(
        canal_r, canal_g, canal_b
    )

if __name__ == "__main__":
    main()
