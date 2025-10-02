import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Agregar el directorio padre al path para importar funciones_comunes.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar funciones desde funciones_comunes.py
from funciones_comunes import (
    cargar_imagen,
    convertir_a_gris,
    aplicar_colormap,
    aplicar_coloracion_personalizada,
    guardar_imagen_color
)

from PIL import Image

def crear_coloracion_azul_oceano(img_gris: np.ndarray) -> np.ndarray:
    """
    Crea una colorización que mantenga los detalles del océano original.
    """
    # Usar colormap 'ocean' que da una progresión más natural azul-verde-cian
    img_coloreada = aplicar_colormap(img_gris, 'ocean')
    return img_coloreada

def crear_visualizacion_simple(img_original, img_coloreada, guardar_como: str = None):
    """
    Crea una visualización simple mostrando: original y coloreada (sin gris).
    """
    # Crear la figura con 2 imágenes solamente
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Colorización de Océano - Ejercicio 7', 
                 fontsize=16, fontweight='bold')
    
    # Imagen original
    if len(img_original.shape) == 3:
        axes[0].imshow(img_original)
    else:
        axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title('Imagen Original', fontweight='bold', fontsize=12)
    axes[0].axis('off')
    
    # Imagen coloreada
    axes[1].imshow(img_coloreada)
    axes[1].set_title('Figura Coloreada (Azul Océano)', fontweight='bold', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Guardar si se especifica
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada como: {guardar_como}")
    
    plt.show()

def mostrar_estadisticas_color(img_original, img_gris, img_coloreada_principal, nombre_coloracion):
    """
    Muestra estadísticas comparativas de las imágenes.
    """
    print("\n" + "="*70)
    print("ESTADÍSTICAS DE COLORIZACIÓN")
    print("="*70)
    
    if len(img_original.shape) == 3:
        print(f"Imagen original:     {img_original.shape[1]}x{img_original.shape[0]} píxeles, {img_original.shape[2]} canales")
    else:
        print(f"Imagen original:     {img_original.shape[1]}x{img_original.shape[0]} píxeles, 1 canal (gris)")
    
    print(f"Imagen en gris:      {img_gris.shape[1]}x{img_gris.shape[0]} píxeles, 1 canal")
    print(f"Imagen coloreada:    {img_coloreada_principal.shape[1]}x{img_coloreada_principal.shape[0]} píxeles, 3 canales")
    
    print(f"\nColorización principal aplicada: {nombre_coloracion}")
    
    # Estadísticas de la imagen en gris
    print(f"\nEscala de grises:")
    print(f"  Rango de intensidad: {img_gris.min()} - {img_gris.max()}")
    print(f"  Intensidad promedio: {img_gris.mean():.2f}")
    print(f"  Desviación estándar: {img_gris.std():.2f}")
    
    # Estadísticas de la imagen coloreada
    print(f"\nImagen coloreada ({nombre_coloracion}):")
    for i, canal in enumerate(['Rojo', 'Verde', 'Azul']):
        canal_data = img_coloreada_principal[:, :, i]
        print(f"  Canal {canal}: rango {canal_data.min()}-{canal_data.max()}, promedio {canal_data.mean():.2f}")

def main():
    """
    Función principal que ejecuta todo el procesamiento.
    """
    # Obtener el directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_imagen = os.path.join(script_dir, "sea.jpg")
    
    print("="*70)
    print("EJERCICIO 7 - COLORIZACIÓN DE OCÉANO")
    print("="*70)
    
    # Cargar la imagen
    print(f"Cargando imagen: {ruta_imagen}")
    img_original = cargar_imagen(ruta_imagen)
    print(f"Dimensiones originales: {img_original.shape}")
    
    # Convertir a escala de grises para aplicar colorización
    img_gris = convertir_a_gris(img_original)
    
    # Crear colorización azul océano (única)
    print("Generando colorización del océano...")
    img_coloreada = crear_coloracion_azul_oceano(img_gris)
    
    # Guardar solo la imagen coloreada
    ruta_salida_img = os.path.join(script_dir, "oceano_coloreado.png")
    guardar_imagen_color(img_coloreada, ruta_salida_img)
    print(f"Imagen coloreada guardada: {ruta_salida_img}")
    
    print("\n✅ Colorización completada")
    print(f"📁 Archivo generado: oceano_coloreado.png")

if __name__ == "__main__":
    main()
