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

def crear_multiples_colorizaciones(img_gris: np.ndarray) -> dict:
    """
    Crea múltiples versiones coloreadas de la imagen usando diferentes técnicas.
    """
    colorizaciones = {}
    
    # Colormaps predefinidos
    colormaps = {
        'ocean': 'Océano (Azul)',
        'viridis': 'Viridis (Verde-Azul)',
        'plasma': 'Plasma (Púrpura-Rosa)',
        'inferno': 'Inferno (Negro-Rojo-Amarillo)',
        'coolwarm': 'Frío-Cálido (Azul-Rojo)',
        'seismic': 'Sísmico (Azul-Blanco-Rojo)'
    }
    
    for cmap_name, cmap_desc in colormaps.items():
        colorizaciones[cmap_desc] = aplicar_colormap(img_gris, cmap_name)
    
    # Colorizaciones personalizadas
    colores_personalizados = {
        'Azul Océano': (30, 144, 255),      # Azul océano
        'Verde Bosque': (34, 139, 34),      # Verde bosque
        'Dorado Atardecer': (255, 215, 0),  # Dorado
        'Púrpura Místico': (148, 0, 211),   # Púrpura
        'Rojo Atardecer': (255, 69, 0),     # Rojo-naranja
        'Turquesa Tropical': (64, 224, 208) # Turquesa
    }
    
    for nombre, color in colores_personalizados.items():
        colorizaciones[nombre] = aplicar_coloracion_personalizada(img_gris, color)
    
    return colorizaciones

def crear_visualizacion_completa(img_original, img_gris, colorizaciones, guardar_como: str = None):
    """
    Crea una visualización completa mostrando la imagen original, en gris y las versiones coloreadas.
    """
    # Calcular el número de imágenes y la disposición del grid
    num_colorizaciones = len(colorizaciones)
    num_cols = 4
    num_rows = (num_colorizaciones + 2 + num_cols - 1) // num_cols  # +2 para original y gris
    
    # Crear la figura
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Aplicación de Color a Imagen en Escala de Grises - Ejercicio 7', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Mostrar imagen original
    idx = 0
    row, col = idx // num_cols, idx % num_cols
    if len(img_original.shape) == 3:
        axes[row, col].imshow(img_original)
    else:
        axes[row, col].imshow(img_original, cmap='gray')
    axes[row, col].set_title('Imagen Original', fontweight='bold', fontsize=12)
    axes[row, col].axis('off')
    
    # Mostrar imagen en gris
    idx = 1
    row, col = idx // num_cols, idx % num_cols
    axes[row, col].imshow(img_gris, cmap='gray')
    axes[row, col].set_title('Escala de Grises', fontweight='bold', fontsize=12)
    axes[row, col].axis('off')
    
    # Mostrar colorizaciones
    for i, (nombre, img_coloreada) in enumerate(colorizaciones.items()):
        idx = i + 2
        row, col = idx // num_cols, idx % num_cols
        
        if row < num_rows and col < num_cols:
            axes[row, col].imshow(img_coloreada)
            axes[row, col].set_title(nombre, fontweight='bold', fontsize=10)
            axes[row, col].axis('off')
    
    # Ocultar ejes vacíos
    total_used = len(colorizaciones) + 2
    for idx in range(total_used, num_rows * num_cols):
        row, col = idx // num_cols, idx % num_cols
        axes[row, col].axis('off')
    
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
    print("EJERCICIO 7 - APLICACIÓN DE COLOR A IMAGEN EN GRIS")
    print("="*70)
    
    # Cargar la imagen
    print(f"Cargando imagen: {ruta_imagen}")
    img_original = cargar_imagen(ruta_imagen)
    print(f"Dimensiones originales: {img_original.shape}")
    
    # Convertir a escala de grises si es necesario
    print("Convirtiendo a escala de grises...")
    img_gris = convertir_a_gris(img_original)
    print(f"Dimensiones en gris: {img_gris.shape}")
    
    # Crear múltiples colorizaciones
    print("Generando múltiples colorizaciones...")
    colorizaciones = crear_multiples_colorizaciones(img_gris)
    print(f"Se generaron {len(colorizaciones)} colorizaciones diferentes")
    
    # Seleccionar la colorización principal (Azul Océano, similar al ejemplo)
    nombre_principal = 'Azul Océano'
    img_coloreada_principal = colorizaciones[nombre_principal]
    
    # Mostrar estadísticas
    mostrar_estadisticas_color(img_original, img_gris, img_coloreada_principal, nombre_principal)
    
    # Crear visualización completa
    print(f"\nGenerando visualización completa...")
    ruta_salida_viz = os.path.join(script_dir, "coloracion_completa_ejercicio7.png")
    
    crear_visualizacion_completa(
        img_original, img_gris, colorizaciones,
        guardar_como=ruta_salida_viz
    )
    
    # Guardar la imagen principal coloreada
    ruta_salida_principal = os.path.join(script_dir, "sea_coloreada_azul.png")
    guardar_imagen_color(img_coloreada_principal, ruta_salida_principal)
    print(f"Imagen coloreada guardada: {ruta_salida_principal}")
    
    # Guardar algunas colorizaciones adicionales
    colorizaciones_destacadas = ['Océano (Azul)', 'Verde Bosque', 'Dorado Atardecer']
    for nombre in colorizaciones_destacadas:
        if nombre in colorizaciones:
            nombre_archivo = nombre.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            ruta_archivo = os.path.join(script_dir, f"sea_{nombre_archivo}.png")
            guardar_imagen_color(colorizaciones[nombre], ruta_archivo)
            print(f"Imagen coloreada guardada: {ruta_archivo}")
    
    print("\n" + "="*70)
    print("RESUMEN DEL PROCESAMIENTO")
    print("="*70)
    print("✅ Imagen convertida a escala de grises")
    print(f"✅ {len(colorizaciones)} colorizaciones generadas")
    print("✅ Visualización completa creada")
    print("✅ Imágenes coloreadas guardadas")
    
    print(f"\n📁 Archivos generados:")
    print(f"  📊 Visualización: coloracion_completa_ejercicio7.png")
    print(f"  🖼️  Imagen principal: sea_coloreada_azul.png")
    print(f"  🎨 Colorizaciones adicionales: sea_*.png")
    
    print(f"\n🎯 Colorización destacada: {nombre_principal}")
    print("   Simula el efecto del océano azul mostrado en el ejemplo")

if __name__ == "__main__":
    main()
