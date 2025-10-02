import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Agregar el directorio padre al path para importar funciones_comunes.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar funciones desde funciones_comunes.py
from funciones_comunes import (
    cargar_imagen_color,
    separar_planos_rgb,
    calcular_area_ocupada
)

from PIL import Image

def crear_visualizacion_completa(img_original, canal_r, canal_g, canal_b, 
                                stats_r, stats_g, stats_b, guardar_como: str = None):
    """
    Crea una visualización completa mostrando:
    - Imagen original
    - Plano Rojo, Verde y Azul
    - Estadísticas de área ocupada
    """
    # Configurar la figura
    fig = plt.figure(figsize=(16, 12))
    
    # Título principal
    fig.suptitle('Separación de Planos RGB y Análisis de Área Ocupada', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Crear grid de subplots
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
    
    # Imagen original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_original)
    ax1.set_title('Imagen Original', fontweight='bold', fontsize=12)
    ax1.axis('off')
    
    # Plano Rojo
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(canal_r, cmap='Reds', vmin=0, vmax=255)
    ax2.set_title(f'Plano Rojo\nÁrea ocupada: {stats_r["porcentaje_ocupacion"]:.1f}%', 
                  fontweight='bold', fontsize=12)
    ax2.axis('off')
    
    # Plano Verde
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(canal_g, cmap='Greens', vmin=0, vmax=255)
    ax3.set_title(f'Plano Verde\nÁrea ocupada: {stats_g["porcentaje_ocupacion"]:.1f}%', 
                  fontweight='bold', fontsize=12)
    ax3.axis('off')
    
    # Plano Azul
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(canal_b, cmap='Blues', vmin=0, vmax=255)
    ax4.set_title(f'Plano Azul\nÁrea ocupada: {stats_b["porcentaje_ocupacion"]:.1f}%', 
                  fontweight='bold', fontsize=12)
    ax4.axis('off')
    
    # Histogramas de cada canal
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(canal_r.flatten(), bins=50, color='red', alpha=0.7, density=True)
    ax5.set_title('Histograma Canal Rojo', fontweight='bold')
    ax5.set_xlabel('Intensidad')
    ax5.set_ylabel('Densidad')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(canal_g.flatten(), bins=50, color='green', alpha=0.7, density=True)
    ax6.set_title('Histograma Canal Verde', fontweight='bold')
    ax6.set_xlabel('Intensidad')
    ax6.set_ylabel('Densidad')
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.hist(canal_b.flatten(), bins=50, color='blue', alpha=0.7, density=True)
    ax7.set_title('Histograma Canal Azul', fontweight='bold')
    ax7.set_xlabel('Intensidad')
    ax7.set_ylabel('Densidad')
    ax7.grid(True, alpha=0.3)
    
    # Gráfico de barras comparativo de áreas ocupadas
    ax8 = fig.add_subplot(gs[1, 0])
    canales = ['Rojo', 'Verde', 'Azul']
    porcentajes = [stats_r['porcentaje_ocupacion'], 
                   stats_g['porcentaje_ocupacion'], 
                   stats_b['porcentaje_ocupacion']]
    colores = ['red', 'green', 'blue']
    
    bars = ax8.bar(canales, porcentajes, color=colores, alpha=0.7)
    ax8.set_title('Comparación de Área Ocupada', fontweight='bold')
    ax8.set_ylabel('Porcentaje (%)')
    ax8.set_ylim(0, 100)
    ax8.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar, porcentaje in zip(bars, porcentajes):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{porcentaje:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Tabla de estadísticas detalladas
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis('off')
    
    # Datos para la tabla
    tabla_datos = [
        ['Canal', 'Píxeles Ocupados', 'Área Ocupada (%)', 'Intensidad Min', 'Intensidad Max', 'Intensidad Promedio'],
        ['Rojo', f"{stats_r['pixeles_ocupados']:,}", f"{stats_r['porcentaje_ocupacion']:.2f}%", 
         stats_r['valor_minimo'], stats_r['valor_maximo'], f"{stats_r['valor_promedio']:.1f}"],
        ['Verde', f"{stats_g['pixeles_ocupados']:,}", f"{stats_g['porcentaje_ocupacion']:.2f}%", 
         stats_g['valor_minimo'], stats_g['valor_maximo'], f"{stats_g['valor_promedio']:.1f}"],
        ['Azul', f"{stats_b['pixeles_ocupados']:,}", f"{stats_b['porcentaje_ocupacion']:.2f}%", 
         stats_b['valor_minimo'], stats_b['valor_maximo'], f"{stats_b['valor_promedio']:.1f}"]
    ]
    
    tabla = ax9.table(cellText=tabla_datos[1:], colLabels=tabla_datos[0], 
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 2)
    
    # Colorear las filas de la tabla
    colores_tabla = ['lightcoral', 'lightgreen', 'lightblue']
    for i, color in enumerate(colores_tabla):
        for j in range(len(tabla_datos[0])):
            tabla[(i+1, j)].set_facecolor(color)
            tabla[(i+1, j)].set_alpha(0.3)
    
    # Guardar si se especifica
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada como: {guardar_como}")
    
    plt.show()

def imprimir_estadisticas_detalladas(stats_r, stats_g, stats_b, dimensiones):
    """
    Imprime estadísticas detalladas de cada canal.
    """
    print("\n" + "="*70)
    print("ANÁLISIS DETALLADO DE ÁREA OCUPADA POR CANAL")
    print("="*70)
    
    print(f"Dimensiones de la imagen: {dimensiones[1]} x {dimensiones[0]} píxeles")
    print(f"Área total: {stats_r['area_total']:,} píxeles")
    
    canales_datos = [
        ("CANAL ROJO", stats_r, "🔴"),
        ("CANAL VERDE", stats_g, "🟢"), 
        ("CANAL AZUL", stats_b, "🔵")
    ]
    
    for nombre, stats, emoji in canales_datos:
        print(f"\n{emoji} {nombre}:")
        print(f"  Píxeles ocupados (>0):     {stats['pixeles_ocupados']:>8,} ({stats['porcentaje_ocupacion']:>5.2f}%)")
        print(f"  Píxeles vacíos (=0):       {stats['pixeles_vacios']:>8,} ({stats['porcentaje_vacio']:>5.2f}%)")
        print(f"  Rango de intensidad:       {stats['valor_minimo']:>3d} - {stats['valor_maximo']:>3d}")
        print(f"  Intensidad promedio:       {stats['valor_promedio']:>8.2f}")
        if stats['pixeles_ocupados'] > 0:
            print(f"  Intensidad promedio (ocupados): {stats['intensidad_promedio_ocupados']:>6.2f}")

def main():
    """
    Función principal que ejecuta todo el procesamiento.
    """
    # Obtener el directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_imagen = os.path.join(script_dir, "fig_05.jpg")
    
    print("="*70)
    print("EJERCICIO 5 - SEPARACIÓN DE PLANOS RGB Y ANÁLISIS DE ÁREA")
    print("="*70)
    
    # Cargar la imagen
    print(f"Cargando imagen: {ruta_imagen}")
    img_rgb = cargar_imagen_color(ruta_imagen)
    print(f"Dimensiones: {img_rgb.shape}")
    
    # Separar planos RGB
    print("Separando planos de color RGB...")
    canal_r, canal_g, canal_b = separar_planos_rgb(img_rgb)
    
    # Calcular área ocupada para cada canal
    print("Calculando área ocupada para cada canal...")
    stats_r = calcular_area_ocupada(canal_r)
    stats_g = calcular_area_ocupada(canal_g)
    stats_b = calcular_area_ocupada(canal_b)
    
    # Mostrar estadísticas detalladas
    imprimir_estadisticas_detalladas(stats_r, stats_g, stats_b, img_rgb.shape)
    
    # Crear visualización completa
    print(f"\nGenerando visualización completa...")
    ruta_salida = os.path.join(script_dir, "separacion_planos_rgb_ejercicio5.png")
    
    crear_visualizacion_completa(
        img_rgb, canal_r, canal_g, canal_b,
        stats_r, stats_g, stats_b,
        guardar_como=ruta_salida
    )
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DEL ANÁLISIS")
    print("="*70)
    print(f"Canal con mayor área ocupada: ", end="")
    max_area = max(stats_r['porcentaje_ocupacion'], stats_g['porcentaje_ocupacion'], stats_b['porcentaje_ocupacion'])
    if stats_r['porcentaje_ocupacion'] == max_area:
        print(f"🔴 ROJO ({stats_r['porcentaje_ocupacion']:.2f}%)")
    elif stats_g['porcentaje_ocupacion'] == max_area:
        print(f"🟢 VERDE ({stats_g['porcentaje_ocupacion']:.2f}%)")
    else:
        print(f"🔵 AZUL ({stats_b['porcentaje_ocupacion']:.2f}%)")
    
    print(f"Canal con menor área ocupada: ", end="")
    min_area = min(stats_r['porcentaje_ocupacion'], stats_g['porcentaje_ocupacion'], stats_b['porcentaje_ocupacion'])
    if stats_r['porcentaje_ocupacion'] == min_area:
        print(f"🔴 ROJO ({stats_r['porcentaje_ocupacion']:.2f}%)")
    elif stats_g['porcentaje_ocupacion'] == min_area:
        print(f"🟢 VERDE ({stats_g['porcentaje_ocupacion']:.2f}%)")
    else:
        print(f"🔵 AZUL ({stats_b['porcentaje_ocupacion']:.2f}%)")
    
    print("\n✅ Procesamiento completado exitosamente")
    print(f"📊 Visualización guardada: separacion_planos_rgb_ejercicio5.png")

if __name__ == "__main__":
    main()
