from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Cargar imagen
    img = Image.open(os.path.join(os.path.dirname(__file__), 'fig_00.jpg')).convert('RGB')
    img_array = np.array(img)
    img_gray = img.convert('L')
    
    print(f"Imagen: {img.size[0]}x{img.size[1]} píxeles")
    
    # Análisis RGB
    colors = ['red', 'green', 'blue']
    labels = ['R', 'G', 'B']
    tonalidades = []
    
    print("\nTONALIDADES DOMINANTES:")
    for i, (color, label) in enumerate(zip(colors, labels)):
        hist, _ = np.histogram(img_array[:, :, i], bins=256, range=(0, 256))
        dom = np.argmax(hist)
        pct = (hist[dom] / img_array[:, :, i].size) * 100
        tonalidades.append(dom)
        print(f"Canal {label}: {dom} ({pct:.1f}%)")
    
    # Análisis escala de grises
    gray_array = np.array(img_gray)
    hist_gray, _ = np.histogram(gray_array, bins=256, range=(0, 256))
    dom_gray = np.argmax(hist_gray)
    media = np.mean(gray_array)
    
    print(f"\nESCALA DE GRISES:")
    print(f"Dominante: {dom_gray}")
    print(f"Media: {media:.1f} ({'oscura' if media < 85 else 'media' if media < 170 else 'clara'})")
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Imagen original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')
    
    # Histograma RGB
    for i, (color, label) in enumerate(zip(colors, labels)):
        hist, bins = np.histogram(img_array[:, :, i], bins=256, range=(0, 256))
        axes[0, 1].plot(bins[:-1], hist, color=color, label=label, alpha=0.7)
        axes[0, 1].axvline(tonalidades[i], color=color, linestyle='--', alpha=0.8)
    
    # Agregar texto con dominantes en esquina superior
    texto_dominantes = f"Dominantes:\nR: {tonalidades[0]}\nG: {tonalidades[1]}\nB: {tonalidades[2]}"
    axes[0, 1].text(0.98, 0.98, texto_dominantes, transform=axes[0, 1].transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9, fontweight='bold')
    
    axes[0, 1].set_title('Histograma RGB')
    axes[0, 1].set_xlabel('Intensidad (0=Oscuro, 255=Brillante)')
    axes[0, 1].set_ylabel('Frecuencia (Cantidad de píxeles)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Imagen gris
    axes[1, 0].imshow(img_gray, cmap='gray')
    axes[1, 0].set_title('Escala de Grises')
    axes[1, 0].axis('off')
    
    # Histograma gris
    axes[1, 1].plot(np.arange(256), hist_gray, color='black', linewidth=2)
    axes[1, 1].axvline(dom_gray, color='red', linestyle='--', label=f'Dominante: {dom_gray}')
    axes[1, 1].axvline(media, color='blue', linestyle=':', label=f'Media: {media:.1f}')
    axes[1, 1].set_title('Histograma Escala de Grises')
    axes[1, 1].set_xlabel('Intensidad de Gris (0=Negro, 255=Blanco)')
    axes[1, 1].set_ylabel('Frecuencia (Cantidad de píxeles)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Conclusiones
    print(f"\nCONCLUSIONES:")
    print(f"RGB dominantes: R={tonalidades[0]}, G={tonalidades[1]}, B={tonalidades[2]}")
    if max(tonalidades) - min(tonalidades) < 30:
        print("Imagen balanceada en RGB")
    else:
        canal_dom = ['Rojo', 'Verde', 'Azul'][np.argmax(tonalidades)]
        print(f"Canal {canal_dom} domina")

if __name__ == "__main__":
    main()