from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os

def calcular_histograma(imagen_path):
    #verificacion
    if not os.path.exists(imagen_path):
        print(f"ERROR: No se encuentra el archivo {imagen_path}")
        return
    
    #abrir imagen
    img = Image.open(imagen_path)
    
    #convertir a rgb
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    #convertir a array numpy
    img_array = np.array(img)
    
    plt.figure(figsize=(10, 6))
    
    colors = ['red', 'green', 'blue']
    labels = ['Red', 'Green', 'Blue']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        canal = img_array[:, :, i].flatten()
        hist, bins = np.histogram(canal, bins=256, range=(0, 256))
        plt.plot(bins[:-1], hist, color=color, label=label, alpha=0.7)
    
    plt.title('Histograma RGB')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'mono.png')
    
    calcular_histograma(image_path)