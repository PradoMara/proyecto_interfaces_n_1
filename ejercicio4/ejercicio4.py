from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import os

#carga imagen
def cargar_imagen(carpeta, archivo):
    ruta = os.path.join(os.path.dirname(__file__), carpeta, archivo)
    return Image.open(ruta) if os.path.exists(ruta) else None

#aplicar plantilla
def aplicar_plantilla(fondo, mujer, forma):
    mujer = mujer.resize(fondo.size).convert('RGB')
    forma = forma.resize(fondo.size).convert('L')
    fondo = fondo.convert('RGB')
    
    # Aplicar desenfoque muy sutil a la máscara
    forma_blur = forma.filter(ImageFilter.GaussianBlur(radius=2))
    
    fondo_arr = np.array(fondo, dtype=np.float32)
    mujer_arr = np.array(mujer, dtype=np.float32)
    mascara = np.array(forma_blur, dtype=np.float32) / 255.0
    
    # Invertir máscara si es necesario
    if np.mean(mascara) > 0.5:
        mascara = 1.0 - mascara
    
    # Aplicar máscara con degradado sutil
    mascara_3d = np.stack([mascara, mascara, mascara], axis=2)
    resultado = fondo_arr * (1 - mascara_3d) + mujer_arr * mascara_3d
    
    return Image.fromarray(np.clip(resultado, 0, 255).astype(np.uint8))

#main
def main():
    # Cargar imagen de lela
    mujer = cargar_imagen('plantillas', 'pla_00.jpg')
    
    # Crear carpeta resultados
    carpeta_resultados = os.path.join(os.path.dirname(__file__), 'resultados')
    os.makedirs(carpeta_resultados, exist_ok=True)
    
    # Procesar imágenes
    figuras = ['fig_01.jpg', 'fig_02.jpg', 'fig_03.jpg', 'fig_04.jpg']
    plantillas = ['pla_01.jpg', 'pla_02.jpg', 'pla_03.jpg', 'pla_04.jpg']
    
    originales, procesadas = [], []
    
    for i, (fig, pla) in enumerate(zip(figuras, plantillas)):
        # Cargar imágenes
        fondo = cargar_imagen('figuras', fig)
        forma = cargar_imagen('plantillas', pla)
        
        if fondo and forma and mujer:
            # Procesar
            resultado = aplicar_plantilla(fondo, mujer, forma)
            
            # Guardar
            resultado.save(os.path.join(carpeta_resultados, f"procesada_{fig}"))
            
            # Para visualización
            originales.append(fondo)
            procesadas.append(resultado)
    
    # Mostrar resultados
    fig, axes = plt.subplots(2, len(originales), figsize=(16, 8))
    for i, (orig, proc) in enumerate(zip(originales, procesadas)):
        axes[0, i].imshow(orig)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(proc)
        axes[1, i].set_title('Procesada')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()