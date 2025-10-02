import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import Tuple, Dict, Union

# ============================================================================
# FUNCIONES BÁSICAS DE CARGA Y CONVERSIÓN DE IMÁGENES
# ============================================================================

def cargar_imagen(ruta: str) -> np.ndarray:
    """
    Carga una imagen usando PIL y la convierte a numpy array.
    Mantiene el formato original (RGB o gris).
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se pudo encontrar {ruta}")
    
    img = Image.open(ruta)
    return np.array(img)

def cargar_imagen_color(ruta: str) -> np.ndarray:
    """
    Carga una imagen usando PIL y la convierte a numpy array RGB.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se pudo encontrar {ruta}")
    
    img = Image.open(ruta).convert('RGB')
    return np.array(img)

def convertir_a_gris(img: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen a escala de grises usando la fórmula estándar.
    Si ya está en gris, la devuelve tal como está.
    """
    if len(img.shape) == 3:
        # Imagen en color RGB, convertir usando fórmula estándar
        img_gris = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2])
        return img_gris.astype(np.uint8)
    else:
        # Ya está en escala de grises
        return img.astype(np.uint8)

# ============================================================================
# FUNCIONES DE SEPARACIÓN DE PLANOS DE COLOR
# ============================================================================

def separar_planos_color(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separa los planos de color RGB de una imagen.
    Devuelve los canales R, G, B por separado.
    """
    canal_rojo = img_rgb[:,:,0]
    canal_verde = img_rgb[:,:,1] 
    canal_azul = img_rgb[:,:,2]
    
    return canal_rojo, canal_verde, canal_azul

def separar_planos_rgb(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Alias para separar_planos_color para compatibilidad.
    """
    return separar_planos_color(img_rgb)

# ============================================================================
# FUNCIONES DE EXTRACCIÓN DE FIGURAS Y MÁSCARAS
# ============================================================================

def extraer_figura_color(img_rgb: np.ndarray, tolerancia: int = 50) -> np.ndarray:
    """
    Extrae la figura de color ignorando el fondo blanco.
    Devuelve una máscara binaria donde 1 = figura, 0 = fondo blanco.
    """
    # Identificar píxeles que NO son blancos (o casi blancos)
    es_blanco = np.all(img_rgb >= (255 - tolerancia), axis=2)
    
    # La figura es todo lo que NO es blanco
    mascara_figura = ~es_blanco
    
    return mascara_figura.astype(np.uint8)

# ============================================================================
# FUNCIONES DE CÁLCULO DE MOMENTOS
# ============================================================================

def calcular_area(mascara: np.ndarray) -> int:
    """
    Calcula el área contando píxeles de la figura.
    """
    return int(np.sum(mascara))

def calcular_momento_crudo(mascara: np.ndarray, p: int, q: int) -> float:
    """
    Calcula el momento crudo M_pq = Σ x^p * y^q
    donde (x,y) son las coordenadas de los píxeles de la figura.
    """
    y_coords, x_coords = np.where(mascara > 0)
    if len(x_coords) == 0:
        return 0.0
    
    momento = np.sum((x_coords.astype(np.float64) ** p) * (y_coords.astype(np.float64) ** q))
    return float(momento)

def calcular_centroide_por_pixeles(mascara: np.ndarray) -> Tuple[float, float]:
    """
    Calcula el centroide promediando las coordenadas de los píxeles.
    """
    y_coords, x_coords = np.where(mascara > 0)
    if len(x_coords) == 0:
        return (0.0, 0.0)
    
    cx = np.mean(x_coords)
    cy = np.mean(y_coords)
    return (float(cx), float(cy))

def calcular_centroide_por_momentos(mascara: np.ndarray) -> Tuple[float, float]:
    """
    Calcula el centroide usando momentos: cx = M10/M00, cy = M01/M00
    """
    M00 = calcular_momento_crudo(mascara, 0, 0)  # Área
    if M00 == 0:
        return (0.0, 0.0)
    
    M10 = calcular_momento_crudo(mascara, 1, 0)
    M01 = calcular_momento_crudo(mascara, 0, 1)
    
    cx = M10 / M00
    cy = M01 / M00
    return (float(cx), float(cy))

def calcular_momento_central(mascara: np.ndarray, cx: float, cy: float, p: int, q: int) -> float:
    """
    Calcula el momento central μ_pq = Σ (x-cx)^p * (y-cy)^q
    """
    y_coords, x_coords = np.where(mascara > 0)
    if len(x_coords) == 0:
        return 0.0
    
    momento_central = np.sum(((x_coords - cx) ** p) * ((y_coords - cy) ** q))
    return float(momento_central)

def calcular_momento_central_normalizado(mu_pq: float, mu_00: float, p: int, q: int) -> float:
    """
    Calcula el momento central normalizado η_pq = μ_pq / μ_00^γ
    donde γ = 1 + (p+q)/2
    """
    if mu_00 == 0:
        return 0.0
    
    gamma = 1 + (p + q) / 2.0
    return mu_pq / (mu_00 ** gamma)

def calcular_momentos_hu(mascara: np.ndarray) -> Tuple[float, float, float]:
    """
    Calcula los primeros 3 momentos de Hu de forma simplificada.
    """
    # Calcular centroide
    cx, cy = calcular_centroide_por_momentos(mascara)
    
    # Calcular momentos centrales necesarios
    mu_00 = calcular_momento_central(mascara, cx, cy, 0, 0)  # Área
    mu_20 = calcular_momento_central(mascara, cx, cy, 2, 0)
    mu_02 = calcular_momento_central(mascara, cx, cy, 0, 2)
    mu_11 = calcular_momento_central(mascara, cx, cy, 1, 1)
    mu_30 = calcular_momento_central(mascara, cx, cy, 3, 0)
    mu_12 = calcular_momento_central(mascara, cx, cy, 1, 2)
    mu_21 = calcular_momento_central(mascara, cx, cy, 2, 1)
    mu_03 = calcular_momento_central(mascara, cx, cy, 0, 3)
    
    # Normalizar
    if mu_00 == 0:
        return (0.0, 0.0, 0.0)
    
    eta_20 = calcular_momento_central_normalizado(mu_20, mu_00, 2, 0)
    eta_02 = calcular_momento_central_normalizado(mu_02, mu_00, 0, 2)
    eta_11 = calcular_momento_central_normalizado(mu_11, mu_00, 1, 1)
    eta_30 = calcular_momento_central_normalizado(mu_30, mu_00, 3, 0)
    eta_12 = calcular_momento_central_normalizado(mu_12, mu_00, 1, 2)
    eta_21 = calcular_momento_central_normalizado(mu_21, mu_00, 2, 1)
    eta_03 = calcular_momento_central_normalizado(mu_03, mu_00, 0, 3)
    
    # Calcular los primeros 3 momentos de Hu
    h1 = eta_20 + eta_02
    h2 = (eta_20 - eta_02)**2 + 4 * eta_11**2
    h3 = (eta_30 - 3*eta_12)**2 + (3*eta_21 - eta_03)**2
    
    return (float(h1), float(h2), float(h3))

# ============================================================================
# FUNCIONES DE ANÁLISIS DE ÁREA OCUPADA
# ============================================================================

def calcular_area_ocupada(canal: np.ndarray, umbral: int = 0) -> dict:
    """
    Calcula el área ocupada de píxeles en un canal de color.
    Considera píxeles "ocupados" aquellos con valor > umbral.
    """
    # Área total de la imagen
    area_total = canal.size
    
    # Píxeles ocupados (con valor > umbral)
    pixeles_ocupados = np.sum(canal > umbral)
    
    # Porcentaje de ocupación
    porcentaje_ocupacion = (pixeles_ocupados / area_total) * 100
    
    # Estadísticas adicionales
    estadisticas = {
        'area_total': area_total,
        'pixeles_ocupados': pixeles_ocupados,
        'pixeles_vacios': area_total - pixeles_ocupados,
        'porcentaje_ocupacion': porcentaje_ocupacion,
        'porcentaje_vacio': 100 - porcentaje_ocupacion,
        'valor_minimo': int(canal.min()),
        'valor_maximo': int(canal.max()),
        'valor_promedio': float(canal.mean()),
        'intensidad_promedio_ocupados': float(canal[canal > umbral].mean()) if pixeles_ocupados > 0 else 0.0
    }
    
    return estadisticas

# ============================================================================
# FUNCIONES DE COLORIZACIÓN
# ============================================================================

def aplicar_colormap(img_gris: np.ndarray, colormap: str = 'ocean') -> np.ndarray:
    """
    Aplica un mapa de colores a una imagen en escala de grises.
    """
    # Normalizar la imagen a rango [0, 1]
    img_normalizada = img_gris / 255.0
    
    # Obtener el colormap de matplotlib
    cmap = plt.cm.get_cmap(colormap)
    
    # Aplicar el colormap
    img_coloreada = cmap(img_normalizada)
    
    # Convertir de vuelta a rango [0, 255] y eliminar canal alpha
    img_coloreada_rgb = (img_coloreada[:, :, :3] * 255).astype(np.uint8)
    
    return img_coloreada_rgb

def aplicar_coloracion_personalizada(img_gris: np.ndarray, color_base: tuple = (0, 100, 255)) -> np.ndarray:
    """
    Aplica una coloración personalizada basada en un color base.
    """
    # Normalizar la imagen de gris
    intensidad = img_gris / 255.0
    
    # Crear imagen RGB vacía
    altura, ancho = img_gris.shape
    img_coloreada = np.zeros((altura, ancho, 3), dtype=np.uint8)
    
    # Aplicar el color base modulado por la intensidad
    for i in range(3):  # R, G, B
        img_coloreada[:, :, i] = (color_base[i] * intensidad).astype(np.uint8)
    
    return img_coloreada

# ============================================================================
# FUNCIONES AUXILIARES PARA VISUALIZACIÓN
# ============================================================================

def crear_imagen_con_centroide(ruta_original: str, cx: float, cy: float, ruta_salida: str):
    """
    Crea una copia de la imagen original con el centroide marcado.
    """
    img = Image.open(ruta_original).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Dibujar una cruz en el centroide
    tamano_cruz = 10
    cx_int, cy_int = int(round(cx)), int(round(cy))
    
    # Línea horizontal
    draw.line([cx_int - tamano_cruz, cy_int, cx_int + tamano_cruz, cy_int], 
              fill=(255, 255, 0), width=3)  # Amarillo
    
    # Línea vertical  
    draw.line([cx_int, cy_int - tamano_cruz, cx_int, cy_int + tamano_cruz], 
              fill=(255, 255, 0), width=3)  # Amarillo
    
    img.save(ruta_salida)

def guardar_imagen_gris(img_gris: np.ndarray, ruta_salida: str):
    """
    Guarda una imagen en escala de grises usando PIL.
    """
    img_pil = Image.fromarray(img_gris, mode='L')
    img_pil.save(ruta_salida)

def guardar_imagen_color(img_color: np.ndarray, ruta_salida: str):
    """
    Guarda una imagen en color usando PIL.
    """
    img_pil = Image.fromarray(img_color, 'RGB')
    img_pil.save(ruta_salida)

# ============================================================================
# FUNCIONES DE ESTADÍSTICAS COMUNES
# ============================================================================

def analizar_estadisticas_canales(canal_r: np.ndarray, canal_g: np.ndarray, canal_b: np.ndarray, img_gris: np.ndarray = None):
    """
    Muestra estadísticas básicas de cada canal de color.
    """
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE LOS CANALES DE COLOR")
    print("="*60)
    
    canales = [
        ("Canal ROJO", canal_r),
        ("Canal VERDE", canal_g), 
        ("Canal AZUL", canal_b)
    ]
    
    if img_gris is not None:
        canales.append(("Escala de GRISES", img_gris))
    
    for nombre, canal in canales:
        print(f"\n{nombre}:")
        print(f"  Valor mínimo: {canal.min():3d}")
        print(f"  Valor máximo: {canal.max():3d}")
        print(f"  Promedio:     {canal.mean():.1f}")
        print(f"  Desv. estándar: {canal.std():.1f}")
