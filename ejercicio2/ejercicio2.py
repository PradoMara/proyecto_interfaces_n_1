import cv2
from matplotlib import pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'mono.png')

img = cv2.imread(image_path, 1)
cv2.imshow('Imagen', img)

colors = ['blue', 'green', 'red']
labels = ['Blue', 'Green', 'Red']

hist=cv2.calcHist([img], [0], None, [256], [0, 256])

fig=plt.subplots(figsize=(10, 5))

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for i, (color, label) in enumerate(zip(colors, labels)):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=color, label=label, alpha=0.7)

plt.title('Histogramas RGB')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()
plt.show()