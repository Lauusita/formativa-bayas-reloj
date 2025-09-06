import cv2
import numpy as np
import matplotlib.pyplot as plt

# Laura Arteta
# Jesús Berdugo

img = cv2.imread("./bayas.jpeg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# capas de color RGB
R = img_rgb[:,:,0]
G = img_rgb[:,:,1]
B = img_rgb[:,:,2]

# píxeles donde R es mayor que G y B (lo rojo predomina)
mask = (
  (R > 215) & 
  (R > G + 50) & 
  (R > B + 50) & 
  ~((R > 200) & (G > 200) & (B > 200))  # descartar blancos
)

# Convertir umbral a uint8 (0 o 255 para mantener la escala)
mask = mask.astype(np.uint8) * 255

edges = cv2.Canny(mask, threshold1=100, threshold2=100)

plt.figure(figsize=(12,6))
plt.subplot(1,3,1); plt.title("Original"); plt.imshow(img_rgb); plt.axis("off")
plt.subplot(1,3,2); plt.title("Frutos resaltados"); plt.imshow(mask, cmap="gray"); plt.axis("off")
plt.subplot(1, 3, 3); plt.title("Bordes"); plt.imshow(edges, cmap="gray"); plt.axis("off")

plt.show()