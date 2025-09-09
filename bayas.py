import cv2
import numpy as np
import matplotlib.pyplot as plt

# Laura Arteta

img = cv2.imread("./bayas.jpeg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# capas de color RGB
R = img_rgb[:,:,0]
G = img_rgb[:,:,1]
B = img_rgb[:,:,2]

# píxeles donde R es mayor que G y B (lo rojo predomina). umbral a uint8 (0 o 255 para mantener la escala)
mask = (
  (R > 215) & 
  (R > G + 50) & 
  (R > B + 50) & 
  ~((R > 200) & (G > 200) & (B > 200))  # descartar blancos
).astype(np.uint8) * 255


contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img2 = img_rgb.copy()

# 3. Dibujar los contornos encima de la copia
cv2.drawContours(img2, contours, -1, (0, 255, 0), 2)

plt.figure(figsize=(12,6))
plt.subplot(1,3,1); plt.title("Original"); plt.imshow(img_rgb); plt.axis("off")
plt.subplot(1,3,2); plt.title("Frutos resaltados"); plt.imshow(mask, cmap="gray"); plt.axis("off")
plt.subplot(1, 3, 3); plt.title("Bordes"); plt.imshow(img2, cmap="gray"); plt.axis("off")

plt.show()
