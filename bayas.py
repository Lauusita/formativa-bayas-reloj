import cv2
import numpy as np
import matplotlib.pyplot as plt

# Laura Arteta

img = cv2.imread("./assets/bayas.jpeg")
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

def dibujar_contornos_manual(imagen, contornos, color, grosor):
  for contorno in contornos:
    # Cada contorno es una lista de puntos
    for i in range(len(contorno)):
      punto_actual = contorno[i][0]
      punto_siguiente = contorno[(i + 1) % len(contorno)][0]  # vecino más cercano
      
      # Dibujar línea entre puntos usando algoritmo de Bresenham simplificado
      x1, y1 = punto_actual[0], punto_actual[1]
      x2, y2 = punto_siguiente[0], punto_siguiente[1]
      
      # Calcular diferencias
      dx = abs(x2 - x1)
      dy = abs(y2 - y1)
      
      # Determinar dirección
      sx = 1 if x1 < x2 else -1
      sy = 1 if y1 < y2 else -1
      
      # Algoritmo de Bresenham para dibujar línea pixel por pixel
      err = dx - dy
      x, y = x1, y1
      
      while True:
        # Dibujar píxel con grosor
        for gx in range(-grosor//2, grosor//2 + 1):
          for gy in range(-grosor//2, grosor//2 + 1):
            px, py = x + gx, y + gy
            if 0 <= px < imagen.shape[1] and 0 <= py < imagen.shape[0]:
              imagen[py, px] = color
        
        # Si llegamos al punto final, salir
        if x == x2 and y == y2:
          break
        
        # Calcular siguiente punto
        e2 = 2 * err
        if e2 > -dy:
          err -= dy
          x += sx
        if e2 < dx:
          err += dx
          y += sy

# Aplicar la función manual
dibujar_contornos_manual(img2, contours, (0, 0, 255), 2)

plt.figure(figsize=(12,6))
plt.subplot(1,3,1); plt.title("Original"); plt.imshow(img_rgb); plt.axis("off")
plt.subplot(1,3,2); plt.title("Frutos resaltados"); plt.imshow(mask, cmap="gray"); plt.axis("off")
plt.subplot(1, 3, 3); plt.title("Bordes"); plt.imshow(img2, cmap="gray"); plt.axis("off")

plt.show()
