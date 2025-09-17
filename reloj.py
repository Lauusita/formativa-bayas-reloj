import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

valid_numbers = np.concatenate([np.arange(start, start+60) for start in range(200, 1000, 100)])
reloj_aleatorio = np.random.choice(valid_numbers)

if reloj_aleatorio <= 959:
    reloj_aleatorio = f"0{reloj_aleatorio}"
else:
    reloj_aleatorio = str(reloj_aleatorio)

# Cargar imagen fija de prueba
img = cv2.imread(f"./assets/data-resampled/clock_{reloj_aleatorio}.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Separar canales
R = img_rgb[:,:,0]
G = img_rgb[:,:,1]
B = img_rgb[:,:,2]

mask = (
  (R < 125) & 
  (G < 125) & 
  (B < 125)
).astype(np.uint8) * 255

# Detectar círculo principal
circles = cv2.HoughCircles(
  gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
  param1=100, param2=30, minRadius=100, maxRadius=0
)

def cut_radius(mask, cx, cy, radius, scale=0.7):
    mask_circle = np.zeros_like(mask, dtype=np.uint8)
    inner_radius = int(radius * scale)
    cv2.circle(mask_circle, (cx, cy), inner_radius, 255, -1)
    return cv2.bitwise_and(mask, mask_circle)


if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    x_center, y_center, radius = circles[0]
    masked = cut_radius(mask, x_center, y_center, radius)
    # bye centro del reloj
    cv2.circle(masked, (x_center, y_center), int(radius*0.18), 0, -1)
else:
    h, w = img.shape[:2]
    x_center, y_center, radius = w//2, h//2, min(h,w)//2
    masked = cut_radius(mask, x_center, y_center, radius)
    cv2.circle(masked, (x_center, y_center), int(radius*0.18), 0, -1)


contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar y ordenar por área
contours = [c for c in contours if cv2.contourArea(c) > 50]
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
angles = []

for cnt in contours:
    if cv2.contourArea(cnt) < 50:
        continue

    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)

    # encuentra el punto más lejano al centro (punta de la manecilla)
    distances = np.hypot(cnt[:,0,0] - x_center, cnt[:,0,1] - y_center)
    idx_tip = np.argmax(distances)
    tip = cnt[idx_tip,0]

    dx = tip[0] - x_center
    dy = y_center - tip[1] 

    # Ángulo en grados respecto a las 12
    angle_clock = (math.degrees(math.atan2(dy, dx)) + 360) % 360
    angle_clock = (90 - angle_clock) % 360
    angles.append((angle_clock, tip))

angle_min, angle_hour = angles[0][0], angles[1][0]

minutes = int(round(angle_min / 6)) % 60
hours = int(angle_hour // 30) % 12 or 12

hours += (minutes/60)
hours = int(hours) if hours != 12 else 12

print(f"Hora estimada: {hours:02d}:{minutes:02d}")

plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.title("Reloj"); plt.imshow(img_rgb); plt.axis("off")
plt.subplot(1,3,2); plt.title("Manecillas separadas"); plt.imshow(masked, cmap="gray"); plt.axis("off")
plt.show()
