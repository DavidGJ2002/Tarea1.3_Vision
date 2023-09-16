import matplotlib.pyplot as plt
import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('monedas.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar el filtro Gaussiano para reducir el ruido
imagen_suavizada = cv2.GaussianBlur(imagen, (5, 5), 1.4)

# Calcular los gradientes utilizando los kernels de Sobel
Ix = cv2.Sobel(imagen_suavizada, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(imagen_suavizada, cv2.CV_64F, 0, 1, ksize=3)

# Calcular la magnitud y el ángulo del gradiente
magnitud_gradiente = np.sqrt(Ix**2 + Iy**2)
angulo_gradiente = np.arctan2(Iy, Ix)

# Supresión no máxima
imagen_suprimida = np.zeros_like(magnitud_gradiente)
angulo_gradiente = angulo_gradiente * 180.0 / np.pi
angulo_gradiente[angulo_gradiente < 0] += 180

for i in range(1, imagen.shape[0] - 1):
    for j in range(1, imagen.shape[1] - 1):
        angulo = angulo_gradiente[i, j]
        if (0 <= angulo < 22.5) or (157.5 <= angulo <= 180):
            vecinos = [magnitud_gradiente[i, j - 1], magnitud_gradiente[i, j + 1]]
        elif 22.5 <= angulo < 67.5:
            vecinos = [magnitud_gradiente[i - 1, j - 1], magnitud_gradiente[i + 1, j + 1]]
        elif 67.5 <= angulo < 112.5:
            vecinos = [magnitud_gradiente[i - 1, j], magnitud_gradiente[i + 1, j]]
        else:
            vecinos = [magnitud_gradiente[i - 1, j + 1], magnitud_gradiente[i + 1, j - 1]]
        
        if magnitud_gradiente[i, j] >= max(vecinos):
            imagen_suprimida[i, j] = magnitud_gradiente[i, j]

# Doble umbral
umbral_bajo = 30
umbral_alto = 100

imagen_umbral = np.zeros_like(imagen_suprimida)
imagen_umbral[(imagen_suprimida >= umbral_bajo) & (imagen_suprimida <= umbral_alto)] = 255

# Mostrar la imagen resultante utilizando matplotlib
plt.figure(figsize=(8, 8))
plt.subplot(121)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')

plt.subplot(122)
plt.imshow(imagen_umbral, cmap='gray')
plt.title('Filtro Canny')
plt.show()