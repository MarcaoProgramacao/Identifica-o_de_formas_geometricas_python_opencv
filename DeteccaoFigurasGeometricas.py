import numpy as np
import cv2

# Leitura de imagem
img = cv2.imread("shapes.png")

# Deixando a imagem cinza
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Definindo os limites
_, limite = cv2.threshold(imgGray, 210, 250, cv2.THRESH_BINARY_INV)

# Definindo os contornos
contornos, _ = cv2.findContours(limite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterar sobre os contornos detectados
for contorno in contornos:
    # Ignorar pequenos contornos (ruído)
    area = cv2.contourArea(contorno)
    if area < 500:
        continue  # Ignorar contornos menores que 500 pixels

    # Aproximação do número de contornos para ter uma maior precisão
    approx = cv2.approxPolyDP(contorno, 0.02 * cv2.arcLength(contorno, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)  # Desenhando contornos nas figuras geométricas
    x, y = approx.ravel()[0], approx.ravel()[1]  # Descobrindo as coordenadas x e y da forma geométrica

    if len(approx) == 3:  # Definindo que se o número de contornos for 3, é um triângulo
        cv2.putText(img, "Triangulo", (x+  30, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    elif len(approx) == 4:  # Definindo que se o número de contornos for 3, é um quadrado ou um retângulo

        x, y, w, h = cv2.boundingRect(approx) # Metodo que nos dará as coordenadas e o tamanho da altura e largura da forma geometrica
        aspectRatio = float(w) / h # Uma simples divisão entre o valor da altura e a largura da forma para verificar seu tipo

        if 0.95 <= aspectRatio <= 1.05: # Como o quadrado possui lados iguais, o valor deve da divisão deve ficar em torno de 1
            cv2.putText(img, "Quadrado", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        else:# Caso contrário, caracteriza-se como um retângulo
            cv2.putText(img, "Retangulo", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    elif len(approx) == 5:  # Definindo que se o número de contornos for 5, é um pentagono
        cv2.putText(img, "Pentagono", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    elif len(approx) == 10:  # Definindo que se o número de contornos for 10, é uma estrela
        cv2.putText(img, "Estrela", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    else:  # Definindo que se não for nenhum dos outros casos, é um circulo
        cv2.putText(img, "Circulo", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

# Exibir a imagem com todas as modificações que foram feitas
cv2.imshow("shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
