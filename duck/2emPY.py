import cv2

# Defina o nome da janela
JANELA = "camera"

# Carregue o classificador Haar para detecção de rosto
xml = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml"
rosto_cascade = cv2.CascadeClassifier(xml)

# Abra a câmera
camera = cv2.VideoCapture(0)

while True:
    # Capture um frame da câmera
    ret, frame = camera.read()

    if not ret:
        break

    # Converta a imagem para escala de cinza
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduza a imagem para acelerar a detecção
    reduzida = cv2.resize(cinza, (0, 0), fx=0.25, fy=0.25)

    # Detecte rostos
    rostos = rosto_cascade.detectMultiScale(
        reduzida,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Para cada rosto detectado, desenhe um retângulo
    for (x, y, w, h) in rostos:
        # Multiplicamos por 4 para ajustar para a escala original
        x, y, w, h = x * 4, y * 4, w * 4, h * 4
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exiba o resultado
    cv2.imshow(JANELA, frame)

    # Saia do loop se pressionar qualquer tecla
    if cv2.waitKey(10) > 0:
        break

# Libere a câmera e feche as janelas
camera.release()
cv2.destroyAllWindows()
