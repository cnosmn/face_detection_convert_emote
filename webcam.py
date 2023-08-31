import cv2

# Webcam akışını başlatın
cap = cv2.VideoCapture(0)  # 0, ilk kamerayı temsil eder

# Üzerine eklemek istediğiniz resmi yükleyin
overlay_image = cv2.imread('/home/gf/Desktop/face_detec/faceDet/emotes/1.jpg')  # Eklemek istediğiniz resmin dosya yolu

# Resmi yeniden boyutlandırın (örneğin, 100x100 piksel)
new_width = 100
new_height = 100
overlay_image = cv2.resize(overlay_image, (new_width, new_height))

while True:
    ret, frame = cap.read()  # Kameradan bir çerçeve al

    # Çerçeve ve üst üste gelecek resim boyutları uyumlu hale getirilir
    overlay_height, overlay_width, _ = overlay_image.shape
    y_offset = 10  # Yükseklik için ofset değeri
    x_offset = 10  # Genişlik için ofset değeri

    # Çerçeve üzerine resmi ekleyin
    for y in range(overlay_height):
        for x in range(overlay_width):
            if x_offset + x < frame.shape[1] and y_offset + y < frame.shape[0]:
                frame[y_offset + y, x_offset + x] = overlay_image[y, x]

    # Görüntüyü gösterin
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakın ve pencereleri kapatın
cap.release()
cv2.destroyAllWindows()