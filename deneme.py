import cv2
import numpy as np
circular_image = cv2.imread("/home/gf/Desktop/face_detec/faceDet/cat-4.jpg")



height, width, _ = circular_image.shape
circle_mask = np.zeros((height, width), dtype=np.uint8)
center = (width // 2, height // 2)
radius = min(width, height) // 2
cv2.circle(circle_mask, center, radius, (255, 255, 255), thickness=-1)
circular_image = cv2.bitwise_and(circular_image, circular_image, mask=circle_mask)

circular_image = cv2.resize(circular_image,(200,200))
x_offset = 100  # Dairesel resmin eklenme başlangıç noktası (x koordinatı)
y_offset = 100  # Dairesel resmin eklenme başlangıç noktası (y koordinatı)
for z in range(circular_image.shape[0]):
    for c in range(circular_image.shape[1]):
        if circular_image[z, c][0] == 0 :
            circular_image[z, c][0] = 10
        if circular_image[z, c][1] == 0 :
            circular_image[z, c][1] = 10
        if circular_image[z, c][2] == 0:
            circular_image[z, c][2] = 10
cv2.imwrite("/home/gf/Desktop/face_detec/faceDet/1-1.jpg",circular_image)