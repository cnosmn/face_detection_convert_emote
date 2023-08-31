import cv2
import numpy as np
import argparse
import dlib
import random


def rect_to_bb(rect): 
    # take a bounding predicted by dlib and convert it 
    # # to the format (x, y, w, h) as we would normally do 
    # # with OpenCV 
    x = rect.left() 
    y = rect.top() 
    w = rect.right() - x 
    h = rect.bottom() - y 
    # return a tuple of (x, y, w, h) 
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
        # resize the image
        
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def draw_rectangle(image,gray,rects,predictor,number):
    
    for (i,rect) in enumerate(rects):

        shape = predictor(gray,rect)
        
        shape = shape_to_np(shape)

        (x,y,w,h) = rect_to_bb(rect)
        # 绘制人脸的矩形框
        # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        # # 设置矩形框的文字部分
        # cv2.putText(image,"Face #{}".format(i+1),(x-10,y-10),
        #             cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)       
        
        emote_img = cv2.imread("/home/gf/Desktop/face_detec/faceDet/emotes/2-1.jpg")
        emote_img = cv2.resize(emote_img,(w,h))
        if image[y:y+h,x:x+w].shape == emote_img.shape:
            # image[y:y+h,x:x+w] = emote_img
            height, width, _ = emote_img.shape
            circle_mask = np.zeros((height, width), dtype=np.uint8)
            center = (width // 2, height // 2)
            radius = min(width, height) // 2
            cv2.circle(circle_mask, center, radius, (255, 255, 255), thickness=-1)
            circular_image = cv2.bitwise_and(emote_img, emote_img, mask=circle_mask)

            circular_image = cv2.resize(circular_image,(w,w))
            x_offset = x  # Dairesel resmin eklenme başlangıç noktası (x koordinatı)
            y_offset = y  # Dairesel resmin eklenme başlangıç noktası (y koordinatı)
            for z in range(circular_image.shape[0]):
                for c in range(circular_image.shape[1]):
                    if circular_image[z, c][0] != 0 and circular_image[z, c][1] != 0 and circular_image[z, c][2] != 0:
                        image[z + y_offset, c + x_offset] = circular_image[z, c]

        
        # for (x,y) in shape:
        #     cv2.circle(image,(x,y),1,(0,0,255),-1)
    
    return image 

def webcam(predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    number = random.randint(1,5)
    
    cam = cv2.VideoCapture(2)
    
    while True:
        check, frame = cam.read()
        key = cv2.waitKey(1)
        image = resize(frame,width = 1000, height = 1500)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        #print(gray.shape)

        rects = detector(gray,1)

        draw_image = draw_rectangle(image=image,gray=gray,rects=rects,predictor=predictor,number=number)
            
        cv2.imshow("Output",draw_image)
        
        if key == 27:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()


# detector = dlib.get_frontal_face_detector()

# predictor = dlib.shape_predictor("/home/gf/Desktop/face_detec/faceDet/weight.dat")

# image = cv2.imread("/home/gf/Desktop/face_detec/faceDet/1.jpg")

webcam("/home/gf/Desktop/face_detec/faceDet/weight.dat")

