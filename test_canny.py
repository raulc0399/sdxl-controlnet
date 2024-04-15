import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def callback(x):
    print(x)

def resize_image(img, resolution):
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    return cv2.resize(img, (W_target, H_target), interpolation=interpolation)

def apply_canny(img, l, u):
    # img_blur = cv2.GaussianBlur(img,(3, 3),0)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(img, l, u)

    # contours = cv2.findContours(canny, 
    #                         cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(canny, contours[0], -1, (255,0,0), thickness = 3)

    return canny

img = cv2.imread(r"D:\raul\stuff\objs\obj4\4j.jpg")
img_to_show = resize_image(img, 512) 

l = 15
u = 55

canny = apply_canny(img_to_show, l, u) 

cv2.namedWindow('image') # make a window with name 'image'
cv2.resizeWindow('image', 600,600)
cv2.createTrackbar('L', 'image', 0, 255, callback) #lower threshold trackbar for window 'image
cv2.createTrackbar('U', 'image', 0, 255, callback) #upper threshold trackbar for window 'image

cv2.setTrackbarPos('L', 'image', l)
cv2.setTrackbarPos('U', 'image', u)

while(1):
    canny_to_show = canny[:, :, None]
    canny_to_show = np.concatenate([canny_to_show, canny_to_show, canny_to_show], axis=2)
    # canny_to_show = resize_image(canny_to_show, 512)
    
    numpy_horizontal_concat = np.concatenate((img_to_show, canny_to_show), axis=1) # to display image side by side
    cv2.imshow('image', numpy_horizontal_concat)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: #escape key
        break
    l = cv2.getTrackbarPos('L', 'image')
    u = cv2.getTrackbarPos('U', 'image')

    canny = apply_canny(img_to_show, l, u)

cv2.destroyAllWindows()