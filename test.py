import cv2
img = cv2.imread('test/1.png', -1)
h, w = img.shape[0:2]
img = cv2.resize(img, (w // 2, h // 2))
cv2.imshow('test', img*255)
cv2.waitKey(0)
print(img.max())