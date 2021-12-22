import cv2 as cv
import imutils

img = cv.imread('data/litterdata/trainimg/images/CD_25.jpg')
imgH, imgW = img.shape[:2]
labels = [0, 0.7675, 0.5625, 0.0325, 0.04375]
#template = img[int(imgH*labels[2]-imgH*labels[4]/2):int(imgH*labels[2]+imgH*labels[4]/2),int(imgW*labels[1]-imgW*labels[3]/2):int(imgW*labels[1]+imgW*labels[3]/2)]
template = img[400:500, 100:200]
# rotated = imutils.rotate_bound(template, -33)
# H,W = rotated.shape[:2]
# print(rotated.shape)
# rotated_grey = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)
# _, mask2 = cv.threshold(rotated_grey[:,:], 1, 255, cv.THRESH_BINARY)
# mask_inv = cv.bitwise_not(mask2)
# roi = img[100:100+H, 100:100+W]
# masked_fg = cv.bitwise_and(rotated,rotated,  mask = mask2)
# masked_bg = cv.bitwise_and(roi, roi, mask = mask_inv)
# added = masked_fg + masked_bg
# img[100:100+H, 100:100+W] = added
# cv.circle(img,(200,200), 100, (0,0,255), 5)
cv.rectangle(img, (100,400) , (200,500),255,14)

cv.imshow("display",template)
cv.imshow("display2",img)
cv.waitKey(0)