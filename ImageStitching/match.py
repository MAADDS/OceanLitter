from os import XATTR_SIZE_MAX
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('ImageStitching/CD_77773-77780/CD_77779.jpg',0)
img2 = cv.imread('ImageStitching/CD_77773-77780/CD_77780.jpg',0)
stitchedimg = cv.imread('ImageStitching/CD_77773-77780/CD_77779-CD_77780.jpg',0)

tempshape = 300

template = img[0:tempshape,0:tempshape]
template2 = img2[0:tempshape,0:tempshape]
w, h = template.shape[::-1]

labelsfile1 = open('data/litterdata/trainimg/labels/CD_77779.txt','r')
labelsfile2 = open('data/litterdata/trainimg/labels/CD_77780.txt','r')

labels1 = labelsfile1.read().splitlines()
labels2 = labelsfile2.read().splitlines()

# All the 6 methods for comparison in a list
meth = 'cv.TM_CCORR_NORMED'
img11 = img.copy()
img22 = img2.copy()
method = eval(meth)

# Apply template Matching
res = cv.matchTemplate(stitchedimg,template,method)
res2 = cv.matchTemplate(stitchedimg,template2,method)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
min_val2, max_val2, min_loc2, max_loc2 = cv.minMaxLoc(res2)

top_left = max_loc # c, d
bottom_right = (top_left[0] + w, top_left[1] + h)

top_left2 = max_loc2 # c, d
bottom_right2 = (top_left2[0] + w, top_left2[1] + h)

matchedtemp1W = bottom_right[0] - top_left[0] # A
matchedtemp1H = bottom_right[1] - top_left[1] # B

matchedtemp2W = bottom_right2[0] - top_left2[0] # A
matchedtemp2H = bottom_right2[1] - top_left2[1] # B

c1, d1 = top_left[0], top_left[1]
c2, d2  = top_left2[0], top_left[1]

y1, x1 = img.shape[:2]
y2, x2 = img2.shape[:2]

Y, X = stitchedimg.shape[:2]

for i, labs in enumerate(labels1):
    splitlabs = labs.split(' ')
    
    for i, elem in enumerate(splitlabs):
        splitlabs[i] = float(elem)
    cenx = (c1+(matchedtemp1W*splitlabs[1]*x1/tempshape))/X
    ceny = (d1+(matchedtemp1H*splitlabs[2]*y1/tempshape))/Y
    cenw = splitlabs[3]*(x1*tempshape/matchedtemp1W)/X
    cenh = splitlabs[4]*(y1*tempshape/matchedtemp1H)/Y
    splitlabsres = [splitlabs[0], cenx, ceny, cenw, cenh]
    TL = (int(cenx*X - (cenw*X/2)), int(ceny*Y - (cenh*Y/2)))
    BR = (int(cenx*X + (cenw*X/2)), int(ceny*Y + cenh*Y/2))
    cv.rectangle(stitchedimg, TL,BR,255,6)
    cv.rectangle(img,(int(splitlabs[1]*x1 - splitlabs[3]*x1/2),int(splitlabs[2]*y1 - splitlabs[4]*y1/2)),(int(splitlabs[1]*x1 + splitlabs[3]*x1/2),int(splitlabs[2]*y1 + splitlabs[4]*y1/2)),(255,0,0),6)
    print(splitlabs)

for i, labs in enumerate(labels2):
    splitlabs = labs.split(' ')

    for i, elem in enumerate(splitlabs):
        splitlabs[i] = float(elem)
    cenx = (c2+(matchedtemp2W*splitlabs[1]*x2/tempshape))/X
    ceny = (d2+(matchedtemp2H*splitlabs[2]*y2/tempshape))/Y
    cenw = splitlabs[3]*(x2*tempshape/matchedtemp2W)/X
    cenh = splitlabs[4]*(y2*tempshape/matchedtemp2H)/Y
    splitlabsres2 =  [splitlabs[0], cenx, ceny, cenw, cenh]
    TL = (int(cenx*X - (cenw*X/2)), int(ceny*Y - (cenh*Y/2)))
    BR = (int(cenx*X + (cenw*X/2)), int(ceny*Y + cenh*Y/2))
    cv.rectangle(stitchedimg, TL,BR,255,6)
    cv.rectangle(img2,(int(splitlabs[1]*x2 - splitlabs[3]*x2/2),int(splitlabs[2]*y2 - splitlabs[4]*y2/2)),(int(splitlabs[1]*x2 + splitlabs[3]*x2/2),int(splitlabs[2]*y2 + splitlabs[4]*y2/2)),(255,0,0),6)
    print(splitlabs)



cv.rectangle(stitchedimg, top_left, bottom_right, 0, 6)
cv.rectangle(stitchedimg,top_left2, bottom_right2, 0, 6)
plt.subplot(131),plt.imshow(img,cmap = 'gray')
plt.subplot(132),plt.imshow(img2,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(stitchedimg,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle(meth)
plt.show()
