import os, math, cv2, imutils, argparse
import numpy as np
from os import XATTR_SIZE_MAX
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
import ntpath
import random

"""
파일 이름을 받아서 열고 이미지의 크기가 1MB보다 크면
크기를 점점 줄여나가서 1MB보다 작게 되면 해당 이미지를 리턴하는 함수
처음부터 1MB보다 작다면 그대로 리턴
"""


def resize(filename): 
    img = cv2.imread(filename)
    width, height = img.shape[:2]
    if height * width * 3 <= 2 ** 25 :
        return img
    i = 2
    t_height, t_width = height, width
    while t_height * t_width * 3 > 2 ** 25:
        t_height = int(t_height / math.sqrt(i))
        t_width = int(t_width / math.sqrt(i))
        i += 1
    height, width = t_height, t_width
    image = Image.open(filename)
    resize_image = image.resize((height, width))
    filename = (
        filename[: -1 * (len(filename.split(".")[-1]) + 1)]
        + "_resized."
        + filename.split(".")[-1]
    )
    resize_image.save(filename)
    img = cv2.imread(filename)
    os.system("del " + filename.replace("/", "\\"))
    return img

def matchlabel(img, img2, stitchedimg, labelsfile1, labelsfile2):
    #img = cv.imread('ImageStitching/CD_7711-7712/CD_7711.jpg',0)
    #img2 = cv.imread('ImageStitching/CD_7711-7712/CD_7712.jpg',0)
    #stitchedimg = cv.imread('ImageStitching/CD_7711-7712/CD_7711-CD_7712.jpg',0)
    tempshape = 300

    template = img[0:tempshape,0:tempshape]
    template2 = img2[0:tempshape,0:tempshape]
    w, h = template.shape[::-1]

    #labelsfile1 = open('data/litterdata/trainimg/labels/CD_7711.txt','r')
    #labelsfile2 = open('data/litterdata/trainimg/labels/CD_7712.txt','r')

    labels1 = labelsfile1.read().splitlines()
    labels2 = labelsfile2.read().splitlines()

    # All the 6 methods for comparison in a list
    meth = 'cv.TM_CCORR_NORMED'
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
    print(labels1)
    print(labels2)

    targetfiledirectory = ''
    target = open(targetfiledirectory,'w+')

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
        #cv.rectangle(stitchedimg, TL,BR,255,6)
        #cv.rectangle(img,(int(splitlabs[1]*x1 - splitlabs[3]*x1/2),int(splitlabs[2]*y1 - splitlabs[4]*y1/2)),(int(splitlabs[1]*x1 + splitlabs[3]*x1/2),int(splitlabs[2]*y1 + splitlabs[4]*y1/2)),(255,0,0),6)
        joinlab = ' '.join(splitlabsres)
        target.write(joinlab)

    for i, labs in enumerate(labels2):
        splitlabs = labs.split(' ')

        for i, elem in enumerate(splitlabs):
            splitlabs[i] = float(elem)
        cenx = (c2+(matchedtemp2W*splitlabs[1]*x2/tempshape))/X
        ceny = (d2+(matchedtemp2H*splitlabs[2]*y2/tempshape))/Y
        cenw = splitlabs[3]*(x2*tempshape/matchedtemp2W)/X
        cenh = splitlabs[4]*(y2*tempshape/matchedtemp2H)/Y
        splitlabsres =  [splitlabs[0], cenx, ceny, cenw, cenh]
        TL = (int(cenx*X - (cenw*X/2)), int(ceny*Y - (cenh*Y/2)))
        BR = (int(cenx*X + (cenw*X/2)), int(ceny*Y + cenh*Y/2))
        #cv.rectangle(stitchedimg, TL,BR,255,6)
        #cv.rectangle(img2,(int(splitlabs[1]*x2 - splitlabs[3]*x2/2),int(splitlabs[2]*y2 - splitlabs[4]*y2/2)),(int(splitlabs[1]*x2 + splitlabs[3]*x2/2),int(splitlabs[2]*y2 + splitlabs[4]*y2/2)),(255,0,0),6)
        joinlab = ' '.join(splitlabsres)
        target.write(joinlab)

    target.close()

    '''cv.rectangle(stitchedimg, top_left, bottom_right, 0, 6)
    cv.rectangle(stitchedimg,top_left2, bottom_right2, 0, 6)
    plt.subplot(131),plt.imshow(img,cmap = 'gray')
    plt.subplot(132),plt.imshow(img2,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(stitchedimg,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()'''



"""
argument parsing
-i 입력 이미지들의 폴더 디렉토리
-o 출력할 결과 이미지의 파일 이름
"""
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--images",
    default="edited_images",
    type=str,
    required=True,
    help="path to input directory of images to stitch",
)
ap.add_argument(
    "-o", "--output", default="ImageStitching", type=str, required=True, help="path to the output image"
)
args = vars(ap.parse_args())


"""
loading images
입력 받은 폴더의 모든 파일들을 리사이즈 시키고 리스트에 로드하는 부분
"""
print("[INFO] loading images...")
imagePaths = Path(args["images"])
images = []

namecounter = 0
for imagePath in imagePaths.rglob("*.jpg"):
    if namecounter == 0:
        firstname = ntpath.basename(imagePath)[:-4]
    image = resize(str(imagePath))
    images.append(image)
    lastname = ntpath.basename(imagePath)[:-4]
    namecounter += 1
    if namecounter > 12:
        break

print(firstname)
print(lastname)
print(args["output"]+"/"+firstname+"-"+lastname+".jpg")

"""
image stitching
이미지 스티칭하여 성공 여부(status)와 결과(stitched)를 받는 부분
"""
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)
"""
exception handling
status가 0인 경우(stitching 성공) 이미지를 아까 받은 이름으로 저장하고 보여준다.
status가 0이 아닌 경우(stitching 실패 예외)의 메시지를 출력한다.
    1 : 이미지를 연결 시키기에 match point가 부족해서 나오는 에러, 이미지를 더 추가시켜줘야 한다.
    2 : 2D 이미지 변환을 하지 못하는 에러, 이미지를 다시 찍는 것을 추천한다.
    3 : 카메라 위치의 에러, 카메라의 방향이 잘못돼서 나오는 에러, 입력 이미지들을 같은 방향으로 회전시키거나 새로운 이미지를 찍어야 한다.
"""
if status == 0:
    # write the output stitched image to disk
    cv2.imwrite(args["output"]+"/"+firstname+"-"+lastname+".jpg", stitched)

    _mask = np.ones(stitched.shape)
    mask_img = stitched * _mask
    mask_img = mask_img[:,:,0]

    count = 0
    w, h = mask_img.shape
    for ww in range(w):
        for hh in range(h):
            if mask_img[ww][hh] != 0:
                count += 1

    area = w*h - count
    print(area)

    # display the output stitched image to our screen
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)
else:
    if status == cv2.STITCHER_ERR_NEED_MORE_IMGS:
        print("[INFO] image stitching failed (1: STITCHER_ERR_NEED_MORE_IMGS)")
    elif status == cv2.STITCHER_ERR_HOMOGRAPHY_EST_FAIL:
        print("[INFO] image stitching failed (2: STITCHER_ERR_HOMOGRAPHY_EST_FAIL)")
    else:
        print(
            "[INFO] image stitching failed (3: STITCHER_ERR_CAMERA_PARAMETERS_ADJUSTMENT_FAIL)"
        )

if __name__ == "__main__":
    pass
