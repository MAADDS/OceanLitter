from pathlib import Path
from PIL import Image

import os, math, cv2, imutils, argparse
import numpy as np


"""
ImageStitcher
Initialized with path str of image folder
stitch() returns stitched single image
"""


class ImageStitcher:
    def __init__(self, input_path):
        self.input_dir = Path(input_path)
        self.output_path = "./output/"
        self.input_images = []

        self.__load_images()

    """ resize : 이미지의 크기가 1MB보다 작아질 떄까지 크기를 점점 줄여나감 """

    def __resize(self, image_path):
        image = cv2.imread(image_path)
        width, height = image.shape[:2]

        if height * width * 3 <= 2 ** 25:
            return image

        i = 2
        t_height, t_width = height, width
        while t_height * t_width * 3 > 2 ** 25:
            t_height = int(t_height / math.sqrt(i))
            t_width = int(t_width / math.sqrt(i))
            i += 1

        height, width = t_height, t_width
        image = Image.open(image_path)
        resized_image = image.resize((height, width))

        image_path = (
            image_path[: -1 * (len(image_path.split(".")[-1]) + 1)]
            + "_resized."
            + image_path.split(".")[-1]
        )

        resized_image.save(image_path)
        image = cv2.imread(image_path)
        os.system("del " + image_path.replace("/", "\\"))

        return image

    """ load_images : 입력 받은 폴더의 모든 파일들을 리사이즈 후 리스트에 로드 """

    def __load_images(self):
        print("🟦 Loading images...")
        for image_path in self.input_dir.rglob("*"):
            print(image_path)
            resized_image = self.__resize(str(image_path))
            self.input_images.append(resized_image)

    """ stitch : 이미지 스티칭 시도, 성공 여부(status)와 결과(stitched) 리턴 """

    def stitch(self):
        print("🟦 Stitching images...")
        stitcher = cv2.createStitcher(ㅓ) if imutils.is_cv3() else cv2.Stitcher_create()
        (status, stitched) = stitcher.stitch(self.input_images)

        if status == 0:  # Stitching Success
            # write the output stitched image to disk
            print(self.output_path + self.input_dir.name)
            cv2.imwrite(self.output_path + self.input_dir.name + ".jpg", stitched)

        else:  # Stitching Fail
            if status == cv2.STITCHER_ERR_NEED_MORE_IMGS:
                # 이미지를 연결 시키기에 match point가 부족해서 나오는 에러, 이미지를 더 추가해야 함
                print("❗ Stitching failed (1: STITCHER_ERR_NEED_MORE_IMGS)")
            elif status == cv2.STITCHER_ERR_HOMOGRAPHY_EST_FAIL:
                # 2D 이미지 변환을 하지 못하는 에러
                print("❗ Stitching failed (2: STITCHER_ERR_HOMOGRAPHY_EST_FAIL)")
            else:
                # 카메라 위치 에러, 카메라의 방향이 잘못돼서 나오는 에러, 입력 이미지들을 같은 방향으로 회전시키거나 새로운 이미지를 찍어야 함
                print(
                    "❗ Stitching failed (3: STITCHER_ERR_CAMERA_PARAMETERS_ADJUSTMENT_FAIL)"
                )
