from skimage import io
from PIL import Image
import cv2
import os
from multiprocessing import Pool
from matplotlib import image as mt_image
from shutil import copyfile

NEG_IMGS_FOLDER_PATH = "../../dataset/TrafficBlockSign/neg_imgs/imgs"


def findBroken(image_path: str):
    # noinspection PyBroadException
    try:
        io_img = io.imread(image_path)
        img = cv2.imread(image_path)
        with Image.open(image_path) as p_img:
            if img is None:
                print(image_path[image_path.rfind("\\") + 1:])
            elif io_img is None:
                print(image_path[image_path.rfind("\\") + 1:])
            elif p_img is None:
                print(image_path[image_path.rfind("\\") + 1:])
    except Exception:
        print(image_path[image_path.rfind("\\") + 1:])
        return False


def process(img_name: str):
    img_path = os.path.join(NEG_IMGS_FOLDER_PATH, img_name)
    findBroken(img_path)


def printBrokenImages():
    img_names = os.listdir(NEG_IMGS_FOLDER_PATH)
    with Pool() as pool:
        try:
            pool.map(process, img_names)
        except Exception as e:
            print(e)


def transform(img_path: str, output_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    ret_img = cv2.merge([img, img, img, img])
    mt_image.imsave(fname=output_path, format="svg", arr=ret_img)


def copy(name):
    src_dir = "../../dataset/TrafficBlockSign/neg_imgs/imgs"
    dst_dir = "../../yolo_sign_dataset/images/train"
    copyfile(os.path.join(src_dir, name), os.path.join(dst_dir, name))


def checkCCTSDB():
    truth_path = "G:\\download\\CCTSDB-master\\GroundTruth\\groundtruth0000-9999.txt"
    main_folder_path = "G:\\download\\CCTSDB-master\\img"


if __name__ == "__main__":
    img = cv2.imread("../header.png", cv2.IMREAD_COLOR)
    cv2.imshow("origin", img)
    w, h, _ = img.shape
    for i in range(w):
        for j in range(h):
            pixel = img[i, j]
            if (max(pixel[0], pixel[1]) - min(pixel[0], pixel[1]) < 20) and \
                    (max(pixel[1], pixel[2]) - min(pixel[1], pixel[2]) < 20):
                img[i, j] = [255, 255, 255]
    cv2.imshow("img", img)
    cv2.imwrite("header.png", img)

    cv2.waitKey()
    print("success")
