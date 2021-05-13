import matplotlib
import numpy as np
from skimage import io
from PIL import Image
import cv2
import os
from multiprocessing import Pool
from matplotlib import image as mt_image
from shutil import copyfile
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

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


def getPath():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename()


def whiteBG(name, diff=5):
    img = cv2.imread(getPath(), cv2.IMREAD_COLOR)
    cv2.imshow("origin", img)
    w, h, _ = img.shape
    for i in range(w):
        for j in range(h):
            pixel = img[i, j]
            if max(pixel[0], pixel[1], pixel[2]) > 240 and \
                    max(pixel[0], pixel[1]) - min(pixel[0], pixel[1]) < diff and \
                    (max(pixel[1], pixel[2]) - min(pixel[1], pixel[2]) < diff):
                img[i, j] = [255, 255, 255]
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.imwrite(name, img)


def plotLine(x, y, xlabel="", ylabel="", title=""):
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(x, y, "bo")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def cutImg():
    file_path = "./resources/info.dat"
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            items = line[:-1].split(" ")
            img_name = items[0]
            boxes = items[2:]
            boxes_count = int(items[1])
            img = cv2.imread(os.path.join("./resources/img", img_name), cv2.IMREAD_COLOR)
            for b_idx in range(0, boxes_count):
                x = int(boxes[b_idx * 4])
                y = int(boxes[b_idx * 4 + 1])
                w = int(boxes[b_idx * 4 + 2])
                h = int(boxes[b_idx * 4 + 3])
                target = img[y:y + h+2, x:x + w+2]
                try:
                    cv2.imshow("cut", target)
                except cv2.error:
                    print(img.shape)
                    print(img_name, x, y, w, h)
                    raise Exception
                cv2.waitKey()


if __name__ == "__main__":
    cutImg()
    print("success")
