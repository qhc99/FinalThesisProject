import math
import random

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


def rotate3d(img, rx, ry, rz, f=1000, dx=0, dy=0, dz=1000):
    w, h = img.shape[0], img.shape[1]
    alpha = rx * math.pi / 180.
    beta = ry * math.pi / 180.
    gamma = rz * math.pi / 180.

    RX = np.array([
        [1, 0, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha), 0],
        [0, np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 0, 1]])
    RY = np.array([
        [np.cos(beta), 0, -np.sin(beta), 0],
        [0, 1, 0, 0],
        [np.sin(beta), 0, np.cos(beta), 0],
        [0, 0, 0, 1]])
    RZ = np.array([
        [np.cos(gamma), -np.sin(gamma), 0, 0],
        [np.sin(gamma), np.cos(gamma), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    A1 = np.array([
        [1, 0, -w / 2],
        [0, 1, -h / 2],
        [0, 0, 0],
        [0, 0, 1]])
    R = np.matmul(np.matmul(RX, RY), RZ)
    T = np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]])
    A2 = np.array([
        [f, 0, w / 2, 0],
        [0, f, h / 2, 0],
        [0, 0, 1, 0]])

    trans = np.matmul(A2, np.matmul(T, np.matmul(R, A1)))
    out = img.copy()
    cv2.warpPerspective(src=img, M=trans, dsize=img.shape[:2], dst=out, flags=cv2.INTER_LANCZOS4)
    return out


def augmentation():
    d = "./resources/cutted_img"
    img_names = os.listdir(d)

    for img_name in img_names:
        img = cv2.imread(os.path.join(d, img_name), cv2.IMREAD_COLOR)
        for n in range(0, 10):
            out = rotate3d(img, 0, 30, 0)
            cv2.imshow("orgin", img)
            cv2.imshow("out", out)
            cv2.waitKey()


if __name__ == "__main__":
    augmentation()
    print("success")
