import math
import random

import matplotlib
import numpy as np
import cv2
import os
from multiprocessing import pool
from shutil import copyfile
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

NEG_IMGS_FOLDER_PATH = "../../dataset/TrafficBlockSign/neg_imgs/imgs"
TRAIN_POS_FOLDER = "C:\\Users\\Nathan\\Documents\\dataset\\TrafficBlockSign\\pos_imgs\\img"
INFO_DAT_PATH = "C:\\Users\\Nathan\\Documents\\dataset\\TrafficBlockSign\\pos_imgs\\info.dat"
BG_FILE = "C:\\Users\\Nathan\\Documents\\dataset\\TrafficBlockSign\\neg_imgs\\bg.txt"


def copy(name):
    src_dir = "../../dataset/TrafficBlockSign/neg_imgs/imgs"
    dst_dir = "../../yolo_sign_dataset/images/train"
    copyfile(os.path.join(src_dir, name), os.path.join(dst_dir, name))


def getPath():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename()


def plotLine(x, y, xlabel="", ylabel="", title=""):
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(x, y, "b-")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def rotate3d(img, rx, ry, rz, f=800, dx=0, dy=0, dz=800):
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
    d = "C:\\Users\\Nathan\\Documents\\dataset\\mask"
    img_names = os.listdir(d)
    # for img_name in img_names:
    #     _augImg(img_name)
    with pool.Pool() as p:
        p.map(func=_augImg, iterable=img_names)


def _augImg(img_name):
    xy_angle = [i for i in range(-36, 37, 1)]
    z_angle = [i for i in range(-12, 13, 1)]
    mask_img_folder = "C:\\Users\\Nathan\\Documents\\dataset\\mask"
    pos_img_folder = "C:\\Users\\Nathan\\Documents\\dataset\\cut_img"
    bg_img_folder = "C:\\Users\\Nathan\\Downloads\\COCO_COCO_2014_Train_Images\\train2014"
    bg_img_names = os.listdir(bg_img_folder)
    pos_img = cv2.imread(os.path.join(pos_img_folder, img_name), cv2.IMREAD_COLOR)
    for n in range(0, 200):
        try:
            bg_img_name = random.sample(bg_img_names, 1)[0]
            rand = random.random()
            third1 = 0.4
            third2 = 0.8
            if rand <= third1:
                bg_img = cv2.imread(os.path.join(bg_img_folder, bg_img_name), cv2.IMREAD_COLOR)
                while bg_img.shape[0] < pos_img.shape[0] and bg_img.shape[1] < pos_img.shape[1]:
                    bg_img = cv2.resize(bg_img, (bg_img.shape[1] * 2, bg_img.shape[2] * 2), cv2.INTER_CUBIC)
                cut_x = random.random()
                cut_y = random.random()
                x_start = int((bg_img.shape[1] - pos_img.shape[1]) * cut_x)
                y_start = int((bg_img.shape[0] - pos_img.shape[0]) * cut_y)
                bg_img = bg_img[y_start:y_start + pos_img.shape[0], x_start:x_start + pos_img.shape[1], :]
            elif third1 < rand <= third2:
                bg_img = np.zeros((pos_img.shape[0], pos_img.shape[1], 3), np.uint8)
            else:
                bg_img = np.zeros((pos_img.shape[0], pos_img.shape[1], 3), np.uint8)
                bg_img += 255
            mask = cv2.imread(os.path.join(mask_img_folder, img_name), cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)

            aug_img = cv2.bitwise_and(pos_img, pos_img, mask=mask)

            rotate_x = random.sample(xy_angle, 1)[0]
            rotate_y = random.sample(xy_angle, 1)[0]
            rotate_z = random.sample(z_angle, 1)[0]
            aug_img = rotate3d(aug_img, rotate_x, rotate_y, rotate_z)

            mask = rotate3d(mask, rotate_x, rotate_y, rotate_z)
            _, mask_inv = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY_INV)

            bg_img = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)
            aug_img += bg_img
            aug_img = _adjustBrightAndContrast(aug_img)
            cv2.imwrite(os.path.join(TRAIN_POS_FOLDER, f"{n}{img_name}"), aug_img)
        except Exception:
            pass
    print(img_name)


c = [i for i in range(-70, 71, 1)]
b = [i for i in range(-70, 71, 1)]


def _adjustBrightAndContrast(img):
    img = np.int16(img)
    contrast = random.sample(c, 1)[0]
    bright = random.sample(b, 1)[0]
    img = img * (contrast / 127 + 1) - contrast + bright
    img = np.clip(img, 0, 255)
    return np.uint8(img)


def writeInfoDat():
    with open(INFO_DAT_PATH, "w") as file:
        names = os.listdir(TRAIN_POS_FOLDER)
        with pool.Pool() as p:
            data = p.map(_getPosLabel, names)
        file.writelines(data)


def _getPosLabel(name):
    img = cv2.imread(os.path.join(TRAIN_POS_FOLDER, name))
    print(name)
    return "./img\\" + name + f" 1 0 0 {img.shape[1] - 1} {img.shape[0] - 1}\n"


def stackImage():
    p = "D:\\cutted_img"
    img_names = os.listdir(p)
    sample_names = random.sample(img_names, 15)
    sample_imgs = [cv2.imread(os.path.join(p, name), cv2.IMREAD_COLOR) for name in sample_names]
    stack = np.ndarray([300, 500, 3], dtype=np.uint8)
    for idx, img in enumerate(sample_imgs):
        img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
        row = idx // 5
        col = idx % 5
        stack[row * 100:(row + 1) * 100, col * 100:(col + 1) * 100] = img
    cv2.imwrite("origin_stack.png", stack)
    pass


def removeAugment():
    path = TRAIN_POS_FOLDER
    names = os.listdir(path)

    def remove(name):
        os.remove(os.path.join(path, name))

    with pool.ThreadPool() as p:
        p.map(remove, names)


def removeCOCOSign():
    with open("C:\\Users\\Nathan\\Libs\\val_id.txt") as ids:
        lines = ids.readlines()
        with pool.Pool() as p:
            p.map(_removeName, lines)


def _removeName(name):
    name = name[:-1]
    directory = "C:\\Users\\Nathan\\Downloads\\COCO_COCO_2014_Val_Images\\val2014"
    os.remove(os.path.join(directory, name))


def writeBG():
    p1 = "C:\\Users\\Nathan\\Downloads\\COCO_COCO_2014_Train_Images\\train2014"
    p2 = "C:\\Users\\Nathan\\Downloads\\COCO_COCO_2014_Val_Images\\val2014"
    l1 = os.listdir(p1)
    l2 = os.listdir(p2)

    bg = open(BG_FILE, "w")

    def _writeBG1(name):
        t = os.path.join(p1, name)
        bg.write(t)
        bg.write("\n")

    def _writeBG2(name):
        t = os.path.join(p2, name)
        bg.write(t)
        bg.write("\n")

    with pool.ThreadPool() as p:
        p.map(_writeBG1, l1)
        p.map(_writeBG2, l2)

    bg.close()


if __name__ == "__main__":
    # writeBG()
    removeAugment()
    augmentation()
    writeInfoDat()
    print("success")
