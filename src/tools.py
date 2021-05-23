import math
import random

import matplotlib
import numpy as np
from skimage import io
from PIL import Image
import cv2
import os
from multiprocessing import Pool
from multiprocessing import pool
from matplotlib import image as mt_image
from shutil import copyfile
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from utils.plots import plot_one_box
from predict import img_resize

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


def _process(img_name: str):
    img_path = os.path.join(NEG_IMGS_FOLDER_PATH, img_name)
    findBroken(img_path)


def printBrokenImages():
    img_names = os.listdir(NEG_IMGS_FOLDER_PATH)
    with Pool() as p:
        try:
            p.map(_process, img_names)
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
    plt.plot(x, y, "b-")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def rotate3d(img, rx, ry, rz, f=2000, dx=0, dy=0, dz=2000):
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
    d = "D:\\cutted_img"
    img_names = os.listdir(d)
    with pool.Pool(8) as p:
        p.map(func=_augImg, iterable=img_names)


# background_img_names = os.listdir("D:\\COCO_COCO_2014_Train_Images\\train2014")


def _augImg(img_name):
    global background_img_names
    xy_angle = [i for i in range(-36, 38, 2)]
    z_angle = [i for i in range(-24, 26, 2)]
    cutted_img_folder = "D:\\cutted_img"
    img = cv2.imread(os.path.join(cutted_img_folder, img_name), cv2.IMREAD_COLOR)
    img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    for n in range(0, 10):
        out = rotate3d(img, random.sample(xy_angle, 1)[0], random.sample(xy_angle, 1)[0],
                       random.sample(z_angle, 1)[0])
        out = removeBorder(out)
        out = cv2.copyMakeBorder(out, 0, 3, 0, 3, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        out = _adjustBrightAndContrast(out)
        cv2.imwrite(os.path.join("D:\\TrafficBlockSign\\pos_imgs\\img", str(n) + img_name), out)
    print(img_name)


c = [i for i in range(-35, 40, 5)]
b = [i for i in range(-35, 40, 5)]


def _adjustBrightAndContrast(img):
    img = np.int16(img)
    contrast = random.sample(c, 1)[0]
    bright = random.sample(b, 1)[0]
    img = img * (contrast / 127 + 1) - contrast + bright
    img = np.clip(img, 0, 255)
    return np.uint8(img)


def removeBorder(out):
    idx = np.argwhere(np.all(out[..., :] <= 10, axis=0))
    out = np.delete(out, idx, axis=1)

    idx = np.argwhere(np.all(out[:, ...] <= 10, axis=1))
    out = np.delete(out, idx, axis=0)

    return out


def writeInfoDat():
    with open("D:\\TrafficBlockSign\\pos_imgs\\info.dat", "w") as file:
        names = os.listdir("D:\\TrafficBlockSign\\pos_imgs\\img")
        with pool.Pool() as p:
            data = p.map(_getPosLabel, names)
        file.writelines(data)


def _getPosLabel(name):
    img = cv2.imread(os.path.join("D:\\TrafficBlockSign\\pos_imgs\\img", name))
    print(name)
    return "./img\\" + name + f" 1 0 0 {img.shape[1] - 3} {img.shape[0] - 3}\n"


def removeAllFiles():
    root = getPath()
    paths = os.listdir(root)

    def remove(name):
        os.remove(os.path.join(root, name))

    with pool.Pool() as p:
        p.map(remove, paths)


def removeCOCO():
    file = getPath()
    with open(file, "r") as f:
        lines = f.readlines()
    coco_root = "D:\\COCO_COCO_2014_Train_Images\\train2014"

    def remove(name):
        os.remove(os.path.join(coco_root, name))
        print(os.path.join(coco_root, name))

    for line in lines:
        line = line[:-1]
        remove(line)


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
    path = "D:\\TrafficBlockSign\\pos_imgs\\img"
    names = os.listdir(path)

    def remove(name):
        os.remove(os.path.join(path, name))

    with pool.ThreadPool() as p:
        p.map(remove, names)


def signDataProcessing():
    GroundTruthTxtPath = "D:\\CCTSDB (CSUST Chinese Traffic Sign Detection Benchmark)\\GroundTruth\\GroundTruth.txt"
    img_folder = "D:\\CCTSDB (CSUST Chinese Traffic Sign Detection Benchmark)\\Images"
    f = open(GroundTruthTxtPath, "r")
    lines = f.readlines()
    lines = [i[:-1] for i in lines if i.endswith("prohibitory\n")]  # prohibitory
    f.close()
    with pool.Pool() as p:
        formatted = p.map(splitData, lines)
    ptr = formatted[0][:-1]
    nested = [formatted[0][:-1]]
    for info in formatted[1:]:
        img_name = info[0]
        ptr_name = ptr[0]
        if img_name == ptr_name:
            box = info[1:-1]
            nested[-1] = nested[-1] + box
        else:
            ptr = info
            nested.append(info[:-1])

    random.shuffle(nested)
    for img_info in nested:
        img_name = img_info[0]
        boxes = img_info[1:]
        img = cv2.imread(os.path.join(img_folder, img_name), cv2.IMREAD_COLOR)
        for i in range(0, len(boxes) // 4):
            cv2.rectangle(img, (int(round(float(boxes[i * 4]), 0)), int(round(float(boxes[i * 4 + 1]), 0))),
                          (int(round(float(boxes[i * 4 + 2]), 0)), int(round(float(boxes[i * 4 + 3]), 0))),
                          [0, 0, 255], thickness=3)
        cv2.imshow("img", img)
        cv2.waitKey()


def splitData(line):
    return line.split(";")


def padImg():
    w = 232
    img = cv2.imread("temp.png", cv2.IMREAD_COLOR)
    print(img.shape)
    left = int((w - img.shape[1]) / 2)
    image = cv2.copyMakeBorder(img, 0, 0, left, w - left - img.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 255])
    print(image.shape)
    cv2.imwrite("folder_structure.png", image)


if __name__ == "__main__":
    padImg()
    print("success")
