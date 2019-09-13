import numpy as np 
import pickle
import cv2
import os
import shutil

def load_data(train_path, val_path, test_path):


    train_x = []
    train_y = []

    val_x = []
    val_y = []

    test_x = []
    test_y = []


    train_paths = list(open(train_path, "r", encoding='utf-8').readlines())
    val_paths = list(open(val_path, "r", encoding='utf-8').readlines())
    test_paths = list(open(test_path, "r", encoding='utf-8').readlines())

    root_dir = "garbage-classification/Garbage classification"

    data = [train_paths, val_paths, test_paths]

    if not os.path.exists(os.path.join(root_dir, "train")):
        os.mkdir(os.path.join(root_dir, "train"))

    if not os.path.exists(os.path.join(root_dir, "val")):
        os.mkdir(os.path.join(root_dir, "val"))

    if not os.path.exists(os.path.join(root_dir, "test")):
        os.mkdir(os.path.join(root_dir, "test"))

    for idx in range(len(data)):

        for path in data[idx]:

            trash_type = "".join([i for i in path[:-7] if i not in "0123456789"])

            if idx == 0:

                if not os.path.exists(os.path.join(root_dir, "train", trash_type)):
                    os.mkdir(os.path.join(root_dir, "train", trash_type))

                newPath = shutil.copy(
                    os.path.join(root_dir, trash_type, path[:-3]),
                    os.path.join(root_dir, "train", trash_type))


            elif idx == 1:

                if not os.path.exists(os.path.join(root_dir, "val", trash_type)):
                    os.mkdir(os.path.join(root_dir, "val", trash_type))

                newPath = shutil.copy(
                    os.path.join(root_dir, trash_type, path[:-3]),
                    os.path.join(root_dir, "val", trash_type))

            elif idx == 2:

                if not os.path.exists(os.path.join(root_dir, "test", trash_type)):
                    os.mkdir(os.path.join(root_dir, "test", trash_type))

                newPath = shutil.copy(
                    os.path.join(root_dir, trash_type, path[:-3]),
                    os.path.join(root_dir, "test", trash_type))

# load_data(
#         "./garbage-classification/one-indexed-files-notrash_train.txt",
#         "./garbage-classification/one-indexed-files-notrash_val.txt",
#         "./garbage-classification/one-indexed-files-notrash_test.txt")

def dis_data_from_test(test_path):

    with open(test_path, "r", encoding='utf-8') as f:

        test_paths = f.readlines()
        test_length = len(test_paths)

        train_path = test_paths[:int(test_length * 0.8)]
        val_path = test_paths[int(test_length * 0.8):]

        root_dir = "garbage-classification/Garbage classification"

        for path in train_path:
            
            print(path)
            trash_type = "".join([i for i in path[:-7] if i not in "0123456789"])

            newPath = shutil.copy(
                    os.path.join(root_dir, trash_type, path[:-3]),
                    os.path.join(root_dir, "train", trash_type))

        for path in val_path:
            
            trash_type = "".join([i for i in path[:-7] if i not in "0123456789"])

            newPath = shutil.copy(
                    os.path.join(root_dir, trash_type, path[:-3]),
                    os.path.join(root_dir, "val", trash_type))

dis_data_from_test("./garbage-classification/one-indexed-files-notrash_test.txt")