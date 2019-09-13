import numpy as np 
import pickle
import cv2
import os

def load_data(train_path, val_path, test_path, input_size=256):


    if not os.path.exists("./pickle"):

        os.makedirs("./pickle")

    if not os.path.exists("./pickle/train.pickle") or \
        not os.path.exists("./pickle/val.pickle") or \
        not os.path.exists("./pickle/test.pickle"):

        train_x = []
        train_y = []

        val_x = []
        val_y = []

        test_x = []
        test_y = []


        train_paths = list(open(train_path, "r", encoding='utf-8').readlines())
        val_paths = list(open(val_path, "r", encoding='utf-8').readlines())
        test_paths = list(open(test_path, "r", encoding='utf-8').readlines())

        data = [train_paths, val_paths, test_paths]

        for idx in range(len(data)):

            for path in data[idx]:
                
                trash_hot_class = [0] * 6
                trash_hot_class[int(path[path.find(" ") + 1]) - 1] = 1

                trash_type = "".join([i for i in path[:-7] if i not in "0123456789"])            
                image = cv2.imread(os.path.join("garbage-classification/Garbage classification", trash_type, path[:-3])).astype(float)
                image = cv2.resize(image, (input_size, input_size))
                image /= 255.0

                if idx == 0:
                    train_x.append(image)
                    train_y.append(trash_hot_class)
                elif idx == 1:
                    val_x.append(image)
                    val_y.append(trash_hot_class)
                elif idx == 2:
                    test_x.append(image)
                    test_y.append(trash_hot_class)

        with open("./pickle/train.pickle", "wb") as f:
            pickle.dump([train_x, train_y], f)
        
        with open("./pickle/val.pickle", "wb") as f:
            pickle.dump([val_x, val_y], f)
        
        with open("./pickle/test.pickle", "wb") as f:
            pickle.dump([test_x, test_y], f)

        return ([train_x, train_y], [val_x, val_y], [test_x, test_y])
    
    else:

        with open("./pickle/train.pickle", "rb") as f:
            train_data = pickle.load(f)
        
        with open("./pickle/val.pickle", "rb") as f:
            val_data = pickle.load(f)
        
        with open("./pickle/test.pickle", "rb") as f:
            test_data = pickle.load(f)

        return (train_data, val_data, test_data)
    

def batch_iter(data, batch_size, num_epochs, shuffle=True):

    data = np.array(data)
    data_size = len(data)

    num_batches_per_ep = int((data_size - 1) / batch_size) + 1

    for epoch in range(num_epochs):

        if shuffle:
            shuffle_idx = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_idx]

        else:
            shuffled_data = data
        
        for batch_num in range(num_batches_per_ep):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx:end_idx]
