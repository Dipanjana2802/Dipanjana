import os
import numpy as np
import time
import shutil
#import cPickle
import random

from PIL import Image, ImageOps


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def move_files(input, output):
    '''
        Input: folder with dataset, where every class is in separate folder
        Output: all images, in format class_number.jpg; output path should be absolute
    '''
    index = -1
    for root, dirs, files in os.walk(input):
        path = root.split('/')
        print ('Working with path ', path)
        print ('Path index ', index)
        filenum = 0
        for file in files:
            fileName, fileExtension = os.path.splitext(file)
            if fileExtension == '.jpg' or fileExtension == '.JPG':
                full_path = '<path to images>' + path[9] + '/'+ file

                if (os.path.isfile(full_path)):
                    file = str(index) + '_' + path[1] + str(filenum) + fileExtension
                    print (output + '/' + file)
                    shutil.copy(full_path, output + '/' + file)
                else:
                    print('No file')
                filenum += 1
        index += 1


def create_text_file(input_path, outpath, percentage):
    '''
        Creating train.txt and val.txt for feeding Caffe
    '''

    images, labels = [], []
    os.chdir(input_path)

    for item in os.listdir('.'):
        if not os.path.isfile(os.path.join('.', item)):
            continue
        try:
            label = int(item.split('_')[0])
            images.append(item)
            labels.append(label)
        except:
            continue

    images = np.array(images)
    labels = np.array(labels)
    images, labels = shuffle_in_unison(images, labels)
    im_length = len(images)
    im_labels = len(labels)
    #print('image length: {}'.format(im_length))
    #print('image length type: {}'.format(type(im_length)))
    #print('image label: {}'.format(im_labels))
    #print('image label type: {}'.format(type(im_labels)))

    train_size = int(im_length*percentage)

    X_train = images[0:train_size]
    y_train = labels[0:train_size]

    X_test = images[train_size:]
    y_test = labels[train_size:]

    os.chdir(outpath)
    print('The current directory for output is: {}'.format(os.getcwd()))

    trainfile = open("train.txt", "w")
    #trainfile.write('Hello Train')
    for i, l in zip(X_train, y_train):
        trainfile.write(i + " " + str(l) + "\n")

    testfile = open("val.txt", "w")

    for i, l in zip(X_test, y_test):
        testfile.write(i + " " + str(l) + "\n")

    trainfile.close()
    testfile.close()

def main():
    caffe_path = '<path to images>'
    new_path = '<path to a temp folder>'
    output_path = '<desired output location>'
    move_files(caffe_path, new_path)
    create_text_file(new_path, output_path, 0.85)

main()
