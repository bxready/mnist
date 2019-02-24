# -*- coding: utf-8 -*-
#
import numpy as np
import os.path
import matplotlib.pyplot as plt
import csv
from PIL import Image
import random
from keras.preprocessing.image import ImageDataGenerator

class kova_mnist_tools():

    messages = {
        'CREATE_OBJECT': 'Create AlexKova Mnist tools Object',
        'SAVE_PROGRESS_TITLE': 'save data to disc processed ...',
        'SAVE_DATA_TO_DISK' : 'Save data to CSV file'
    }


    def __init__(self,
                 batch_file_mask = 'example_b_{}.csv',
                 target_file_mask = 'example_{}.csv',
                 train_file='train.csv',
                 path_to_files = ''
                 ):
        print(self.messages['CREATE_OBJECT'])
        self.batch_file_mask = path_to_files + batch_file_mask
        self.target_file_mask = path_to_files + target_file_mask
        self.train_file = path_to_files + train_file


    def load_train_dataset(self, batch_mode = False, batch_count=0):
        """
        function load dataset from kaggle train data csv
        example:
        ------------------------------------------------
        data_set = load_train_dataset('files/dev_train.csv')
        -------------------------------------------------
        """
        train_dataset = np.array([]);
        file_name = self.train_file if not batch_mode else self.batch_file_mask.format(batch_count)
        if os.path.isfile(file_name):
            train_dataset = np.loadtxt(file_name, skiprows=1, delimiter=',')
            print(file_name)
        return train_dataset


    def split_dataset(self, train_dataset_element):
        """
        function split kaggle dataset into images and values
        example:
        ------------------------------------------------
        dataset = load_train_dataset('files/dev_train.csv')
        train_images, train_values = split_dataset(dataset)
        -------------------------------------------------
        """
        train_images = train_dataset_element[:, 1:]
        train_values = train_dataset_element[:, 0]
        return train_images, train_values


    def split_dataset_element(self, train_dataset_element):
        """
        function split kaggle dataset element into image and value
        example:
        ------------------------------------------------
        dataset = split_dataset_element('files/dev_train.csv')
        train_image, train_value = split_dataset_element(dataset)
        -------------------------------------------------
        """
        train_image = train_dataset_element[1:len(train_dataset_element)]
        train_value = train_dataset_element[0]
        return train_image, train_value



    def show_dataset_images(self, train_image, train_value, multiple = False):
        """
        procedure show kaggle dataset element image by image
        example:
        ------------------------------------------------
        show_dataset_images(train_image, train_value)
        -------------------------------------------------
        """
        if multiple > 0:
            for i in train_image:
                self.show_dataset_images(i, train_value)
        else:
            plt.imshow(Image.fromarray(train_image.reshape(28, 28)))
            plt.show()



    def norm(self, image, multiple = False):
        if multiple:
            for i in range(len(image)):
                image[i] = self.norm(image[i])
        else:
            # maxval = np.argmax(image)
            # minval = np.argmin(image)
            # delta = (maxval-minval) if maxval-minval > 255  else 255
            # image = image * (255 / delta)
            image = np.around(image, 0)
        return image


    def generate_images(self, n, train_dataset_element, show_result = False, return_original = False):
        """
        function generate N random images from modify kaggle train element
        example:
        ------------------------------------------------
        data_set = load_train_dataset('files/dev_train.csv')
        generate_images(20, data_set[5], show_result=True)
        -------------------------------------------------
        """

        train_image, train_value = self.split_dataset_element(train_dataset_element)

        if show_result:
            self.show_dataset_images(train_image, train_value)

        train_image = train_image.reshape(1, 28, 28)
        x_train_ml = np.expand_dims(train_image, axis=3)

        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.10,
            width_shift_range=0.10,
            height_shift_range=0.10,
            #brightness_range = [0.9, 1.2]
        )

        train_batches = datagen.flow(x_train_ml)
        new_images = np.array([next(train_batches)[0] for i in range(n)])
        new_images = np.squeeze(new_images, axis=3)

        gen_images = train_image
        new_images = self.norm(new_images, True);
        gen_images = np.append(gen_images, new_images, axis=0)

        for i in new_images:
            if show_result:
                self.show_dataset_images(i, train_value)

        train_values = np.zeros(len(gen_images)).reshape(len(gen_images), 1) + train_value
        gen_images = gen_images.reshape(len(gen_images), 784)
        gen_images = np.append(train_values, gen_images, axis=1)

        if not return_original:
            gen_images = np.delete(gen_images, [0], axis=0)

        return np.around(gen_images)


    """
    procedure save dataset to csv width show progress
    example:
    ------------------------------------------------
    save_data('files/example.csv', data_set)
    -------------------------------------------------
    """
    def save_data(self, file_name, dataset, show_progress = False):
        dataset = dataset.astype(np.int)
        if show_progress:
            print(self.messages['SAVE_DATA_TO_DISK'])
        with open(file_name, 'w', newline='') as f:
            wr = csv.writer(f, delimiter=",")
            k = 0
            for i in dataset:
                k += 1
                if show_progress:
                    print("\rSaving {} of {} ...   ".format(k, len(dataset)), end="")
                wr.writerow(i)


    """
    procedure generate new batches of dataset
    example:
    ------------------------------------------------
    create_train_batches(data_set, 500, 5)
    -------------------------------------------------
    """
    def create_train_batches(self, data_set, batch_size = 500, children_image_count = 5):
        print("Start prepare dataset")

        operation_len = len(data_set)
        tmp_file = self.batch_file_mask

        k = 0
        batch_count = 0
        batch_elements = 0

        for i in data_set:
            if k == 0:
                batch_data_set = np.array([i])
            k += 1
            print("\rGeneration in progress. Row {} of {} , batches: {} of {}...   ".format(
                k, operation_len, batch_count,  round(operation_len / batch_size)), end="")

            new_images = self.generate_images(children_image_count, i, show_result=False, return_original=False)

            #data_set = np.append(data_set, new_images, axis=0)
            batch_data_set = np.append(batch_data_set, new_images, axis=0)
            batch_elements += 1

            if batch_elements > batch_size:
                batch_elements = 0
                batch_count += 1
                batch_data_set = np.delete(batch_data_set, [0], axis=0)
                self.save_data(tmp_file.format(batch_count), batch_data_set)
                batch_data_set = np.array([i])

        if len(batch_data_set) > 0:
            batch_count += 1
            self.save_data(tmp_file.format(batch_count), batch_data_set)


    def create_dataset_from_batches(self, batch_counts, postfix=''):
        print("Loading kagle dataset data ...   ")
        data_set = self.load_train_dataset()
        print("Loading kagle dataset data succesfull")
        tmp_file = self.batch_file_mask
        for i in range(1, batch_counts + 1):
            batch_data = self.load_train_dataset(True, i)
            data_set = np.append(data_set, batch_data, axis=0)
            print("Prepare {} of {} batches ...   Rows in dataset: {}".format(i, batch_counts, len(data_set)))
        print("End prepare")
        print("Start shuffle")
        random.shuffle(data_set)
        print("End shuffle")
        print("Saving dataset")
        file_name = self.target_file_mask.format(postfix)
        self.save_data(file_name, data_set, True)


    def show_dataset(self, dataset):
        x_train, y_train = self.split_dataset(dataset)
        self.show_dataset_images(x_train, y_train, True)
