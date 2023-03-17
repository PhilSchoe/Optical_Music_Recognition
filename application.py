import os
import cv2 as cv
import numpy as np
from omr_model import OMRModel


def run():
    print("Running Application")

    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    dataset_path = "E:/Datasets/primusCalvoRizoAppliedSciences2018/package_aa"
    directory_entries = os.listdir(dataset_path)
    batch_size = 10

    # Prepare input data

    input_image_batch = prepare_input_images(dataset_path, directory_entries, batch_size)

    # Prepare output data

    word_to_int = get_word_to_int_dictionary()
    output_label_batch = prepare_output_labels(dataset_path, directory_entries, word_to_int, batch_size)

    # TODO: Split data into training and validation sets?

    # TODO: Create dataset objects?

    # Build model

    vocabulary_size = len(word_to_int)
    OMRModel.build_model(input_image_batch.shape[2], input_image_batch.shape[1], vocabulary_size)

    print("Application Stopped")


def get_word_to_int_dictionary():
    vocabulary_file = open("vocabulary_semantic.txt", "r")
    vocabulary_list = vocabulary_file.read().splitlines()

    word_to_int = {}

    for word in vocabulary_list:
        if word_to_int.get(word) is None:
            index = len(word_to_int)
            word_to_int[word] = index

    vocabulary_file.close()

    return word_to_int


def prepare_output_labels(dataset_path, directory_entries, word_to_int, batch_size):
    batch_label_list = []

    for i in range(batch_size):
        entry_name = directory_entries[i]
        entry_path = dataset_path + "/" + entry_name

        label_filepath = entry_path + "/" + entry_name + ".semantic"
        label_file = open(label_filepath, "r")
        label_list = label_file.read().strip().split("\t")
        label_file.close()

        batch_label_list.append([word_to_int[label] for label in label_list])

    # Test to find out what sparse_tuple_from function from tf-end-to-end gitHub does
    # targets = sparse_tuple_from(batch_label_list)

    return batch_label_list


def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def prepare_input_images(dataset_path, directory_entries, batch_size):
    # Load and resize batch of images

    image_height = 180
    image_list = []

    for i in range(batch_size):
        entry_name = directory_entries[i]
        entry_path = dataset_path + "/" + entry_name

        image_filepath = entry_path + "/" + entry_name + ".png"
        image = load_and_resize_image(image_filepath, image_height)

        """
        cv.imshow("First Image", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        """

        image_list.append(image / 255.0)

    # Make all images the same dimension

    max_width = max(image.shape[1] for image in image_list)
    print("Max image width is: ", max_width)

    channel_number = 1
    padding_value = 0

    batch_image_list = np.ones(shape=[batch_size, image_height, max_width, channel_number],
                               dtype=np.float32) * padding_value

    for i, image in enumerate(image_list):
        batch_image_list[i, 0:image.shape[0], 0:image.shape[1], 0] = image[:, :, 0]

        """
        cv.imshow("batch_image", batch_image_list[i])
        cv.waitKey(0)
        cv.destroyAllWindows()
        """

    return batch_image_list


def load_and_resize_image(image_filepath, image_height):
    image = cv.imread(image_filepath)

    # old_width / old_height * new_height
    image_width = int(image.shape[1] / image.shape[0] * image_height)
    resized_image = cv.resize(image, (image_width, image_height))

    return resized_image
