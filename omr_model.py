import tensorflow as tf
from tensorflow import keras
from keras import layers


class OMRModel:

    @staticmethod
    def build_model(image_width, image_height):
        model_input = layers.Input(shape=(image_width, image_height, 1), name="input_image", dtype="float32")

        model_intermediate, new_shape = OMRModel.__build_convolutional_block(model_input)
        model_intermediate = layers.Reshape(target_shape=new_shape, name="reshape")(model_intermediate)

        model_output = model_intermediate
        model = keras.models.Model(model_input, model_output, name="OMR_Model")

        optimizer = keras.optimizers.Adam()

        model.compile(optimizer=optimizer)

        model.summary()

    @staticmethod
    def __build_convolutional_block(block_input):

        # First convolutional layer: 32 filters; kernel size: 3 x 3; Batch Normalization; Leaky ReLu activation function
        # MaxPooling layer with window size: 2 x 2
        # TODO: What about kernel_initializer?
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", name="conv_0")(block_input)
        x = layers.BatchNormalization(name="conv_0_bn")(x)
        x = layers.LeakyReLU(alpha=0.2, name="conv_0_leaky_relu")(x)
        x = layers.MaxPooling2D((2, 2), name="max_pool_0")(x)

        width_reduction = 2
        height_reduction = 2

        # Second convolutional layer: 64 filters; kernel size: 3 x 3; Batch Normalization;
        # Leaky ReLu activation function
        # MaxPooling layer with window size: 2 x 2

        x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="conv_1")(x)
        x = layers.BatchNormalization(name="conv_1_bn")(x)
        x = layers.LeakyReLU(alpha=0.2, name="conv_1_leaky_relu")(x)
        x = layers.MaxPooling2D((2, 2), name="max_pool_1")(x)

        width_reduction *= 2
        height_reduction *= 2

        # Third convolutional layer: 128 filters; kernel size: 3 x 3; Batch Normalization;
        # Leaky ReLu activation function
        # MaxPooling layer with window size: 2 x 2

        x = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", name="conv_2")(x)
        x = layers.BatchNormalization(name="conv_2_bn")(x)
        x = layers.LeakyReLU(alpha=0.2, name="conv_2_leaky_relu")(x)
        x = layers.MaxPooling2D((2, 2), name="max_pool_2")(x)

        width_reduction *= 2
        height_reduction *= 2

        # Fourth convolutional layer: 256 filters; kernel size: 3 x 3; Batch Normalization;
        # Leaky ReLu activation function
        # MaxPooling layer with window size: 2 x 2

        last_filter_size = 256;

        x = layers.Conv2D(filters=last_filter_size, kernel_size=(3, 3), padding="same", name="conv_3")(x)
        x = layers.BatchNormalization(name="conv_3_bn")(x)
        x = layers.LeakyReLU(alpha=0.2, name="conv_3_leaky_relu")(x)
        x = layers.MaxPooling2D((2, 2), name="max_pool_3")(x)

        width_reduction *= 2
        height_reduction *= 2

        new_width = int(block_input.shape[1]) // width_reduction
        new_height = (int(block_input.shape[2]) // height_reduction) * last_filter_size
        new_shape = (new_width, new_height)

        return x, new_shape
