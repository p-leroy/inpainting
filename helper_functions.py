import cv2
import keras
import numpy as np
import wandb

class CreateAugment(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, X, y, batch_size=32, dim=(32, 32), n_channels=3, shuffle=True):
        # Initialization
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor((len(self.X)) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        # Generate data
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, idxs):
        # X_batch is a matrix of masked images used as input
        X_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        # y_batch is a matrix of original images used for computing error from reconstructed image
        y_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))

        # Iterate through random indexes
        for i, idx in enumerate(idxs):
            image_copy = self.X[idx].copy()

            # Get mask associated to that image
            masked_image = self.__createMask(image_copy)

            # Append and scale down
            X_batch[i, ] = masked_image / 255
            y_batch[i, ] = self.y[idx] / 255

        return X_batch, y_batch

    def __createMask(selfself, img):
        # Prepare masking matrix
        mask = np.full((32, 32, 3), 255, np.uint8)
        for _ in range(np.random.randint(1, 10)):
            # Get random x locations to start line
            x1, x2 = np.random.randint(1, 32), np.random.randint(1, 32)
            # Get random y location to start line
            y1, y2, = np.random.randint(1, 32), np.random.randint(1, 32)
            # Get random thickness of the line drawn
            thickness = np.random.randint(1, 3)
            # Draw black line on the white mask
            cv2.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        # Mask the image
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image


def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))


class PredictionLogger(keras.callbacks.Callback):
    def __init__(self):
        super(PredictionLogger, self).__init__()

    # The callback will be executed after an epoch is completed
    def on_epoch_end(self, logs, epoch):
        # Pick a batch and sample the masked images, masks, and the labels
        sample_idx = 54
        [masked_images, masks], sample_labels = testgen[sample_idx]

        # Initialize empty lists store intermediate results
        m_images = []
        binary_masks = []
        predictions = []
        labels = []

        # Iterate over the batch
        for i in range(32):
            # Our inpainting model accepts masked images and masks as its inputs then use perform inference
            inputs = [B]
            inpainted_image = model.predict(inputs)

            # Append the results to the respective lists
            m_images.append(masked_images[i])
            binary_masks.append(masks[i])
            predictions.append(inpainted_image.reshape(inpainted_image.shape[1:]))
            labels.append(sample_labels[i])

        # Log the results on wandb run page

        wandb.log({"masked_images": [wandb.Image(m_image) for m_image in m_images]})
        wandb.log({"masks": [wandb.Image(mask) for mask in binary_masks]})
        wandb.log({"predictions": [wandb.Image(inpainted_image) for inpainted_image in predictions]})
        wandb.log({"labels": [wandb.Image(label) for label in labels]})


class InpaintingModel:
    """
    Build UNET like model for image inpainting task.
    """
    def prepare_model(self, input_size=(32, 32, 3)):
        inputs = keras.layers.Input(input_size)

        conv1, pool1 = self.__ConvBlock(32, (3, 3), (2, 2), 'relu', 'same', inputs)
        conv2, pool2 = self.__ConvBlock(64, (3, 3), (2, 2), 'relu', 'same', pool1)
        conv3, pool3 = self.__ConvBlock(128, (3, 3), (2, 2), 'relu', 'same', pool2)
        conv4, pool4 = self.__ConvBlock(256, (3, 3), (2, 2), 'relu', 'same', pool3)

        conv5, up6 = self.__UpConvBlock(512, 256, (3, 3), (2, 2), (2, 2), 'relu', 'same', pool4, conv4)
        conv6, up7 = self.__UpConvBlock(256, 128, (3, 3), (2, 2), (2, 2), 'relu', 'same', up6, conv3)
        conv7, up8 = self.__UpConvBlock(128, 64, (3, 3), (2, 2), (2, 2), 'relu', 'same', up7, conv2)
        conv8, up9 = self.__UpConvBlock(64, 32, (3, 3), (2, 2), (2, 2), 'relu', 'same', up8, conv1)

        conv9 = self.__ConvBlock(32, (3, 3), (2, 2), 'relu', 'same', up9, False)

        outputs = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)

        return keras.models.Model(inputs=[inputs], outputs=[outputs])

    def __ConvBlock(self, filters, kernel_size, pool_size, activation, padding, connecting_layer, pool_layer=True):
        conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   activation=activation, padding=padding)(connecting_layer)
        conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   activation=activation, padding=padding)(conv)
        if pool_layer:
            pool = keras.layers.MaxPooling2D(pool_size)(conv)
            return conv, pool
        else:
            return conv

    def __UpConvBlock(self, filters, up_filters, kernel_size, up_kernel_size, up_stride, activation, padding,
                      connecting_layer, shared_layer):
        conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   activation=activation, padding=padding)(connecting_layer)
        conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   activation=activation, padding=padding)(conv)
        up = keras.layers.Conv2DTranspose(filters=up_filters, kernel_size=up_kernel_size,
                                          strides=up_stride, padding=padding)(conv)
        up = keras.layers.concatenate([up, shared_layer], axis=3)

        return conv, up
