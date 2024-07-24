from autoencoder import Autoencoder
import os
import numpy as np


#global variables
learning_Rate = 0.0001
batch_Size = 5
num_Epochs = 200
spectrogramSaveDir = ("/Users/antoinenguyen/Downloads/data/Parsed_Pump_noise_spectrogram/")


def loadFSDD(spectrogramPath):
    #eventually fill it up with all the spectrogram's
    x_train = []
    for root, _, fileNames in os.walk(spectrogramPath):
        for fileName in fileNames:
                # grabs the file path for each spectrogram
                filePath = os.path.join(root, fileName)
                # loads the spectrogram because it ends in .npy
                spectrogram = np.load(filePath,allow_pickle=True)  # first dimension is # of bins second dimension is # of frames
                x_train.append(spectrogram)

    # cant return immediately because autoencoder expects 3 dimensions but spectrogram is held as 2 dimensions
    x_train = np.array(x_train)
    x_train = x_train[None,...] # -> (number of samples in data set, number of bins, number of frames, 1)
    return x_train





def train(x_train, learning_Rate, batch_Size, num_Epochs):
    autoencoder = Autoencoder(
        input_shape=(325, 256, 87),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2,
    )
    autoencoder.summary()
    autoencoder.compile(learning_Rate)
    autoencoder.train(x_train,batch_Size,num_Epochs)
    return autoencoder


if __name__ == "__main__":
    x_train = loadFSDD(spectrogramSaveDir)
    autoencoder = train(x_train, learning_Rate, batch_Size, num_Epochs)
    autoencoder.save("model")

autoencoder = Autoencoder.load("model")
autoencoder.summary()




