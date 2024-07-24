from tensorflow.keras import Model, backend
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import os
import pickle

class Autoencoder:
    def __init__(self, input_shape,conv_filters, conv_kernels, conv_strides, latent_space_dim):
        #width by height by channel (1 channel)
        self.input_shape = input_shape

        #going to a be list for amount of filter per layer
        self.conv_filters = conv_filters
        #kernel size (3x3 or NxN)
        self.conv_kernels = conv_kernels
        #number of strides
        self.conv_strides = conv_strides

        #integer for bottleneck (amount of axes)
        self.latent_space_dim = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None

        #number of convolutional layers
        self._num_conv_layers = len(conv_filters)

        #intiate required variables for storage
        self._shape_before_bottleneck = None
        self._model_input = None

        #builds the encoder/decoder/model
        self._build()



    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        Mean_Square_Error_Loss = MeanSquaredError()
        self.model.compile(optimizer, loss=Mean_Square_Error_Loss)

    def train(self, x_train, batch_size, num_epochs):
        #fit is native to keras api (second argument is the expected output (we get back what we put in))
        self.model.fit(x_train, x_train, batch_size=batch_size, epochs=num_epochs,shuffle=True)

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._reshape_layer(dense_layer)
        conv_transpose_layer = self._add_conv_transpose_layers(reshape_layer)
        output = self._decoder_output(conv_transpose_layer)
        self.decoder = Model(decoder_input, output, name="decoder")

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        #output of the encoder (compressed image)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _build_autoencoder(self):
        model_input = self._model_input
        # output of the entire class (output of the autoencoder)
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output)


    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    #create all convolutional blocks in an encoder
    def _add_conv_layers(self, encoder_input):
        #x is essentially a graph of layers
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        # adds a convolutional block to a graph of layers consisting of conv 2d + Relu + batch normalization.
        layer_num = layer_index + 1
        conv_layer = Conv2D(
            #feteching number of filters for a specific layer
            filters= self.conv_filters[layer_index],
            kernel_size= self.conv_kernels[layer_index],
            strides= self.conv_strides[layer_index],
            padding= "same",
            name =f"encoder_conv_layer_{layer_num}"
        )
        #getting a keras layer and applying it to a graph of layers (x)
        x = conv_layer(x)
        #applying a ReLU layer onto x
        x = ReLU(name= f"encoder_relu_{layer_num}")(x)
        #batch normalization
        x = BatchNormalization(name= f"encoder_BN_{layer_num}")(x)
        return x

    def _add_bottleneck(self, x):
        #store the shape of data to check decoder (taking only the width height and channel amount)
        self._shape_before_bottleneck = backend.int_shape(x)[1:]
        #flatten data and add bottleneck (dense layer)
        x = Flatten()(x)

        #pass flatten data into a dense layer (output of the encoder layer)
        x = Dense(self.latent_space_dim, name="encoder_output")(x)
        return x

    def _add_decoder_input(self):
        return Input(shape=(self.latent_space_dim, ), name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        #want same number of neurons as before we flatten the data in the bottleneck (shape of bottleneck)
        num_of_neurons = np.prod(self._shape_before_bottleneck) # [1,2,4] -> 8 multiply all the numbers in the 3d array to turn into a scalar
        dense_layer = Dense(num_of_neurons, name="dense_decoder")(decoder_input) #applies it to a graph of layers
        return dense_layer

    # go back to the same dimensionality as the original input
    def _reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    #add conv transpose block
    def _add_conv_transpose_layers(self,x):
#loop through all conv layers in reverse order to mirror the first convolutional layers
        for layer_index in reversed(range(1, self._num_conv_layers)):
            # [0,1,2] -> [2,1,0]
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self,layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(filters=self.conv_filters[layer_index],
                                               kernel_size=self.conv_kernels[layer_index],
                                               strides=self.conv_strides[layer_index],
                                               padding="same",
                                               name=f"decoder_conv_transpose_layer_{layer_num}")
        #updating the graph with the layers (x is the generic graph of layers) in the reverse order
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_BatchNormalization_{layer_num}")(x)
        return x

    def _decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(filters=1, # can interpret spectrogram as grey scale so channel can be 1
                                               kernel_size=self.conv_kernels[0], #using 0 to get first convultional layer data in terms of strides/kernels
                                               strides=self.conv_strides[0],
                                               padding="same",
                                               name=f"decoder_conv_transpose_layer_{self._num_conv_layers}")
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="output_layer")(x)
        return output_layer

    def save(self, save_folder="."):
        self._create_save_folder_if_one_does_not_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def _create_save_folder_if_one_does_not_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            pass

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "my_model.weights.h5")
        self.model.save_weights(save_path)

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, "my_model.weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

