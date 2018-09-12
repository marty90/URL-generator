#!/usr/bin/python3

# Imports
from __future__ import print_function, division
from tensorflow.contrib.keras.api.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.contrib.keras.api.keras.layers import BatchNormalization, Activation
from tensorflow.contrib.keras.api.keras.layers import LeakyReLU, LSTM, SimpleRNN, GRU, Bidirectional
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.optimizers import Adam, RMSprop
from tensorflow.contrib.keras.api.keras.backend import expand_dims
from tensorflow.contrib.keras.api.keras.layers import UpSampling1D, Conv1D, LocallyConnected1D
from tensorflow.contrib.keras.api.keras.activations import softmax
import tensorflow as tf
import argparse
import string
import sys
import numpy as np

# Disable Warnings
tf.logging.set_verbosity(tf.logging.ERROR)

# Constants
save_interval = 100
learning_rate=0.0002

# Arguments
parser = argparse.ArgumentParser(description='URL GAN')
parser.add_argument('--input_data', metavar='input_data', type=str)
parser.add_argument('--url_len', metavar='url_len', type=int, default=200)  
parser.add_argument('--batch_size', metavar='batch_size', type=int, default=128)  
parser.add_argument('--print_size', metavar='print_size', type=int, default=500)  
parser.add_argument('--epochs', metavar='epochs', type=int, default=1000)  
parser.add_argument('--noise_shape', metavar='noise_shape', type=int, default=8)  
parser.add_argument('--generator_layers', metavar='generator_layers', type=str, default="8:8")  
parser.add_argument('--discriminator_layers', metavar='discriminator_layers', type=str, default="8:4:2")  
parser.add_argument('--generator_activation', metavar='generator_activation', type=str, default="tanh")  
parser.add_argument('--discriminator_activation', metavar='discriminator_activation', type=str, default="tanh")  
parser.add_argument('--dropout_value', metavar='dropout_value', type=float, default=0.8)  
parser.add_argument('--discriminator_savefile', metavar='discriminator_savefile', type=str, default="/dev/null")
parser.add_argument('--generator_savefile', metavar='generator_savefile', type=str, default="/dev/null")  
parser.add_argument('--generated_savefile', metavar='generated_savefile', type=str, default="/dev/null")  
globals().update(vars(parser.parse_args()))

# Create noise_shape
noise_shape=(noise_shape,)

# Define Alphabet
alphabet = string.ascii_lowercase + string.digits + "/:._-()=;?&" # MUST BE EVEN
dictionary_size = len(alphabet) + 1
url_shape = (url_len, dictionary_size)


def main():

    # Define Functions
    build_generator = build_generator_dense
    build_discriminator = build_discriminator_dense

    # Build dictionary
    dictionary = {}
    reverse_dictionary = {}
    for i, c in enumerate(alphabet):
        dictionary[c]=i+1
        reverse_dictionary[i+1]=c

    # Build Oprimizer
    optimizer = Adam(learning_rate, 0.5)

    # Build and compile the discriminator
    print ("*** BUILDING DISCRIMINATOR ***")
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', 
        optimizer=optimizer,
        metrics=['accuracy'])

    # Build and compile the generator
    print ("*** BUILDING GENERATOR ***")
    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    # The generator takes noise as input and generated samples
    z = Input(shape=noise_shape)
    gen = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The valid takes generated samples as input and determines validity
    valid = discriminator(gen)

    # The combined model  (stacked generator and discriminator) takes
    # noise as input => generates samples => determines validity 
    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    # Load the dataset
    data = []   
    for line in open(input_data,"r").read().splitlines():
        this_sample=np.zeros(url_shape)

        line = line.lower()
        if len ( set(line) - set(alphabet)) == 0 and len(line) < url_len:
            for i, position in enumerate(this_sample):
                this_sample[i][0]=1.0

            for i, char in enumerate(line):
                this_sample[i][0]=0.0
                this_sample[i][dictionary[char]]=1.0
            data.append(this_sample)
        else:
            print("Uncompatible line:",  line)
        
    print("Data ready. Lines:", len(data))
    X_train = np.array(data)
    print ("Array Shape:", X_train.shape)
    half_batch = int(batch_size / 2)

    # Start Training
    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of data
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        samples = X_train[idx]
        noise_batch_shape = (half_batch,) + noise_shape
        noise = np.random.normal(0, 1, noise_batch_shape)

        # Generate a half batch of new data
        gens = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(samples, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gens, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


        # ---------------------
        #  Train Generator
        # ---------------------

        noise_batch_shape = (batch_size,) + noise_shape
        noise = np.random.normal(0, 1, noise_batch_shape)

        # The generator wants the discriminator to label the generated samples as valid (ones)
        valid_y = np.array([1] * batch_size)

        # Train the generator
        g_loss = combined.train_on_batch(noise, valid_y)

        # Plot the progress
        print ("%d [D loss: %0.3f, acc.: %0.3f%%] [G loss: %0.3f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval, print some examples
        if epoch % save_interval == 0:
            generated_samples=[]
            r, c = 5, 5
            noise_batch_shape = (print_size,) + noise_shape
            noise = np.random.normal(0, 1, noise_batch_shape)
            gens = generator.predict(noise)

            for url in gens:
                this_url_gen = ""
                for position in url:
                    this_index = np.argmax(position)
                    if this_index != 0:
                        this_url_gen += reverse_dictionary[this_index]

                print(this_url_gen)
                generated_samples.append(this_url_gen)

    # Save networks
    discriminator.save(discriminator_savefile)
    generator.save(generator_savefile)

    # Save Samples
    fo = open(generated_savefile, "w")
    for url in generated_samples:
        print (url, file=fo)
    fo.close()


def build_generator_dense():
    
    model = Sequential()

    # Add arbitrary layers
    first = True
    for size in generator_layers.split(":"):
        size = int(size)
        if first:
            model.add(Dense(size, input_shape=noise_shape, activation=generator_activation))
        else:
            model.add(Dense(size, activation=generator_activation))

        model.add(Dropout(dropout_value))
        first = False

    # Add the final layer
    model.add(Dense(  np.prod(url_shape) , activation="tanh"))
    model.add(Dropout(dropout_value))
    model.add(Reshape(url_shape))
    model.summary()

    # Build the model
    noise = Input(shape=noise_shape)
    gen = model(noise)

    return Model(noise, gen)

def build_discriminator_dense():

    model = Sequential()
    model.add(Flatten(input_shape=url_shape))

    # Add arbitrary layers
    for size in discriminator_layers.split(":"):
        size = int(size)
        model.add(Dense(size, activation=discriminator_activation))
        model.add(Dropout(dropout_value))

    # Add the final layer, with a single output
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # Build the model
    gen = Input(shape=url_shape)
    validity = model(gen)
    return Model(gen, validity)


# Main
if __name__ == '__main__':
    main()


