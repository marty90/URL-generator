# URL generator

This piece of code use a Generative Adversarial Network to train two models for (i) generating new URLs, and (ii) distinguishing URLs belonging to the training class. It uses the Keras library to model neural networks. Both generator and discriminator are multilayer perceptrons.

It needs a (potentially large) set of URLs as input. Those URLs must be homogeneous in order to let the models to succesfully catch the distribution of characters within them. For example, you may provide it a set of URLs for a video streaming service, or an AntiVirus update.

Running the code allows you to train a generator and a discriminator. Those models can be saved to disk in Keras format, and reused in any further piece of code. A sample of generated URLs can be saved as well.

For any information or request write to [martino.trevisan@polito.it](mailto:martino.trevisan@polito.it).

## Prerequisites
You need Python3 with Tensorflow and numpy installed.

## Usage
Simply run the script. The only needed argument is `input_data`.
```
./train_gan.py --input_data <input_data> [arguments...]
```

Several arguments are allowed:
  * `input_data`: text file containing input URLs, one per line.
  * `url_len`: Maximum allowed URL length allowed by the GAN. Longer URLs will be discarded.
  * `batch_size`: Batch size used in training.
  * `print_size`: Number of example generated URLs to print periodically on screen, and on disk at the end.
  * `epochs`: Duration of training in epochs.
  * `noise_shape`: Input noise shape, integer.
  * `generator_layers`: Hidden layers of generator model, separated by `:`. Example: `4:8:16`.
  * `discriminator_layers`:  Hidden layers of discriminator model, separated by `:`. Example: `8:4:2`.
  * `generator_activation`: Generator activation function, using Keras names. Example: `tanh`, `relu`...
  * `discriminator_activation`: Discriminator activation function, using Keras names. Example: `tanh`, `relu`...
  * `dropout_value`: Dropout value to apply to hidden layers.
  * `discriminator_savefile`: File where Keras discriminator model is saved on disk.
  * `generator_savefile`:  File where Keras generator model is saved on disk.
  * `generated_savefile`:  File where example generated URLs are saved (`print_size` examples are saved).
  
  Enjoy ;)
