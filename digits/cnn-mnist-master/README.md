# cnn-mnist
This is a Python3 / Tensorflow implementation of a convolutional network model for MNIST dataset.

## Setup

To run this code you need the following:

- Python3.3+
- Numpy
- TensorFlow

## Training the model

Use the `main.py` script to train the model. To train the default model on [MNIST](http://yann.lecun.com/exdb/mnist/) simply use:

```
python3 main.py
```

This code will automatically output the predictions of <em>test-image</em>, which is in the `mnist_data/` folder.<br/>
<br/>
To use the pretrained model append the `-l` flag after the command.<br/>
<br/>
To only predict the test images using the pretrained model append the `-l -e 0` flag after the command.


## Performance

Train/Validation: 60000/5000<br/>
<br/>
Train the model for 30 epochs.

| Data | Accuarrcy |
| :---: |:---:|
| train | 0.9989 |
| validation | 0.9932 |
