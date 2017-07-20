import mxnet as mx


def mlp_layer(input_layer, n_hidden, activation=None, BN=False):

    """
    A MLP layer with activation layer and BN
    :param input_layer: input sym
    :param n_hidden: # of hidden neurons
    :param activation: the activation function
    :return: the symbol as output
    """

    # get a FC layer
    l = mx.sym.FullyConnected(data=input_layer, num_hidden=n_hidden)
    # get activation, it can be relu, sigmoid, tanh, softrelu or none
    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if BN:
        l = mx.sym.BatchNorm(l)
    return l


def get_mlp_sym():

    """
    :return: the mlp symbol
    """

    data = mx.sym.Variable("data")
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data_f = mx.sym.flatten(data=data)

    # Your Design
    l = mlp_layer(input_layer=data_f, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)

    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return mlp


def conv_layer(input_layer,
               kernel_size=(3,3),
               kernel_num=32,
               stride=(1,1),
               pad=(0, 0),
               activation='relu',
               pooling=True,
               pool_type="max",
               pool_size=(2,2),
               pool_stride=(2,2),
               BN=True):
    """
    :return: a single convolution layer symbol
    """
    l = mx.sym.Convolution(data=input_layer, kernel=kernel_size, num_filter=kernel_num,stride=stride, pad=pad)
    if BN:
        l = mx.sym.BatchNorm(l)
    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if pooling:
        l = mx.sym.Pooling(data=l, pool_type=pool_type, kernel=pool_size, stride=pool_stride)
    return l


def get_conv_sym():

    """
    :return: symbol of a convolutional neural network
    """
    data = mx.sym.Variable("data")
    # todo: design the CNN architecture
    # How deep the network do you want? like 4 or 5
    # How wide the network do you want? like 32/64/128 kernels per layer
    # How is the convolution like? Normal CNN? Inception Module? VGG like?
    l = conv_layer(input_layer=data,kernel_num=64, activation="relu")
    l = conv_layer(input_layer=l,kernel_num=128, activation="relu")
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    cnn = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return cnn
