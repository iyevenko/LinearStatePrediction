import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import InputLayer, Conv2D, Flatten, Dense, Lambda, LayerNormalization, Reshape, MaxPool2D


class LinearDynamicsInitializer(tf.keras.initializers.Initializer):

    def __init__(self):
        super().__init__()

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.constant([[-1, 0],[0, -1],[2, 0],[0, 2]], dtype=tf.float32)


def conv_model(img_shape, linear_hint=False):
    encoder = tf.keras.Sequential([
        InputLayer((img_shape)),
        Conv2D(1, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        MaxPool2D(),
        Conv2D(1, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        MaxPool2D(),
        Conv2D(1, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        MaxPool2D(),
        # Conv2D(1, (1, 1), strides=(1, 1), padding='same', activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        # Dense(64, activation='relu'),
        # Dense(64, activation='relu'),
        # Dense(64, activation='relu'),
        Dense(2)
    ])
    predictor = tf.keras.Sequential([
        InputLayer((4,)),
        Dense(2, kernel_initializer=LinearDynamicsInitializer() if linear_hint else 'glorot_uniform')
    ])
    if linear_hint:
        predictor.trainable = False

    return encoder, predictor


def Argmax2D(input_shape, beta=10):
    # A hard argmax wouldn't be differentiable so a soft argmax has to be used
    # Soft argmax is implemented by taking the dot product of softmax and tf.range
    # then
    @tf.function
    def soft_amax(x, beta=10):
        # x => (batch_size, n+1, c)
        n = x.shape.as_list()[1] - 1
        sm = tf.nn.softmax(x * beta, axis=1)
        x_range = tf.range(n + 1, dtype=tf.float32)
        x_range = tf.reshape(x_range, sm.shape[1:])
        amax = tf.reduce_sum(sm * x_range, axis=1)
        eb = tf.exp(tf.cast(beta, tf.float32))
        idx = ((eb + n) * amax - n*(n + 1)/2) / (eb - 1)
        # Normalize index to [0, 1]
        idx = idx / (n+1)
        return idx

    h, w, _ = input_shape

    @tf.function
    def amax2d(x):
        # x => (batch_size, h, w, c)
        x_max = tf.reduce_max(x, axis=1)
        y_max = tf.reduce_max(x, axis=2)

        # Normalize Inputs
        x_mu = tf.reduce_mean(x_max)
        x_std = tf.math.reduce_std(x_max)
        x_norm = (x_max-x_mu)/x_std

        y_mu = tf.reduce_mean(y_max)
        y_std = tf.math.reduce_std(y_max)
        y_norm = (y_max-y_mu)/y_std

        x_samax = soft_amax(x_norm, beta=beta)
        y_samax = soft_amax(y_norm, beta=beta)

        return tf.stack([x_samax, y_samax], axis=1)

    return Lambda(amax2d)


def conv_argmax_model(img_shape, beta=10):
    encoder = tf.keras.Sequential([
        InputLayer((img_shape)),
        Conv2D(1, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        Argmax2D(img_shape, beta=beta),
        Reshape((2, ))
    ])
    predictor = tf.keras.Sequential([
        InputLayer((4,)),
        Dense(2)
    ])

    return encoder, predictor

if __name__ == '__main__':
    shape = (64, 64, 1)
    h, w, c = shape
    beta = 1

    x_, y_ = (63, 63)
    y = tf.constant([[x_, y_]])

    encoder, predictor = conv_argmax_model(shape, beta=beta)
    x = tf.reshape(tf.one_hot(h*y_+x_, depth=h*w), (1, h, w, 1))
    with tf.GradientTape() as tape:
        tape.watch(x)
        out = encoder(x)
        loss = tf.keras.losses.MSE(y, out)

    grads = tape.gradient(loss, x)
    print(f'Coordinates: {out}')
    print(f'Mean Gradient: {tf.reduce_mean(grads)}')
    print(f'Loss {loss}')

    print(predictor(tf.concat([out, out], axis=-1)).shape)

