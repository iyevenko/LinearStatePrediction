import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    shape = (4, 4)
    h, w = shape

    num_examples = 1000

    y = tf.random.uniform((num_examples, 2), minval=0, maxval=h, dtype=tf.int32)
    x = tf.map_fn(lambda coords: tf.reshape(tf.one_hot(tf.tensordot(coords, [h, 1], 1), h*w), shape), y, dtype=tf.float32)
    y = tf.cast(y, tf.float32)

    def Argmax2D(h, w):
        # A hard argmax wouldn't be differentiable so a soft argmax has to be used
        # Soft argmax implemented by taking dot product of softmax and tf.range
        # https://stackoverflow.com/questions/46926809/getting-around-tf-argmax-which-is-not-differentiable
        # Beta=10 provides accurate enough results while still allowing gradient to pass through
        # Can crank up beta during inference
        @tf.function
        def soft_amax(x, beta=10):
            x_range = tf.range(x.shape.as_list()[-1], dtype=tf.float32)
            return tf.reduce_sum(tf.nn.softmax(x * beta, axis=-1) * x_range, axis=-1)

        @tf.function
        def amax2d(inputs):
            flat = tf.keras.layers.Flatten()(inputs)
            amax = soft_amax(flat)
            x = amax % w
            y = amax // h
            return tf.stack([x, y])

        return tf.keras.layers.Lambda(amax2d)

    for i in range(5):
        x_ = x[i:i+1]
        y_ = y[i:i+1]

        with tf.GradientTape() as tape:
            tape.watch(x_)
            result = Argmax2D(h, w)(x_)
            loss = tf.keras.losses.MSE(y_, result)

        grads = tape.gradient(loss, x_)
        print(x_)
        print(result)
        print(grads)

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Conv2D(1, (3, 3), activation='relu'),
    #     Argmax2D(h, w)
    # ])
    #
    # model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    # history = model.fit(x, y, batch_size=128, epochs=50)
    #
    # plt.plot(history.history['loss'])
    # plt.show()
    #
    # # print(model.layers[-1].kernel)
    #
    # y_test = tf.reshape(tf.one_hot(2*h+2, depth=h*w),(1, h, w))
    # print(y_test[0, 2, 2])
    # pred = model.predict(y_test)
    #
    # print(pred)
