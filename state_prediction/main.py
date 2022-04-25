import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from env import Ball, Grid
from model import conv_model, conv_argmax_model
from schedules import ConstantSchedule, RandomSchedule, DifficultyRampSchedule


def visualize_encodings(encoder, grid):
    print('Plotting Encodings')
    r = grid.ball.r
    encodings = np.empty((grid.h, grid.w, 2))
    for x in range(grid.w):
        for y in range(grid.h):
            ball = Ball(r, x, y, 0, 0)
            grid.reset(ball)
            img = grid.get_grid()
            img = tf.reshape(img, (1, grid.h, grid.w, 1))
            encoding = encoder(img)
            encoding = np.squeeze(encoding)
            encodings[y, x, :] = encoding

    n_dim = encodings.shape[-1]
    fig, axs = plt.subplots(1, n_dim)
    for i, ax in enumerate(axs):
        ax.imshow(encodings[:,:,i], cmap='RdBu')
    plt.show()


def train_models(encoder, predictor, episodes, schedule, visualize=False):
    _, H, W, _ = encoder.input.type_spec.shape
    r = 3
    x, y = (W/2, H/2)
    ddx, ddy = (0, 0)

    dx, dy = schedule.get_vel()

    ball = Ball(r, x, y, dx, dy)
    grid = Grid(W, H, ball)

    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(1e-2)

    history = []
    for i in tqdm(range(episodes)):

        grid_img = grid.show(1)
        img = tf.constant(grid_img, dtype=tf.float32) / 255.
        x0 = tf.reshape(img, (1, H, W, 1))
        x1 = tf.reshape(img, (1, H, W, 1))

        prev_imgs = [x0, x1]

        done = False
        while not done:
            grid_img = grid.show(1)
            x0, x1 = prev_imgs

            with tf.GradientTape(persistent=True) as tape:
                img = tf.constant(grid_img, dtype=tf.float32) / 255.
                img = tf.reshape(img, (1, H, W, 1))
                s0 = encoder(x0)
                s1 = encoder(x1)
                s2 = encoder(img)
                s2_pred = predictor(tf.concat([s0, s1], -1))
                loss = loss_fn(s2, s2_pred)
            print(s2.numpy())

            encoder_grads = tape.gradient(loss, encoder.trainable_variables)
            predictor_grads = tape.gradient(loss, predictor.trainable_variables)

            optimizer.apply_gradients(zip(encoder_grads, encoder.trainable_variables))
            optimizer.apply_gradients(zip(predictor_grads, predictor.trainable_variables))

            prev_imgs = [x1, img]

            done = grid.update(ddx, ddy)

            history.append(float(loss))

        dx, dy = schedule.get_vel()
        ball = Ball(r, x, y, dx, dy)
        grid.reset(ball)

    cv2.destroyAllWindows()

    if visualize:
        visualize_encodings(encoder, grid)

    print([tf.squeeze(w) for w in encoder.trainable_variables if 'conv' in w.name])
    print(predictor.weights)
    plt.plot(history)
    plt.show()


if __name__ == '__main__':
    constant_episodes = 10
    ramp_episodes = 0
    episodes = constant_episodes + ramp_episodes

    # schedule = ConstantSchedule(1)
    # schedule = DifficultyRampSchedule(constant_episodes, ramp_episodes, 1)
    schedule = RandomSchedule(1)

    # encoder, predictor = conv_model((64, 64, 1), linear_hint=False)
    # encoder.summary()
    encoder, predictor = conv_argmax_model((64, 64, 1), beta=10)
    conv_init = encoder.layers[0].weights[0][:,:,0,0]
    train_models(encoder, predictor, episodes, schedule, visualize=True)

    print(conv_init)
    plt.title('OLD:')
    plt.imshow(conv_init)
    plt.show()

    conv_kernel = encoder.layers[0].weights[0][:,:,0,0]
    print(conv_kernel)
    plt.title('NEW:')
    plt.imshow(conv_kernel)
    plt.show()