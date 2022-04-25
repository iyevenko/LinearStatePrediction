from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Ball():

    def __init__(self, r, x, y, dx, dy):
        self.r = r
        self.pos = np.array([x, y], np.float32)
        self.vel = np.array([dx, dy], np.float32)

    def update(self, ddx, ddy):
        self.pos += self.vel
        self.vel += np.array([ddx, ddy], np.float32)

        return self.pos


class Grid():

    def __init__(self, w, h, ball):
        self.w = w
        self.h = h
        self.ball = ball
        self.count = 0

    def update(self, ddx, ddy):
        self.ball.update(ddx, ddy)
        x, y = self.ball.pos
        r = self.ball.r
        done = (x-r < 0) or (x+r >= self.w) or (y-r < 0) or (y+r >= self.h)
        return done

    def reset(self, new_ball):
        self.ball = new_ball

    def get_grid(self):
        grid = np.zeros((self.h, self.w))
        x, y = np.int32(self.ball.pos)
        r = self.ball.r
        cv2.rectangle(grid, (x-r, y-r), (x+r, y+r), (255,255,255), -1)
        return grid

    def show(self, delay=0):
        grid = self.get_grid()
        cv2.imshow('frame', grid)
        cv2.waitKey(delay)
        # fname = f'/home/iyevenko/Documents/LinearStatePrediction/pics/env/{self.count:03d}.jpg'
        # cv2.imwrite(fname, grid)
        self.count += 1
        return grid
