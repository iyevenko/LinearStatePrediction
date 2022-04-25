import numpy as np


class LearningSchedule():

    def __init__(self):
        self.count = 0


class ConstantSchedule():

    def __init__(self, ball_speed):
        super().__init__()
        self.vel = ball_speed

    def get_vel(self):
        return 0, self.vel


class RandomSchedule(LearningSchedule):

    def __init__(self, ball_speed):
        super().__init__()
        self.vel = ball_speed

    def get_vel(self):
        theta = np.random.uniform(high=2*np.pi)
        dx = self.vel * np.cos(theta)
        dy = self.vel * np.sin(theta)
        return dx, dy



class DifficultyRampSchedule(LearningSchedule):

    def __init__(self, const_episodes, ramp_episodes, ball_speed):
        super().__init__()
        self.const_eps = const_episodes
        self.ramp_eps = ramp_episodes
        self.vel = ball_speed

    def get_vel(self):
        if self.count < self.const_eps:
            dx, dy = 0, self.vel
        else:
            d = np.pi * (self.count - self.const_eps) / self.ramp_eps
            theta = np.random.uniform(np.pi / 2 - d, np.pi / 2 + d)
            dx = self.vel * np.cos(theta)
            dy = self.vel * np.sin(theta)

        self.count += 1
        return dx, dy

