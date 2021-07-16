import datetime
import time
import os
import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def data_preparation(N, K, max_step, f, render=False):
    game_data = []
    time_start = time.time()
    for i in range(N):
        game_steps = []
        obs = env.reset()
        score = 0
        max_position = obs[0]
        for step in range(max_step):
            if render:
                env.render()
            action = f(obs)
            game_steps.append((obs, action))
            obs, reward, done, info = env.step(action)
            score += reward
            if max_position < obs[0]:
                max_position = obs[0]
            if done:
                break
        game_data.append((score + max_position, game_steps))

    print("Data preparation time: " + str(time.time() - time_start))
    game_data.sort(key=lambda s: s[0], reverse=True)

    training_set = []
    for i in range(K):
        for step in game_data[i][1]:
            if step[1] == 0:
                training_set.append((step[0], [1, 0, 0]))
            elif step[1] == 1:
                training_set.append((step[0], [0, 1, 0]))
            else:
                training_set.append((step[0], [0, 0, 1]))

    print("{0}/{1}th score:{2}".format(K, N, game_data[K - 1][0]))
    if render:
        for i in game_data:
            print("Score:{0}".format(i[0]))

    return training_set


def build_model():
    with tf.device('/cpu:0'):
        model = Sequential()
        model.add(Dense(128, name='dense_1', input_dim=2, activation='relu'))
        model.add(Dense(48, name='dense_2', activation='relu'))
        model.add(Dense(3, name='dense_3', activation='softmax'))
        model.compile(optimizer='Adam', loss='mse')
    return model


def train_model(model, training_set):
    X = np.array([i[0] for i in training_set]).reshape(-1, 2)
    Y = np.array([i[1] for i in training_set]).reshape(-1, 3)
    time_start = time.time()
    with tf.device('/cpu:0'):
        model.fit(X, Y,
                  epochs=100,
                  verbose=0,
                  validation_data=(X, Y),
                  callbacks=[tb_callback])
    print("Train time: " + str(time.time() - time_start))


def make_tensorboard_dir(dir_name):
    root_logdir = os.path.join(os.curdir, dir_name)
    sub_dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(root_logdir, sub_dir_name)


if __name__ == '__main__':
    print(tf.config.list_logical_devices('GPU'))
    print(tf.test.gpu_device_name())
    print(device_lib.list_local_devices())

    tb_log_dir = make_tensorboard_dir("tensor_log")
    tb_callback = keras.callbacks.TensorBoard(log_dir=tb_log_dir,
                                              histogram_freq=1,
                                              profile_batch='500,520')

    goal_steps = 200
    N = 100
    K = 10
    self_play_count = 12

    env = gym.make('MountainCar-v0')

    model = build_model()
    data = data_preparation(N, K, goal_steps, f=lambda s: random.randrange(0, 3))
    train_model(model, data)


    def predictor(obs):
        with tf.device('/cpu:0'):
            return np.random.choice([0, 1, 2], p=model(obs.reshape(-1, 2), training=False)[0].numpy())


    for i in range(self_play_count):
        print("Self Play:{0}".format(i))
        data = data_preparation(N, K, goal_steps, f=predictor, render=False)
        train_model(model, data)

    data_set = data_preparation(100, 100, goal_steps, f=predictor, render=False)
    data_preparation(10, 0, goal_steps, f=predictor, render=True)
