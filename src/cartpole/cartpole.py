import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib


def data_preparation(N, K, f, render=False):
    game_data = []
    for i in range(N):
        score = 0
        game_steps = []
        obs = env.reset()
        for step in range(goal_steps):
            if render:
                env.render()
            action = f(obs)
            game_steps.append((obs, action))
            obs, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        game_data.append((score, game_steps))

    game_data.sort(key=lambda s: -s[0])

    training_set = []
    for i in range(K):
        for step in game_data[i][1]:
            if step[1] == 0:
                training_set.append((step[0], [1, 0]))
            else:
                training_set.append((step[0], [0, 1]))

    print("{0}/{1}th score:{2}".format(K, N, game_data[K - 1][0]))
    if render:
        for i in game_data:
            print("Score:{0}".format(i[0]))

    return training_set


def build_model():
    with tf.device('/cpu:0'):
        model = Sequential()
        model.add(Dense(128, input_dim=4, activation='relu'))
        model.add(Dense(52, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=Adam(), loss='mse')
    return model


def train_model(model, training_set):
    X = np.array([i[0] for i in training_set]).reshape(-1, 4)
    y = np.array([i[1] for i in training_set]).reshape(-1, 2)
    with tf.device('/cpu:0'):
        model.fit(X, y, epochs=10)


if __name__ == '__main__':
    print(tf.config.list_logical_devices('GPU'))
    print(tf.test.gpu_device_name())
    print(device_lib.list_local_devices())
    env = gym.make('CartPole-v1')
    goal_steps = 500
    N = 100
    K = 10
    self_play_count = 10
    model = build_model()
    training_data = data_preparation(N, K, lambda s: random.randrange(0, 2))
    train_model(model, training_data)


    def predictor(s):
        with tf.device('/cpu:0'):
            return np.random.choice([0, 1], p=model(s.reshape(-1, 4), training=False)[0].numpy())


    for i in range(self_play_count):
        print("Self Play:{0}".format(i))
        K = (N // 9 + K) // 2
        training_data = data_preparation(N, K, predictor)
        train_model(model, training_data)

    data_preparation(100, 100, predictor, True)
