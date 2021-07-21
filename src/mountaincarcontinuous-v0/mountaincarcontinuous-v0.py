import random
import time

import numpy as np
import tensorflow as tf

import gym
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential


def analysis(env, max_round, action_function, reward_function, save_function, render=False, model=None):
    round = 0
    score = 0
    datas = []
    saves = save_function()
    start_obs = env.reset()
    for round in range(max_round):
        if render:
            env.render()
        action = action_function(model, start_obs)
        obs, reward, done, info = env.step(action)
        datas.append((round, obs, reward, done, info, action))
        score += reward_function(reward, obs, saves)
        if done:
            break

    min_velocity = datas[0][1][1]
    min_velocity_round = 0
    max_velocity = min_velocity
    max_velocity_round = 0
    max_position = datas[0][1][0]
    max_position_round = 0
    min_position = datas[0][1][0]
    min_position_round = 0
    rewards = [0, 0]
    for data in datas:
        print("round ", data[0],
              " obs: ", data[1],
              ", reward: ", data[2],
              ", done: ", data[3],
              ", info: ", data[4],
              ", action: ", data[5])
        if data[2] > 0:
            rewards[0] += 1
        else:
            rewards[1] += 1
        if data[1][1] < min_velocity:
            min_velocity = data[1][1]
            min_velocity_round = data[0]
        if data[1][1] > max_velocity:
            max_velocity = data[1][1]
            max_velocity_round = data[0]
        if data[1][0] > max_position:
            max_position = data[1][0]
            max_position_round = data[0]
        if data[1][0] < min_position:
            min_position = data[1][0]
            min_position_round = data[0]

    print("Round:", round)
    print("Score:", score)
    print("Positive reward: ", rewards[0])
    print("Negative reward: ", rewards[1])
    print("Start Velocity: {0}, Start Position: {1}".format(start_obs[1],start_obs[0]))
    print("Min Velocity: ", min_velocity, ", Min Velocity Round: ", min_velocity_round)
    print("Max Velocity: ", max_velocity, ", Max Velocity Round: ", max_velocity_round)
    print("Max Position: ", max_position, ", Max Position Round: ", max_position_round)
    print("Max Position Diff: ", max_position - start_obs[0])
    print("Min Position: ", min_position, ", Min Position Round: ", min_position_round)
    print("Min Position Diff: ", start_obs[0] - min_position)


def data_preparation(env, run_count, sample_count, max_round, action_function, reward_function, sort_key_function,
                     save_function, render=False, model=None):
    run_data = []
    time_start = time.time()
    for current_run_count in range(run_count):
        score = 0
        run_step = []
        obs = env.reset()
        saves = save_function()
        for current_round in range(max_round):
            if render:
                env.render()
            action = action_function(model, obs)
            run_step.append((obs, action))
            obs, reward, done, info = env.step(action)
            score += reward_function(reward, obs, saves)
            if done:
                break
        run_data.append((score, run_step))
    print("Run Time: ", time.time() - time_start)

    run_data.sort(key=sort_key_function)

    training_set = []
    for current_sample_count in range(sample_count):
        for step in run_data[current_sample_count][1]:
            if step[1][0] == 1:
                training_set.append((step[0], [0, 0, 1]))
            elif step[1][0] == 0:
                training_set.append((step[0], [0, 1, 0]))
            else:
                training_set.append((step[0], [1, 0, 0]))

    print("{0}/{1}th score:{2}".format(sample_count, run_count, run_data[sample_count - 1][0]))
    if render:
        for i in run_data:
            print("Score:{0}".format(i[0]))

    return training_set


def build_model():
    with tf.device('/cpu:0'):
        model = Sequential()
        model.add(Dense(128, name='dense_1', input_dim=2, activation='relu'))
        model.add(Dense(64, name='dense_2', activation='relu'))
        model.add(Dense(3, name='dense_3', activation='softmax'))
        model.compile(optimizer='Adam', loss='mse')
    return model


def train_model(model, training_set, build_function_X, build_function_Y, epochs, callbacks):
    X = build_function_X(training_set)
    Y = build_function_Y(training_set)
    time_start = time.time()
    with tf.device('/cpu:0'):
        model.fit(X, Y,
                  epochs=epochs,
                  verbose=0,
                  validation_data=(X, Y),
                  callbacks=callbacks)
    print("Train time: " + str(time.time() - time_start))


def build_x(training_set):
    return np.array([i[0] for i in training_set]).reshape(-1, 2)


def build_y(training_set):
    return np.array([i[1] for i in training_set]).reshape(-1, 3)


def predict_action(model, obs):
    # with tf.device('/cpu:0'):
    #     action = model(obs.reshape(-1, 2), training=False).numpy()[0]
    #     action = (action - 0.5) * 2
    #     return action
    with tf.device('/cpu:0'):
        percentage = model(obs.reshape(-1, 2), training=False)[0].numpy()
        return [np.random.choice([-1, 0, 1], p=percentage)]


def calculate_reward(reward, obs, saves):
    ret_reward = reward
    if saves['max_position'] == -1.3:
        saves['max_position'] = obs[0]
    else:
        if saves['max_position'] < obs[0]:
            ret_reward += (obs[0] - saves['max_position']) * 100
            saves['max_position'] = obs[0]
    return ret_reward


def build_save():
    return {'max_position': -1.3}


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    env.reset()

    run_count = 100
    sample_count = 10
    max_round = 200
    training_count = 10
    max_height = -10

    # analysis(env, max_round,
    #          action_function=lambda model, obs: [random.randrange(-1, 1)],
    #          reward_function=calculate_reward,
    #          save_function=build_save,
    #          render=True)

    model = build_model()

    training_set = data_preparation(env, run_count, sample_count, max_round,
                                    action_function=lambda model, obs: [random.randrange(-1, 2)],
                                    reward_function=calculate_reward,
                                    sort_key_function=lambda s: -s[0],
                                    save_function=build_save,
                                    render=False)
    train_model(model, training_set, build_x, build_y, 100, [])

    for i in range(training_count):
        print("Train Round {0}".format(i))
        training_set = data_preparation(env, run_count, sample_count, max_round,
                                        action_function=predict_action,
                                        reward_function=calculate_reward,
                                        sort_key_function=lambda s: -s[0],
                                        save_function=build_save,
                                        render=False, model=model)
        train_model(model, training_set, build_x, build_y, 100, [])

    training_set = data_preparation(env, 10, 10, max_round,
                                    action_function=predict_action,
                                    reward_function=calculate_reward,
                                    sort_key_function=lambda s: -s[0],
                                    save_function=build_save,
                                    render=True, model=model)
