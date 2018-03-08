import gym
import gym_pathfinding

import tensorflow as tf 
import numpy as np 

from time import sleep
from macn.model import MACN, VINConfig




# VIN conf
k = 10
ch_i = 2
ch_h = 150
ch_q = 4

# DNC conf
hidden_size = 256
memory_size = 32
word_size = 8
num_reads = 4
num_writes = 1

episodes = 200

imsize = 9

def main():
    # MACN model
    macn = MACN(
        image_shape=[imsize, imsize, 2],
        vin_config=VINConfig(k=k, ch_h=ch_h, ch_q=ch_q), 
        access_config={
            "memory_size": memory_size, "word_size": word_size, "num_reads": num_reads, "num_writes": num_writes
        }, 
        controller_config={
            "hidden_size": hidden_size
        }
    )

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/weights_batch.ckpt")

        env = gym.make('partially-observable-pathfinding-free-{n}x{n}-d2-v0'.format(n=imsize))

        dones = 0
        for episode in range(episodes):
            env.seed(episode)
            print(episode, end="\r")

            model_state = sess.run([macn.state_in])

            state = env.reset()
            for timestep in range(15):
                # env.render()
                # sleep(0.2)

                grid, grid_goal = parse_state(state)

                actions_probabilities, model_state = sess.run([macn.prob_actions, macn.state_out], feed_dict={
                    macn.X: [np.stack([grid, grid_goal], axis=2)],
                    macn.state_in: model_state
                })
                
                action = np.argmax(actions_probabilities)
                state, reward, done, _ = env.step(action)

                if done:
                    dones += 1
                    break
        print("accuracy : {}/{}".format(dones, episodes))
        env.close()

def parse_state(state):
    goal = state == 3
    state[goal] = 0

    return state, create_goal_grid(state.shape, goal)

def create_goal_grid(shape, goal):
    goal_grid = np.zeros(shape, dtype=np.int8)
    goal_grid[goal] = 10
    return goal_grid


if __name__ == "__main__":
    main()
