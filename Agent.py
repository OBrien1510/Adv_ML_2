# Some inspiration taken from https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial6
import numpy as np
import random
import pickle
from keras.models import load_model, Sequential
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, adam
import gym
from Task1 import ImgReader
import cv2
from collections import deque
import datetime
from scipy.stats import sem, t
class Agent:

    def __init__(self, x, y, model1_link, model2_link, e=0.01, alpha=0.1, learning_rate=0.001, train=False, gamma=0.99, reinforcement=True):

        """
        :param x: The size of the observation space
        :param y: The size of the output/action space
        :param model_link: Where to find a pretrained model if desired
        :param e: The epsilon to be used by the agent
        :param alpha: not used
        :param learning_rate: The learning rate for reward updates
        :param train: Whether to initialize a new model or not (True for new model)
        :param gamma: discount factor for reward updates
        :param reinforcement: Whether the agent should use the reinforcement model or the machine learning model
        """

        self.input_width = x
        self.output_width = y
        self.gamma = gamma
        self.e = e
        self.learning_rate = learning_rate
        self.model1_link = model1_link
        self.model2_link = model2_link
        # using deque object because we want to the model to 'forget' old states that may have been duplicated since
        self.deck = deque(maxlen=2000)
        self.alpha = alpha
        self.train_model = train
        self.rewards = list()
        with open("./Lunar_Lander_models/label_encoder.pkl", 'rb') as fh:
            self.label_encoder = pickle.load(fh)
        self.reinforcement = reinforcement
        self.train = train
        self.model = self.get_model()


    def get_model(self):

        # we use our supervised learning model
        if not self.reinforcement:
            return load_model(self.model1_link)

        # else use pretrained reinforcement model
        elif not self.train_model:
            return load_model(self.model2_link)

	# else create a new model
        else:
            model = Sequential()
            model.add(Dense(300, input_shape=(self.input_width,), use_bias=True, init="lecun_uniform"))
            model.add(LeakyReLU(alpha=0.01))
            model.add(Dense(300, use_bias=True, init="lecun_uniform"))
            model.add(LeakyReLU(alpha=0.01))
            model.add(Dense(self.output_width, use_bias=True, init="lecun_uniform"))
            model.add(Activation("linear"))
            model.compile(loss='mse',
                          optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))

            model.summary()
            return model

    def choose_action(self, env, image, e):

        if not self.reinforcement:
            #self.model.summary()
            greedy_choice = self.model.predict(image)
            greedy_choice = np.argmax(greedy_choice)
            return greedy_choice
        else:

            rand_num = np.random.rand(1)
            # predict rewards given the current environment
            choice = self.model.predict(env)

            if rand_num[0] < e:

                # choose random action
                return np.random.randint(0, len(choice[0]))
            else:

                # choose 'greedy' option
                return np.argmax(choice[0])

    def save_rewards(self, reward):

        self.rewards.append(reward)

    def save_state(self, observation, action, reward, newObservation, done):

        state = {
            "observation": observation,
            "reward": reward,
            "action": action,
            "new_observation": newObservation,
            "done": done
        }
        self.deck.append(state)

    def train_batch(self, batch):

        # sample random states and their corresponding actions and rewards and train model on this sample
        sample = random.sample(self.deck, batch)

        # set up lists to hold training batches
        X_batch = list()
        y_batch = list()

        # each sample is a dictionary
        for state in sample:

            observation = state["observation"]
            new_observation = state["new_observation"]
            reward = state["reward"]
            action = state["action"]
            done = state["done"]

            # get reward predictions for both current state and new state
            current_state = self.model.predict(observation.reshape(1, len(observation)))[0]
            predicted_state = self.model.predict(new_observation.reshape(1, len(new_observation)))[0]

            # using an off policy approach
            # ie take the best action given the current environment, even if it doesn't comply with policy being used
            # calculate subsequent reward for taking this action
            if done:
                target = reward
            else:
                # new reward for current state will be the current reward + the max reward from the next
                # state multiplied by some discount factor
                target = reward + self.gamma * np.max(predicted_state)

            X_batch.append(np.array(observation.copy()))
            y_sample = current_state.copy()
            # update current state's reward wrt to the action taken and the subsequent gain from the next state's max action
            y_sample[action] = target
            y_batch.append(y_sample)

            if done:

                X_batch.append(np.array(new_observation.copy()))

                y_batch.append(np.array([[reward] * self.output_width])[0])

        self.model.fit(np.array(X_batch), np.array(y_batch), batch_size=len(sample)+1, verbose=0, epochs=1)


if __name__ == "__main__":

    env = gym.make('LunarLander-v2')
    env = env.unwrapped

    xn = env.observation_space.shape[0]
    yn = env.action_space.n
    print("xn", xn)
    print("yn", yn)
    model1_file_path = "./Lunar_Lander_models/Task1.hdf5"
    model2_file_path = "./Lunar_Lander_models/Task2.hdf5"

    agent = Agent(xn, yn, model1_file_path, model2_file_path, reinforcement=True, train=False)

    episodes = 1000
    episode_rewards = np.ones(episodes)
    total = 0
    start = 128
    # whether to use the reinforcement model or the supervised model
    reinforcement = True
    # set to True for agent to always take random actions (overrides previous boolean)
    random_choice = False
    count = 0
    last_50_av = 0
    e = 0.1
    # whether to train the model or not (IMPORTANT: Leave on False during evaluation, True may overwrite important data)
    train = False
    wins = 0
    # txt_path set to 'dump' for submission in case you run my code. I don't want to append to the final test txt files I made during the actual tests
    txt_path = "./test_outputs/dump.txt"
    for i in range(episodes+1):
        observation = env.reset()
        observation = observation.reshape(1, -1)
        episode_reward = 0
        step_count = 0
        # keep track of recent states
        past_states = deque(maxlen=40)
        rest = False
        while True:
            env.render()
            # use supervised method
            if not reinforcement and not random_choice:

                # Access the rednered scrnen image
                raw_image = env.render(mode='rgb_array')

                # Prepare the image for presentation to the network
                processed_image = cv2.resize(raw_image, (100, 100), interpolation=cv2.INTER_CUBIC)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
                processed_image = np.array(processed_image, dtype=np.float)
                processed_image = processed_image.reshape((1, 100, 100, 1))
                processed_image = processed_image / 255
                
                action = agent.choose_action(env, processed_image, e)
                #print("action", action)
            
            # use random choice method
            elif not random_choice:

                observation = np.reshape(np.array(observation), (1, 8))
                action = agent.choose_action(np.array(observation), None, 0)

            # use reinforcement model
            else:

                rand_int = np.random.randint(0, yn)
                action = rand_int

            new_state, reward, done, info = env.step(action)

            # if the lander has moved further away from the middle or if the lunar lander has moved upwards
            # add an extra penalty
            if train and len(past_states) >= 1 and (abs(past_states[-1][0]) < abs(new_state[0]) or abs(past_states[-1][1]) < abs(new_state[1])):
                reward -= 5

            # new_state[0] = x, new_state[1] = y
            past_states.append((new_state[0], new_state[1]))

            # if the x coords are exactly 0 for all past states then the lander must have landed safely
            rest = all((j[1] == 0) for j in past_states)

            if rest:
                #print(past_states)
                print("Win")
                reward = 200
                done = True

            episode_reward += reward
            total += reward


            #print("Current episode reward", episode_reward)

            agent.save_state(observation[0], action, reward, new_state, done)

            observation = new_state

            # set a limit on how long an episode can take so an agent doesn't get in a stalemate
            if step_count >= 2000:
                done = True

            if train and count >= start:
                agent.train_batch(128)

            count += 1

            if done:
                # final reward is not negative then the lander did not crash
                if reward > 0:
                    wins += 1
                past_states.clear()
                break

            step_count += 1

        # reduce epsilon for next episode (less random) cap @ 0.1
        e = e*0.995
        e = max(0.05, e)
        episode_rewards[i-1] = episode_reward
        string = "Episode {0} reward: {1}".format(i, episode_reward)
        print(string)
        #with open(txt_path, 'a+') as fh:
            #fh.write(string + "\n")
        last_50_av += episode_reward
        if i % 50 == 0 and i != 0:
            string = "Average Reward over the last 50 episodes {0}".format(last_50_av/50)
            print(string)
            with open(txt_path, 'a+') as fh:
                fh.write(string + "\n")
            last_50_av = 0
            if train:
                agent.model.save("./Lunar_Lander_models/Task2a.hdf5")
                print("Saving Model")

        if train and i % 200 == 0 and i != 0 :

            print("Making backup")
            agent.model.save("./Lunar_Lander_models/Task2_{0}.hdf5".format(str(datetime.datetime.now())))

    string = "Average episode over {0} episodes: {1}\n".format(episodes, total/episodes)
    string2 = "Wins over {0} episodes: {1}\n".format(episodes, wins)
    print(string)
    print(string2)
    with open(txt_path, "a+") as fh:
        fh.write(string)
        fh.write(string2)

    std = np.std(episode_rewards)
    se = sem(episode_rewards)
    av = np.mean(episode_rewards)
    h = se * t.ppf(1.95/2, episodes-1)
    ubound = av + h
    lbound = av - h

    string = "95% confidence Interval: {0} <---> {1}\n".format(lbound, ubound)

    with open(txt_path, "a+") as fh:
        fh.write(string)



