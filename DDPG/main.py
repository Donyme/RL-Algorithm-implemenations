import gym
import numpy as np
from ddpg_pytorch import Agent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    agent = Agent(input_dims=env.observation_space.shape, env=env, 
                  n_actions = env.action_space.shape[0])
    n_games =250

    figure_file = 'plots/pendulum.png'
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            new_observation, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, new_observation, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False


    for i in range(n_games):
        observation = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.choose_action(observation, evaluate)
            new_observation, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, new_observation, done)
            if not load_checkpoint:
                agent.learn()
            observation = new_observation

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models() 

        print("episode: {}, score: {}, avg_score: {}".format(i, score, best_score))    

    if not load_checkpoint:
        games = [i for i in range(n_games)]
        plt.plot(games, score_history)
        plt.title("Score plot")
        plt.xlabel("Game number")
        plt.ylabel("Score")
        plt.show()
        plt.savefig("tmp/plot.png")

