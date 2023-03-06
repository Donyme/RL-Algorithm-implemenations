import gym
import numpy as np
from Actor_critic import Agent
fimport matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)
    n_games = 1800

    filename = 'cartpole.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation = observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

    games = [i for i in range(n_games)]
    plt.plot(games, score_history)
    plt.title("Score plot")
    plt.xlabel("Game number")
    plt.ylabel("Score")
    plt.savefig("results/plot.png")
