import gym
import numpy as np
from Agent import Agent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = Agent(alpha=1e-2, n_actions=env.action_space.n)
    n_games = 50
    batch_size = 5000

    filename = 'cartpole.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    render = False
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        batch_observation = []
        batch_actions     = []
        batch_weights     = []
        batch_returns     = []
        batch_lens        = []
        
        observation = env.reset()
        done = False
        ep_rewards = []
        
        finished_rendering_this_epoch = False
        
        while True:
            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_observation.append(observation.copy())
            action = agent.choose_action(observation)
            observation, reward, done, info = env.step(action)

            batch_actions.append(action)
            ep_rewards.append(reward)

            if done:
                ep_return, ep_length = sum(ep_rewards), len(ep_rewards)
                batch_returns.append(ep_return)
                batch_lens.append(ep_length)

                batch_weights += [ep_return] * ep_length

                observation, done, ep_rewards = env.reset(), False, []

                finished_rendering_this_epoch = True
                if len(batch_observation) > batch_size:
                        break
        
        batch_loss = agent.learn(batch_states=batch_observation, batch_actions=batch_actions, batch_weights=batch_weights)

        avg_score = np.mean(batch_returns)
        print("Epoch number: {}, Loss : {}, Average_returns: {}".format(i, batch_loss.item(), avg_score))
        score_history.append(avg_score)

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