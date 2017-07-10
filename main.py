import numpy as np
from ddqn.environment import EP
from ddqn.DDQN import DDQNAgent
import os
import gc



def pplot(ax,y, colors=['b']):
    x = range(len(y))
    if ax.lines:
        for line in ax.lines:
            line.set_xdata(x)
            line.set_ydata(y)
    else:
        for color in colors:
            ax.plot(x, y, color)
    fig.canvas.draw()


def check_path(path):
    if path != os.getcwd():
        os.chdir(path)



if __name__ == "__main__":

    path = os.getcwd()
    check_path(path)
    state_size = 6
    action_size = 5
    action_space = np.linspace(-10000, 10000, action_size)

    env = EP('./EPlus/small_office.fmu', state_size=state_size,
             battery_capacity=30000, action_space=action_space)

    agent = DDQNAgent(state_size, action_size, memory_lenght=10000,
                      discount=0.99, epsilon=1)
    done = False
    batch_size = 20
    load_weights = True
    episodes = 10
    times = 50
    reward_list = []
    r_list = np.array([])
    total_reward = 0

    if 'save' not in os.listdir('.'):
        os.mkdir('./save/')

    if load_weights:
        agent.load('./save/ddqn.h5')


    for e in range(episodes):
        check_path(path)
        states = env.reset()
        states = np.reshape(states, [states.shape[0], state_size])

        for time in range(times):
            actions = agent.act(states)
            next_state, reward, done = env.step(actions)
            next_state = np.reshape(next_state, [next_state.shape[0], state_size])
            agent.remember(states, actions, reward, next_state, done)
            states = next_state
            total_reward += reward.sum()
            reward_list.append(total_reward)
            r_list = np.append(r_list, reward)
            agent.update_target_network()

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 5 == 0:
            agent.save("./save/ddqn.h5")
            print("episode: {}/{}, score: {}, e: {}"
                  .format(e, episodes,time, agent.epsilon))
            gc.collect()
