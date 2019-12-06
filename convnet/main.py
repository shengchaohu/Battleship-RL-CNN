import torch
import numpy as np
import copy
from game import Game
from models.m_random import ModelRandom
from models.m_convnet import ModelConvnet
from environment import Environment
import pickle
import matplotlib.pyplot as plt


def main():
    """Entry point of the application.

    :returns: None

    """
    DIM = 10
    SHIPS = [2,3,3,4,5]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    conv_model = ModelConvnet("Vikram", DIM, len(SHIPS), device).to(device)
    conv_model.load_state_dict(torch.load('./saved_models/convenet.torch'))

    model = train_Convnet(DIM, SHIPS, device, conv_model)

    torch.save(model.state_dict(),'./saved_models/convenet.torch' )

    # play a game as a test, CNN vs Random player
    # enviranment sets the board and ship placement for these two guys
    g = Game(model, ModelRandom("Betal", DIM, len(SHIPS), device), Environment(DIM, SHIPS, "Vikram"), Environment(DIM, SHIPS, "Betal"))
    g.play()

def test_average():
    '''
    run all the sample games saved in env_sample.pkl and get a sample distribution
    '''
    DIM = 10
    SHIPS = [2,3,3,4,5]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    conv_model = ModelConvnet("Vikram", DIM, len(SHIPS), device).to(device)
    conv_model.load_state_dict(torch.load('./saved_models/convenet.torch'))

    rounds=[]
    with open('env_sample.pkl','rb') as f:
        e = pickle.load(f)

    for ee in e:
        g = Game(conv_model, ModelRandom("Betal", DIM, len(SHIPS), device), ee , Environment(DIM, SHIPS, "Betal"))
        resRound = g.play()
        if resRound:
            rounds.append(resRound)

    print(sum(rounds)/len(rounds))
    bin=max(rounds)-min(rounds)+1
    plt.hist(rounds,bins=bin,color='red',histtype='stepfilled',alpha=0.75)
    plt.show()

def train_Convnet(DIM, SHIPS, device, pretrained_model=None):
    """Train a convnet model.
    :returns: None

    """
    if pretrained_model:
        agent = pretrained_model
    else:
        agent = ModelConvnet("Vikram", DIM, len(SHIPS), device)
    agent.to(device)
    env = Environment(DIM, SHIPS, "Vikram")
    batch_size = 1024
    num_episodes = 2000 # 2000 games
    max_running_avg = 64

    batch = 0
    total_moves = 0

    # initialize a 4d array (batch_size, channel, length, width) to put in CNN
    inputs = np.empty([batch_size, 1, DIM, DIM])
    labels = np.empty([batch_size, DIM, DIM])

    for e in range(num_episodes):
        env.reset()
        state = env.get_state()
        done = False
        episode_moves = 0 # number of moves taken in this episode

        for time in range(DIM*DIM):
            # 不管怎么样一百步肯定有结果了
            action = agent.move(state)
            episode_moves += 1
            _, next_state = env.step(action)
            next_input, open_locations, hit, sunk, done = next_state
            inputs[batch] = next_input
            # 用船实际摆放的位置作为target/label
            labels[batch, :, :] = env.get_ground_truth()

            if done == True:
                # total_moves和episode_moves都是我这局用了几步
                total_moves += episode_moves
                episode_moves = 0
                if e % max_running_avg == 0 and e != 0:
                    # 64局print一次结果
                    print("Episodes: {}, Avg Moves: {}".format(e,float(total_moves)/float(max_running_avg)))
                    total_moves = 0
                # 如果提前结束了，那就退出这一句，开始下一句
                break

            batch += 1

            if batch == batch_size:
                # 积攒1024张图然后玩一次？
                agent.replay(inputs, labels)
                batch = 0
 
            state = next_state

        if done == False:
            # 这个应该永远碰不到，除非bug了
            print(env.placement)
            print(inputs,actions, hits)
            # break

    return agent


if __name__ == "__main__":
    main()
    # test_average()
