from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random


class ModelConvnet(nn.Module):
    def __init__(self, name, dim, num_ships, device):
        super(ModelConvnet, self).__init__() 
        """Initialize the Convnet Model
        """

        self.device = device

        self.name = name
        self.dim = dim

        self.epsilon = 0.0
        self.learning_rate = 0.0000001

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, num_ships+1, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_ships+1)

        self.fc = nn.Linear(dim*dim*(num_ships+1), dim*dim) 

        # self.softmax = nn.Softmax()
        # self.criterion = nn.MSELoss()

        self.softmax = nn.LogSoftmax()
        self.criterion = nn.NLLLoss()
        
    def forward(self, x):
        """Calculate the forward pass
        :returns: action scores

        """
        x = F.relu(self.bn1(self.conv1(x))) # x.shape=(1,32,10,10)
        x = F.relu(self.bn2(self.conv2(x))) # x.shape=(1,64,10,10)
        x = F.relu(self.bn3(self.conv3(x))) # x.shape=(1,6,10,10)

        # view就类似reshape
        logits = self.fc(x.view(x.size(0), -1)) # logits.shape=(1,100)

        # 注意logits已经被flatten成1*100 size了
        return logits, self.softmax(logits)

    def move(self, state):
        """ Obtain the next action
        this method is used after the model is trained, each time pick the one with highest prob
        :returns: Tuple of x,y coordinates
        """

        d = self.dim
        inputs,open_locations,_,_,_ = state
        open_locations = open_locations.flatten()

        # Sets the module in evaluation mode.
        self.eval()
        inputs = torch.Tensor(inputs).unsqueeze(0).unsqueeze(0).to(self.device)
        # the input 'picture' should be of size (1,1,10,10), 4d tensor
        assert(list(inputs.shape)==[1,1,10,10])
        # 通过forward来计算出我应该走哪一步
        _, logprobs = self.forward(inputs)
        # logprobs是1*100，logprobs[0]就是变成(100)的size
        logprobs = logprobs[0].detach().cpu().numpy() # + np.random.random(d*d)*1e-8
        # also try some other open locations -- exploratory
        max_idx = np.argmax(logprobs + open_locations*1e5)
        # get the argmax coordinates
        x,y = divmod(max_idx.item(),d)

        return x,y

    def replay(self, inputs, labels):
        ''' Replay an episode and train the model '''

        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay=0.000) 
        batch_size = len(inputs) # =1024
        minibatch_size = 128
        samples = 0

        self.train()

        while samples < 10*batch_size:
            # 一批样本可以用来训练多次？
            samples += minibatch_size
            idxs = np.random.randint(0, 1024, [minibatch_size])
            
            # indxs就是说每次挑128个出来
            input_mbatch = inputs[idxs, :, :]
            label_mbatch = labels[idxs, :, :].reshape([minibatch_size, -1])
            # print(input_mbatch.shape) (128, 1, 10, 10)
            # print(label_mbatch.shape) (128, 100)

            input_mbatch = torch.Tensor(input_mbatch).to(self.device)
            label_mbatch = torch.Tensor(label_mbatch).to(self.device)

            optimizer.zero_grad()
            logits, logprobs = self.forward(input_mbatch)
            # print(logits.shape) (128, 100)
            # print(logprobs.shape) (128, 100)
            loss = torch.mean(torch.sum(- label_mbatch * logprobs, 1))

            # label_mbatch = label_mbatch / torch.sum(label_mbatch, 1, keepdims=True)
            # print(logprobs[0])
            # print(label_mbatch[0])
            # assert(1==2)
            # loss = self.criterion(logprobs,label_mbatch)

            # 这一步会计算grandient
            loss.backward()
            # 然后optimizer.step()就是我们写的参数更新那一步
            optimizer.step()

        # print("loss is ",loss)

    def __str__(self):

        return "%s (Convnet)"%(self.name)

