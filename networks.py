# User Defined Policy Network -- body should be same size for both
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def softmax(x):
      es = [0]*len(x)
      for i,ix in enumerate(x):
            es[i] = np.exp(ix)
      res = [0]*len(x)
      s = sum(es)
      for i in range(len(res)):
             res[i] = es[i]/s
      return res

class PolicyNet(nn.Module):
      def __init__(self,inputs,outputs,epochs=100):
              super(PolicyNet,self).__init__()

              self.fc1 = nn.Linear(inputs,128)
              self.act1 = nn.Sigmoid()

              self.conv1 = nn.Conv1d(1,3,kernel_size=3)
              self.act2 = nn.Sigmoid()

              self.batch_norm1 = nn.BatchNorm1d()

              self.conv2 = nn.Conv1d(3,9,kernel_size=3)
              self.act3 = nn.Sigmoid()

              self.flatten1 = nn.Flatten()

              self.fc2 = nn.Linear(128 * 9, 128 * 3)
              self.act4 = nn.Sigmoid()

              self.batch_norm2 = nn.BatchNorm1d()

              self.fc3 = nn.Linear(128 * 3, 128)
              self.act5 = nn.Sigmoid()

              self.out = nn.Linear(128,outputs)

              self.optimizer = optim.SGD(self.parameters(),lr=1e-3)
              self.loss = nn.CrossEntropyLoss()

              self.epochs = epochs
      
      def forward(self,x):
             # fully connected 1
             x = self.act1(self.fc1(x))
             # conv layer 1
             x = self.act2(self.act2(self.conv1(x)))
             # batch norm 1
             x = self.batch_norm1(x)
             # conv layer 2
             x = self.act3(self.conv2(x))
             # flatten
             x = self.flatten1(x)
             # fully connected 2
             x = self.act4(self.fc2(x))
             # batch norm 2
             x = self.batch_norm2(x)
             # fully connected 3
             x = self.act5(self.fc3(x))
             # output layer
             return self.out(x)
      
      def train(self,x,y):
            for epoch in range(self.epochs):
                    t_loss : int = 0
                    for i,state in enumerate(x):
                          state = torch.tensor(state)
                          truth = torch.tensor(y[i])

                          self.optimizer.zero_grad()
                          
                          loss = self.loss(self(state),truth)
                          t_loss += int (loss)
                          loss.backward()

                          self.optimizer.step()
                    print ('epoch:',epoch,'loss:',t_loss)
      
class ValueNet(nn.Module):
      def __init__(self,inputs,epochs=100):
              super(ValueNet,self).__init__()

              self.fc1 = nn.Linear(inputs,128)
              self.act1 = nn.Sigmoid()

              self.conv1 = nn.Conv1d(1,3,kernel_size=3)
              self.act2 = nn.Sigmoid()

              self.batch_norm1 = nn.BatchNorm1d()

              self.conv2 = nn.Conv1d(3,9,kernel_size=3)
              self.act3 = nn.Sigmoid()

              self.flatten1 = nn.Flatten()

              self.fc2 = nn.Linear(128 * 9, 128 * 3)
              self.act4 = nn.Sigmoid()

              self.batch_norm2 = nn.BatchNorm1d()

              self.fc3 = nn.Linear(128 * 3, 128)
              self.act5 = nn.Sigmoid()

              self.out = nn.Linear(128,1)

              self.optimizer = optim.SGD(self.parameters(),lr=1e-3)
              self.loss = nn.MSELoss()

              self.epochs = epochs
      
      def forward(self,x):
             # fully connected 1
             x = self.act1(self.fc1(x))
             # conv layer 1
             x = self.act2(self.act2(self.conv1(x)))
             # batch norm 1
             x = self.batch_norm1(x)
             # conv layer 2
             x = self.act3(self.conv2(x))
             # flatten
             x = self.flatten1(x)
             # fully connected 2
             x = self.act4(self.fc2(x))
             # batch norm 2
             x = self.batch_norm2(x)
             # fully connected 3
             x = self.act5(self.fc3(x))
             # output layer
             return self.out(x)
      
      def train(self,x,y):
             for epoch in range(self.epochs):
                    t_loss : int = 0
                    for i,state in enumerate(x):
                          state = torch.tensor(state)
                          truth = torch.tensor(y[i])

                          self.optimizer.zero_grad()
                          
                          loss = self.loss(self(state),truth)
                          t_loss += int (loss)
                          loss.backward()

                          self.optimizer.step()
                    print ('epoch:',epoch,'loss:',t_loss)