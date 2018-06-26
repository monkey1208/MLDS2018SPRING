import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, action_size, use_cnn=False):
        super(PolicyNet, self).__init__()
        self.use_cnn = use_cnn
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.saved_log_probs = []
        self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(8,8), stride=(4,4)),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=(4,4), stride=(2,2)),
                nn.ReLU()
                )
        if use_cnn:
            input_size = 2048 # need to change
        self.fc = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
                )
    def forward(self, imgs):
        if self.use_cnn:
            imgs = imgs.view(80,80).unsqueeze(0).unsqueeze(0)
            out = self.cnn(imgs).view(-1)
        else:
            out = imgs.view(-1)
        out = self.fc(out)
        # return action
        return F.softmax(out, dim=0)


class QNet(nn.Module):
    def __init__(self, channel_size, hidden_size, action_size, duel=False):
        super(QNet, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(channel_size, 32, kernel_size=(8,8), stride=(4,4)),
                #nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2)),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1)),
                nn.ReLU()
                )
        self.duel = duel
        if duel:
            self.value = nn.Sequential(
                    nn.Linear(64*7*7, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                    )
            self.advantage = nn.Sequential(
                    nn.Linear(64*7*7, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, action_size)
                    )
            #self.value = nn.Linear(hidden_size, 1)
            #self.advantage = nn.Linear(hidden_size, action_size)
        else:
            self.fc = nn.Sequential(
                    nn.Linear(64*7*7, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, action_size)
                    )
        
    def forward(self, imgs):
        batch_size = imgs.size(0)
        imgs = imgs.permute(0,3,2,1) #0 batch_size 3 channel_size
        out = self.cnn(imgs).view(batch_size, -1)
        if self.duel:
            #out = self.fc(out)
            v = self.value(out)
            a = self.advantage(out)
            out = v.expand_as(a) + (a - a.mean(1, keepdim=True).expand_as(a))
        else:
            out = self.fc(out)
        return out

class PolicyAC(nn.Module):
    def __init__(self, input_size, hidden_size, action_size, use_cnn=False):
        super(PolicyAC, self).__init__()
        self.use_cnn = use_cnn
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.saved_log_probs = []
        self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(8,8), stride=(4,4)),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=(4,4), stride=(2,2)),
                nn.ReLU()
                )
        if use_cnn:
            input_size = 2048 # need to change
        self.fc = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                )
        self.action = nn.Linear(hidden_size, action_size)
        self.v = nn.Linear(hidden_size, 1)
    def forward(self, imgs):
        if self.use_cnn:
            imgs = imgs.view(80,80).unsqueeze(0).unsqueeze(0)
            out = self.cnn(imgs).view(-1)
        else:
            out = imgs.view(-1)
        out = self.fc(out)
        action_out = self.action(out)
        v_out = self.v(out)
        # return action
        return F.softmax(action_out, dim=0), v_out
