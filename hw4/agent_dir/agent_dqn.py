from agent_dir.agent import Agent
import numpy as np
from agent_dir.model import QNet
from agent_dir.replaymemory import ReplayMemory
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random, math
from itertools import count
import ipdb

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)
        random.seed(1208)
        self.duel = False
        self.Qnet = QNet(4, 512, 4, self.duel)
        self.target_net = QNet(4, 512, 4, self.duel)
        if torch.cuda.is_available():
            self.Qnet = self.Qnet.cuda()
            self.target_net = self.target_net.cuda()
            
        self.EPS_START = 1
        self.EPS_END = 0.025
        self.EPS_DECAY = 200
        self.steps_done = 0
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            self.Qnet.load_state_dict(torch.load('dqnmodel.pkl')) # 64 best!!!
        else:
            self.batch_size = 32
            self.gamma = 0.99
            self.optimizer = optim.RMSprop(self.Qnet.parameters(), lr=1.5e-4)
            self.memory = ReplayMemory(10000)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass
    def update_param(self):
        tran = self.memory.sample(self.batch_size)
        states, actions, next_states, rewards = zip(*tran)
        states = to_var(torch.stack(states))
        actions = to_var(torch.cat(actions))
        rewards = to_var(torch.cat(rewards))
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, next_states)))
        non_final_next_states = to_var(torch.stack([s for s in next_states if s is not None]))
        if torch.cuda.is_available():
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            non_final_mask = non_final_mask.cuda()
            non_final_next_states = non_final_next_states.cuda()
        # Q(s_t) get the Q(s_t, a_t) by gather
        model_predict_reward = self.Qnet(states).gather(1, actions.unsqueeze(1))
        # Compute V(s_{t+1}) for all next states.
        next_state_values = to_var(torch.zeros(self.batch_size))
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + rewards
        loss = F.mse_loss(model_predict_reward, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self):
        """
        Implement your training algorithm here
        """
        fout = open('dqn_duel_score.log','w')
        fout.write('episode,score\n')
        for episode in count(1):
            state = self.env.reset()
            state = torch.Tensor(state)
            score = 0
            while 1:
                action = self.make_action(state, False)
                state_p, reward, done, _ = self.env.step(action[0])
                score += reward
                state_p = torch.Tensor(state_p)
                if done:
                    #ipdb.set_trace()
                    state_p = None
                self.memory.push(state, action, state_p, torch.Tensor([reward])) # st, at, st', reward
                state = state_p
                if len(self.memory) > self.batch_size:
                    #update Qnet model
                    if self.steps_done % 4 == 0:
                        loss = self.update_param()
                        #print('loss:{:.4f}'.format(loss.data[0]))
                    #update target model
                    if self.steps_done % 1000 == 0:
                        self.target_net.load_state_dict(self.Qnet.state_dict())
                if done:
                    break
            score = int(score)
            print('Episode{}, score={}'.format(episode+1, score))
            fout.write('{},{}\n'.format(episode+1, score))
            if (episode+1) % 100 == 0:
                #torch.save(self.Qnet.state_dict(), 'dqnmodel/dqn_episode{}_score{}.pkl'.format(episode+1, score))
                torch.save(self.Qnet.state_dict(), 'dqnmodel/duel/dqn_episode{}_score{}.pkl'.format(episode+1, score))
        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        if test:
            observation = torch.Tensor(observation)
            eps = self.EPS_END
        else:
            eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps:
            #trust own model
            action = self.Qnet(to_var(observation).unsqueeze(0)) # unsqueeze to make state batch dimension
            action = action.data[0].max(0)[1]
        else:
            action = torch.LongTensor([random.randrange(4)])
            if torch.cuda.is_available():
                action = action.cuda()
        if test:
            return action[0]
        return action

