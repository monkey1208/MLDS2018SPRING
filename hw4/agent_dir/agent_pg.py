from agent_dir.agent import Agent
import scipy
from scipy import misc
import numpy as np
from agent_dir.model import PolicyNet

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import ipdb
from tqdm import tqdm

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    o = o[:,:,0]*(0.2126) + o[:,:,1]*(0.7152) + o[:,:,2]*(0.0722) # rgb to grayscale
    y = o.astype(np.uint8)
    resized = misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2).ravel()


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)
        self.policyModel = PolicyNet(6400, 256, 3, use_cnn=True)
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.policyModel.load_state_dict(torch.load('pgmodel.pkl'))
        if torch.cuda.is_available():
            self.policyModel = self.policyModel.cuda()
        self.optimizer = optim.RMSprop(self.policyModel.parameters(), lr=1e-4)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        self.old_state = np.zeros((80,80,1)).ravel()
        pass

    def finish_episode(self, rewards,actions):
        R = 0
        gamma = 0.99
        policy_rewards = []
        for reward in rewards[::-1]:
            #if a player wins, the state will be refresh
            if reward != 0:
                R = 0
            R = gamma*R + reward
            #insert to make it in the right order
            policy_rewards.insert(0, R)
        #standardlize
        policy_rewards = torch.Tensor(policy_rewards)
        policy_rewards = (policy_rewards - policy_rewards.mean())/(policy_rewards.std() + np.finfo(np.float32).eps.item())
        #update
        self.optimizer.zero_grad()
        policy_loss = []
        policy_loss = 0
        for (log_probs, reward, i) in zip(self.policyModel.saved_log_probs, policy_rewards, range(policy_rewards.size(0))):
            #policy_loss.append(-log_probs * reward)
            policy_loss -= (log_probs*reward)
        #policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policyModel.saved_log_probs[:]
    
    def train(self):
        fout = open('score.log', 'w')
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        tq = range(10000)
        for episode in tq:
            old_state = np.zeros((80,80,1)).ravel()
            rewards = []
            actions = []
            state = self.env.reset()
            score = 0
            for t in range(10000):
                state = prepro(state)
                s_p = state - old_state
                old_state = state
                action = self.make_action(s_p, test=False)
                state, reward, done, _ = self.env.step(action.data[0] + 1) #action is 1 2 3
                actions.append(action)
                rewards.append(reward)
                score += reward
                #if (t+1) % 1500 == 0:
                #    self.finish_episode(rewards)
                #    rewards = []
                if done:
                    break
            self.finish_episode(rewards,actions)
            print('Episode{}, score={}'.format(episode+1, score))
            fout.write('Episode{}, score={}\n'.format(episode+1, score))
            if (episode+1) % 200 == 0:
                 torch.save(self.policyModel.state_dict(), 'pgmodel/pg_episode{}_score{}.pkl'.format(episode+1, score))
        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        if test:
            state = prepro(observation)
            observation = state - self.old_state
            self.old_state = state
        observation = to_var(torch.Tensor(observation))
        probs = self.policyModel(observation)
        sampled_action = probs.multinomial(1)
        if not test:
            m = Categorical(probs)
            self.policyModel.saved_log_probs.append(m.log_prob(sampled_action))
        else:
            return sampled_action.data[0] + 1
        return sampled_action

