import matplotlib
matplotlib.use('Agg') # NOQA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipdb
data = pd.read_csv('dqnscore1.log')
data_duel = pd.read_csv('dqn_duel_score.log')
score = data['score'].values
episode = data['episode'].values
score_duel = data_duel['score'].values
episode_duel = data_duel['episode'].values
score2 = []
episode2 = []
score2_duel = []
episode2_duel = []
for i in range(0,20000,100):
    score2.append(np.average(score[i:i+100]))
    episode2.append(episode[i])
    score2_duel.append(np.average(score_duel[i:i+100]))
    episode2_duel.append(episode_duel[i])
#plt.figure()
plt.title('DQN')
plt.xlabel("Episodes")
plt.ylabel("Score")
plt.ylim(0,20)
fig, ax = plt.subplots()
line1, = ax.plot(episode2, score2)
line2, = ax.plot(episode2_duel, score2_duel)
plt.legend([line1, line2], ['DQN', 'DQN with dueling'])
ax.set_xlabel("Episodes")
ax.set_ylabel("Reward")
ax.set_title("DQN")
#plt.plot(episode2, score2)
plt.savefig('dqn_duel.png')

