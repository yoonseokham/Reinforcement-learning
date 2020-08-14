import numpy as np
import matplotlib.pyplot as plt
import gym
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning)
# 상수 정의
ENV = 'CartPole-v0'  # 태스크 이름
GAMMA = 0.99  # 시간할인율
MAX_STEPS = 200  # 1에피소드 당 최대 단계 수
NUM_EPISODES = 1000  # 최대 에피소드 수
BATCH_SIZE=32
env=gym.make(ENV)

num_states = env.observation_space.shape[0]  # 태스크의 상태 변수 수(4)를 받아옴
num_actions = env.action_space.n  # 태스크의 행동 가짓수(2)를 받아옴
model=nn.Sequential()
model.add_module('fc1',nn.Linear(num_states,32))
model.add_module('relu1',nn.ReLU())
model.add_module('fc2',nn.Linear(32,32))
model.add_module('relu2',nn.ReLU())
model.add_module('fc3',nn.Linear(32,num_actions))
model.optimizer=optim.Adam(model.parameters(),lr=0.0001)
Transition=namedtuple('Transition',('state','action','next_state','reward'))

def get_action(state,episode):
    epsilion=0.5*(1/(episode+1))
    if epsilion<=np.random.uniform(0,1):
        with torch.no_grad():
            model.eval()
            #print("dd",model(state).max(1)[1])
            action=model(state).max(1)[1].view(1,1)

    else:
        action=torch.LongTensor([[random.randrange(2)]])
    return action
def replay():
    if len(Transition_mem)<32:
        print("too small Transition")
        return
    batched=Transition_mem.sample()

    batch=Transition(*zip(*batched))
    state_batch=torch.cat(batch.state)
    action_batch=torch.cat(batch.action)
    next_state_batch=torch.cat([s for s in batch.next_state if s is not None])
    reward_batch=torch.cat(batch.reward)
    model.eval()
    state_action_value=model(state_batch).gather(1,action_batch)
    mask=tuple(map(lambda s :s is not None,batch.next_state))
    non_final_mask=torch.ByteTensor(mask)
    next_state_value=torch.zeros(32)
    next_state_value[non_final_mask]=model(next_state_batch).max(1)[0].detach()
    expected_state_value=reward_batch+GAMMA*next_state_value

    model.train()
    loss=F.smooth_l1_loss(state_action_value,expected_state_value.unsqueeze(1))
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
# def replay():
#     model.eval()
#     if len(Transition_mem)<32:
#         print(" to small trinsiton")
#         return
#     sampled=Transition_mem.sample()
#     batch=Transition(*zip(*sampled))
#     #print("batch:",batch)
#     state_batch=torch.cat(batch.state)
#     action_batch=torch.cat(batch.action)
#     reward_batch=torch.cat(batch.reward)
#     nfns=torch.cat([s for s in batch.next_state if s is not None])
#
#     sav=model(state_batch).gather(1,action_batch)
#     mask=tuple(map(lambda s :s is not None,batch.next_state))
#     nfm=torch.ByteTensor(mask)
#
#     nsv=torch.zeros(BATCH_SIZE)
#     nsv[nfm]=model(nfns).max(1)[0].detach()
#     esav=reward_batch+GAMMA*nsv
#     model.train()
#     loss=F.smooth_l1_loss(sav,esav.unsqueeze(1))
#     model.optimizer.zero_grad()
#     loss.backward()
#     model.optimizer.step()
class mem:
    def __init__(self):
        self.memory=[]
        self.capacity=10000
        self.indx=0
    def push(self,state,action,next_state,reward):
        if len(self.memory)<10000:
            self.memory.append(None)
        self.memory[self.indx]=Transition(state,action,next_state,reward)
        self.indx=(self.indx+1)%self.capacity
    def __len__(self):
        return len(self.memory)
    def sample(self):
        return random.sample(self.memory,32)

Transition_mem=mem()
count=0
for i in range(NUM_EPISODES):
    observation = env.reset()
    state=torch.from_numpy(observation).type(torch.FloatTensor)
    state=torch.unsqueeze(state,0)
    print("episode: ",i)
    # print(len(Transition_mem))
    if count==15:
        break
    for j in range(MAX_STEPS):
            action=get_action(state,i)
            observation_next,_,done,_=env.step(action.item())
            if done==True:
                next_state=None
                if j<195:
                    reward=torch.Tensor([-1.0])
                else:
                    reward=torch.Tensor([1.0])
            else:
                reward=torch.Tensor([0.0])
                next_state=torch.from_numpy(observation_next).type(torch.FloatTensor)
                next_state=torch.unsqueeze(next_state,0)
            Transition_mem.push(state,action,next_state,reward)
            #
            replay()
            #
            state=next_state
            if done ==True:
                if j==199:
                    count+=1
                else:
                    count=0
                print("steps:",j)
                break

# for j in range(3):
observation = env.reset()
state=observation
state=torch.from_numpy(state).type(torch.FloatTensor)
state=torch.unsqueeze(state,0)
# env.monitor.start('/tmp/cartpole-experiment-1', force=True)
for i in range(200):
        # env.render()
    with torch.no_grad():
        model.eval()
        action=model(state).max(1)[1].view(1,1)
        observation_next, _, done, _ = env.step(
                    action.item())
        if done:
            break
        else:
            state_next = observation_next  # 관측 결과를 그대로 상태로 사용
            state_next = torch.from_numpy(state_next).type(
                        torch.FloatTensor)  # numpy 변수를 파이토치 텐서로 변환
            state_next = torch.unsqueeze(state_next, 0)
        state = state_next
    env.render()
    time.sleep(.1)
    # print(i)
env.close()
# env.monitor.close()
        # env.close()
