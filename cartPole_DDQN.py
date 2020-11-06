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
GAMMA = 0.999  # 시간할인율
MAX_STEPS = 200  # 1에피소드 당 최대 단계 수
NUM_EPISODES = 500  # 최대 에피소드 수
BATCH_SIZE=32
env=gym.make(ENV)
print(env.observation_space)
num_states = env.observation_space.shape[0]  # 태스크의 상태 변수 수(4)를 받아옴
num_actions = env.action_space.n  # 태스크의 행동 가짓수(2)를 받아옴
model_m=nn.Sequential()
model_m.add_module('fc1',nn.Linear(num_states,32))
model_m.add_module('relu1',nn.ReLU())
model_m.add_module('fc2',nn.Linear(32,32))
model_m.add_module('relu2',nn.ReLU())
model_m.add_module('fc3',nn.Linear(32,32))
model_m.add_module('relu3',nn.ReLU())
model_m.add_module('fc4',nn.Linear(32,num_actions))
model_m.optimizer=optim.Adam(model_m.parameters(),lr=0.0001)

model_t=nn.Sequential()
model_t.add_module('fc1',nn.Linear(num_states,32))
model_t.add_module('relu1',nn.ReLU())
model_t.add_module('fc2',nn.Linear(32,32))
model_t.add_module('relu2',nn.ReLU())
model_t.add_module('fc3',nn.Linear(32,32))
model_t.add_module('relu3',nn.ReLU())
model_t.add_module('fc4',nn.Linear(32,num_actions))
model_t.optimizer=optim.Adam(model_m.parameters(),lr=0.0001)
Transition=namedtuple('Transition',('state','action','next_state','reward'))

def get_action(state,episode):
    epsilion=0.5*(1/(episode+1))
    if epsilion<=np.random.uniform(0,1):
        with torch.no_grad():
            model_m.eval()
            # print("dd",model_m(state))
            action=model_m(state).max(1)[1].view(1,1)

    else:
        action=torch.LongTensor([[random.randrange(2)]])
    # print(action.shape)
    return action
    #action의 크기는
def replay():
    if len(Transition_mem)<32:
        print("too small Transition")
        return
    batched=Transition_mem.sample()

    batch=Transition(*zip(*batched))
    state_batch=torch.cat(batch.state)
    action_batch=torch.cat(batch.action)
    non_final_next_state_batch=torch.cat([s for s in batch.next_state if s is not None])
    reward_batch=torch.cat(batch.reward)


    model_m.eval()
    model_t.eval()


    state_action_value=model_m(state_batch).gather(1,action_batch)

    #Qm(st,at)


    mask=tuple(map(lambda s :s is not None,batch.next_state))
    non_final_mask=torch.ByteTensor(mask)

    a_m=torch.zeros(BATCH_SIZE).type(torch.LongTensor)

    a_m[non_final_mask]=model_m(non_final_next_state_batch).detach().max(1)[1]
    a_m_non_final_next_states=a_m[non_final_mask].view(-1,1)


    next_state_value=torch.zeros(BATCH_SIZE)
    next_state_value[non_final_mask]=model_t(non_final_next_state_batch).gather(1,a_m_non_final_next_states).detach().squeeze()

    expected_state_value=reward_batch+GAMMA*next_state_value

    model_m.train()
    loss=F.smooth_l1_loss(state_action_value,expected_state_value.unsqueeze(1))
    model_m.optimizer.zero_grad()
    loss.backward()
    model_m.optimizer.step()
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
    if i%2==0:
        model_t.load_state_dict(model_m.state_dict())
    # if count==15:
        # break
    for j in range(MAX_STEPS):
            action=get_action(state,i)
            observation_next,reward,done,_=env.step(action.item())
            # if observation_next[0]>=0.5:
            #     reward=torch.Tensor([1.0])
            #     done =True
            #     next_state=None
            # elif j<195:
            #     reward=torch.Tensor([0.0])
            #     next_state=torch.from_numpy(observation_next).type(torch.FloatTensor)
            #     next_state=torch.unsqueeze(next_state,0)
            # else:
            #     reward=torch.Tensor([-1.0])
            #     next_state=torch.from_numpy(observation_next).type(torch.FloatTensor)
            #     next_state=torch.unsqueeze(next_state,0)
            # Transition_mem.push(state,action,next_state,reward)
            # reward=observation_next[0]+0.5
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
        model_m.eval()
        action=model_m(state).max(1)[1].view(1,1)
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
