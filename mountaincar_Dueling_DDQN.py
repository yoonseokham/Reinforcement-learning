from DuelingNetwork.model import Net
import numpy as np
import matplotlib.pyplot as plt
import gym
import random
import torch
from torch import nn
from torch import optim

from collections import namedtuple
import warnings
import time
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=UserWarning)
# 상수 정의
ENV = 'CartPole-v0'
# ENV = 'MountainCar-v0'  # 태스크 이름
GAMMA = 0.999  # 시간할인율
MAX_STEPS = 200  # 1에피소드 당 최대 단계 수
NUM_EPISODES = 400  # 최대 에피소드 수
BATCH_SIZE=32
env=gym.make(ENV)
print(env.observation_space)
num_states = env.observation_space.shape[0]  # 태스크의 상태 변수 수(2)를 받아옴
num_actions = env.action_space.n  # 태스크의 행동 가짓수(3)를 받아옴
model_m=Net(num_states,32,num_actions)
model_m.optimizer=optim.Adam(model_m.parameters(),lr=0.0001)

model_t=Net(num_states,32,num_actions)
model_t.optimizer=optim.Adam(model_m.parameters(),lr=0.0001)
Transition=namedtuple('Transition',('state','action','next_state','reward'))

def get_action(state,episode):
    epsilion=(1/(episode+1))
    if epsilion<=np.random.uniform(0,1):
        with torch.no_grad():
            model_m.eval()

            action=model_m(state).max(1)[1].view(1,1)
            '''
            state가 [1,num_states]이므로 model(state)의 결과는 당연히
            [1,num_states]이 되고 거기다가 max(1)[1]을 하면 열원소중 가장큰 원소의 인덱스를 가지는
            [1]짜리 텐서가 반환된다 그 결과를 다시 [1,1]로 바꾸기위해 view를 이용함
            '''

    else:
        action=torch.LongTensor([[random.randrange(num_actions)]])
    #action의 shape은 [1,1]
    return action
def replay(fast_end):
    if len(Transition_mem)<32:
        print("too small Transition")
        return
    # elif fast_end is False:
    batched=Transition_mem.sample()
    # else:
    #     Transition_mem.memory.sort(key=lambda element:element[3],reverse=True)
    #     batched=Transition_mem.memory[:32]
    '''
    batched >> [(state,action,reward,next_state),(state,action,reward,next_state),(state,action,reward,next_state)]
    요 상태이고 당연히 reward의 shape은 [1] 나머지는 [1,n] 인상태
    ex)
    batched: [Transition(state=tensor([[-0.4734,  0.0023]]),
                         action=tensor([[2]]),
                         next_state=tensor([[-0.4705,  0.0029]]),
                         reward=tensor([0.])),
               Transition(state=tensor([[-0.4696, -0.0027]]),
                          action=tensor([[1]]),
                          next_state=tensor([[-0.4727, -0.0032]]),
                          reward=tensor([0.])).....
    '''
    batch=Transition(*zip(*batched))
    '''
    batch >> namedtuple형태로 state에 모든 state들이 쭉 튜플형태로 저장됨
    ex)
    batch: Transition(state=(tensor([[-0.4734,  0.0023]]), tensor([[-0.4696, -0.0027]]),
                            tensor([[-0.3772, -0.0010]]), tensor([[-0.4313,  0.0058]]), tensor([[-0.3769,  0.0013]]), tensor([[-0.3944,  0.0044]]),

    '''
    state_batch=torch.cat(batch.state)
    # print(state_batch.shape)
    action_batch=torch.cat(batch.action)
    non_final_next_state_batch=torch.cat([s for s in batch.next_state if s is not None])
    reward_batch=torch.cat(batch.reward)


    model_m.eval()
    model_t.eval()


    state_action_value=model_m(state_batch).gather(1,action_batch)

    #Qm(st,at)


    mask=tuple(map(lambda s :s is not None,batch.next_state))
    # print(mask)
    non_final_mask=torch.ByteTensor(mask)

    a_m=torch.zeros(BATCH_SIZE).type(torch.LongTensor)

    a_m[non_final_mask]=model_m(non_final_next_state_batch).detach().max(1)[1]
    # print(a_m.shape)
    a_m_non_final_next_states=a_m[non_final_mask].view(-1,1)
    # print(a_m_non_final_next_states)

    next_state_value=torch.zeros(BATCH_SIZE)
    next_state_value[non_final_mask]=model_t(non_final_next_state_batch).gather(1,a_m_non_final_next_states).detach().squeeze()
    # print(reward_batch.shape)
    # print(next_state_value.shape)
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
        if len(self.memory)<self.capacity:
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
    fast_end=False
    total_reward=0
    observation = env.reset()
    state=torch.from_numpy(observation).type(torch.FloatTensor)
    state=torch.unsqueeze(state,0)
    '''
    state는 [3]짜리 텐서였는데 [1,3]이 됨
    '''
    print("episode: ",i)
    # print(len(Transition_mem))
    if i%2==0:
        model_t.load_state_dict(model_m.state_dict())
    # if count==15:
        # break
    for j in range(MAX_STEPS):
            action=get_action(state,i)
            observation_next,reward,done,_=env.step(action.item())
            if done==True:
                next_state=None
                if j>=198:
                    reward=torch.Tensor([1.0])
                else:
                    fast_end=True
                    reward=torch.Tensor([-1.0])
            else:
                reward=torch.Tensor([0.0])
                next_state=torch.from_numpy(observation_next).type(torch.FloatTensor)
                next_state=torch.unsqueeze(next_state,0)
            total_reward+=reward
            Transition_mem.push(state,action,next_state,reward)
            #
            replay(fast_end)
            #
            state=next_state
            if done ==True:
                print("steps:",j)
                print("total reward:",total_reward)
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
    time.sleep(0.01)
    # print(i)
env.close()
# env.monitor.close()
        # env.close()
