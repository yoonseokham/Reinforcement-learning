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
from frameSave import save_frames_as_gif
model = torch.load("pth/LunaLander2.pth")
# model = torch.load("pth/LunaLanderWithRandom.pth")

model.eval()
ENV = 'LunarLander-v2'
env=gym.make(ENV)
observation = env.reset()
state=observation
state=torch.from_numpy(state).type(torch.FloatTensor)
state=torch.unsqueeze(state,0)
end_state=0
frames=[]
for i in range(600):

    if i<20:
        env.step(torch.LongTensor([[random.randrange(4)]]).item())
        print("random act!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        with torch.no_grad():
            model.eval()
            action=model(state).max(1)[1].view(1,1)
            observation_next, _, done, _ = env.step(action.item())
            print(observation_next)
            if done or abs(observation_next[6]-1)<0.000001 and  abs(observation_next[7]-1)<0.000001:
                end_state+=1
                if end_state>10:
                    if abs(observation_next[0])<=0.1:
                        print("sucess")
                    else:
                        print("fail")
                    break
            else:
                state_next = observation_next  # 관측 결과를 그대로 상태로 사용
                state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy 변수를 파이토치 텐서로 변환
                state_next = torch.unsqueeze(state_next, 0)
                state = state_next
    frames.append(env.render(mode="rgb_array"))
    time.sleep(0.01)
    # print(i)
env.close()
save_frames_as_gif(frames,filename='LunarLander.gif')
