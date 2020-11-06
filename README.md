# Reinforcement-learning
## Dueling DQN으로 구현한 CartPole,MountainCar,LunraLander입니다.
### 1.CartPole
<img src="https://raw.githubusercontent.com/ha-mulan/Reinforcement-learning/master/gif/CartPole.gif"> 

### 2. MountainCar
<img src="https://raw.githubusercontent.com/ha-mulan/Reinforcement-learning/master/gif/MountainCar.gif">  

### 3.Lunar Lander
<img src="https://raw.githubusercontent.com/ha-mulan/Reinforcement-learning/master/gif/LunarLander.gif">

* actionSpace  
  * 아무 부스터도 쓰지않는다.  
  * 좌,우,하 를 향한 부스터를 쓰는 조합  
* rewardFunction   
   * 착륙에 성공할 수 있도록 보상을 정해준다. 
   * spaceShip의 x축 속도와 y축 속도를 원소로 가지는 벡터와 spaceShip의 위치벡터의 코사인 유사도 가 높으면 적절히 높은 보상  
   * spaceShip의 각속도가 낮을수록 적절히 높은 보상  
   * 안정적으로 착지하면 높은 reward,착지하지 못하고 crash시 매우 낮은 reward 학습 
   * 빠른 착륙을 유도시키기 위해 discount factor를 설정  
* training  
  * 학습초기(낮은 episode)에는 e greedy알고리즘을 따라 랜덤적인 action을 취한다.   
  * 충돌을 반복하며 낮은 보상을 가진다. 낮은 보상중 그나마 높은 보상을 얻기위한 행동을 취한다.(spaceShip의 속도벡터와 위치벡터의 코사인 유사도를 높이려고 노력한다...등등)  
  * 어쩌다 한번 착륙에 성공하서부터 높은 보상이 DDQN모델에 학습되고 점점 착륙을 성공하는 경우가 많아진다 .  
  * 600 episode 정도 설정하였고 학습 후반에는 90퍼센트 이상 착륙에 성공한다.  
* test
  * 초반 30프레임정도 랜덤 액션을 통해 agent의 위치와 속도, 각도와 각속도를 일부러 불안정하게 설정   
  * 30프레임 이후부터 DDQN모델에 자신의 state를 입력하여 해당 state에서 취할수 있는 액션들의 가치를 계산  
  * 가장 높은 가치를 가지는 action을 취해 안정적으로 착지하는 모습을 보임  

