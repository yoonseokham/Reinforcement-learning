import torch
from torch import*
import numpy as np
'''
numpy
'''
t1=np.array([0,1,2,3,4,5,6],dtype='f')
print("t1:",t1)
print("t1의 rank:",t1.ndim)
print("t1의 shape:",t1.shape)
t2=np.array([[0,1,2,3,4,5,6],[7,8,9,10,11,12,13]],dtype='f')
print(t2[0,1],"\n",t2[:,-1])


'''
torch
'''

t1=torch.FloatTensor([0,1])
t2=torch.Tensor([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]).type(torch.float32)
print("t2:\n",t2)
print("t2 dim:",t2.dim())
print("t2 shape:",t2.shape)
print("t2 size:",t2.size())

print("t2[:,:-1]:",t2[:,:-1])
print("t2[:,:-1]:",t2[:,-1])

'''
broadcasting
'''
m1=torch.Tensor([[1,1],[2,2]]).type(torch.float32)
m2=torch.FloatTensor([[1]])
#덧셈가능하게 크기를 맞추고 덧셈연산을 한 결과가 나옴
print(m1+m2)
m1=torch.Tensor([[1],[2]]).type(torch.float32)
m2=torch.FloatTensor([[3,4]])
#m2의 크기가 m1에 맞춰진 상태로 덧셈을 함
print(m1+m2)
'''
operation

multiplication
'''
m1=torch.Tensor([[1,1],[2,2]]).type(torch.float32)
m2=torch.FloatTensor([[1,2,3],[4,5,6]])
print("matrix multiplication:\n",m1.matmul(m2))

m1=torch.Tensor([[1],[2]]).type(torch.float32)
m2=torch.FloatTensor([[1,2,3],[4,5,6]])
print("scalar multiplication:\n",m1.mul(m2))

'''
operation

mean and sum

'''
print(m2[:])
print(m2[0].mean())
print(m2.mean(dim=0))
print(m2.mean(dim=1))
print(m2.mean(dim=-1))

print(m2.sum(dim=0))
print(m2.sum(dim=1))
print(m2.sum(dim=-1))
'''
Max and ArgMax
'''

print(m2.max()) # value만 나옴
print(m2.max(dim=0)) #차원 지정시 value와 인덱스 둘다 나옴

'''
view *******************
원소의 수를 유지하면서 텐서의 크기를 변경하는 함수

'''
t=np.array([
            [[1,2,3,4],[5,6,7,8]],
            [[9,10,11,12],[13,14,15,16]]
            ])
print(t.shape)
t=torch.Tensor(t).type(torch.float32)
print(t.view([-1,8]))
print(t.view([-1,4,4]))
'''
squeeze 1인 차원을 제거함
'''
print(t.view([-1,4,4]).shape)
print(t.view([-1,4,4]).squeeze().shape)
print(t.view([4,-1,4]).shape)
print(t.view([4,-1,4]).squeeze().shape)
'''
unsqueeze 특정위치에 1인 차원을 추가한다
인자가 차원을 추가할 인덱스 이다.
'''
t=np.array([1,2,3,4])
t=torch.FloatTensor(t)
print("size:",t.size()[0])
print("t :",t)
print("t.unsqueeze(0) :",t.unsqueeze(0))
print("t.view(-1,t.size[0]) :",t.view(-1,t.size()[0]))
print("t.unsqueeze(0) :",t.unsqueeze(-1))
'''
텐서 자료형 and 자료형 변환

torch.float32=torch.FloatTensor()
torch.float64
torch.int32
torch.int64=torch.LongTensor()
torch.ByteTensor([True,False])
'''
t=torch.LongTensor([1,2,3,4,5])
t=t.float()
print(t)
t=torch.ByteTensor([True,False,True,False,True])
t=t.long()
print(t)
'''
concatenate
cat and stack

'''
x = torch.FloatTensor([[1, 2],
                       [3, 4]])
y = torch.FloatTensor([[5, 6],
                       [7, 8]])
z = torch.FloatTensor([[9, 10],
                       [11, 12]])
print(torch.cat([x,y,z],dim=0))
print(torch.cat((x,y),dim=1))
print(torch.cat((x,y),dim=1).view(1,-1))
print(torch.cat((x,y),dim=1).view(-1,1))
print(torch.cat((x,y),dim=1).view(-1,1).squeeze())
x = torch.FloatTensor([1, 2]
                       )
y = torch.FloatTensor([5, 6]
                       )
z = torch.FloatTensor([9, 10]
                       )
print(torch.stack([x,y,z],dim=0))
print(torch.cat([x.unsqueeze(0),y.unsqueeze(0),z.unsqueeze(0)]))
print(torch.stack([x,y,z],dim=1))
'''
ones_like/zeros_like
크기는 똑같지만 1이나 0으로 가득찬 텐서 생성
'''
x=torch.FloatTensor([1,2,3])
y=torch.ones_like(x)
z=torch.zeros_like(x)
print(y)
print(z)
'''
in-place operation
연산뒤에 _ <요ㅗ거 붙이면 연산과 동시에 결과 덮어씀

'''
x=torch.LongTensor([[1]])
print(x.shape)
print(x.max(1)[1].view(1,1))
print(x.view(1,-1))
print(x.view(1,-1).expand(-1,2))

example_tuples = [ ('A', 3, 'a'), ('B', 1, 'b'), ('C', 2, 'c') ]

# 정렬
example_tuples.sort(key = lambda element : element[1],reverse=True)
print(example_tuples)
