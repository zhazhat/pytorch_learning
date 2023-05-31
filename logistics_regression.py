import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import time
"""
logistic回归又称为对数几率回归，即为y=sigmoid(w*x+b)
可以用y是否大于0.5来解决二分类问题
"""

"""数据准备，这里用随机种子随机生成100个正样本与负样本，样本特征数为2，标签分别为0，1"""
torch.manual_seed(0)   #布置随机种子，保证每次运行生成同样的数据


num_data=100   #数据数目

lr_rate=0.01

mean=1.5     #x均值

x0=torch.normal(mean,1,(100,2))
y0=torch.zeros(100,1)
x1=torch.normal(-mean,1,(100,2))
y1=torch.ones(100,1)
train_x=torch.cat((x0,x1),dim=0)   #按行连接
train_y=torch.cat((y0,y1),dim=0)
#模型搭建
class logis_regre(torch.nn.Module):
    def __init__(self):
        super(logis_regre,self).__init__()
        self.features=nn.Linear(2,1)   #隐层神经元数为2（特征数）
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=self.features(x)
        x=self.sigmoid(x)
        return x
LR=logis_regre()   #实例化模型
'''LR=nn.Sequential(nn.Linear(2,1),
                    nn.Sigmoid())'''

lossfunc=nn.BCELoss()
print(LR)
optim=torch.optim.SGD(LR.parameters(),lr=lr_rate,momentum=0.8)
print("before updata w:{} b:{}".format(LR.features.weight,LR.features.bias))
#迭代训练
for iteration in range(100):
    #前向传播
    output=LR(train_x)
   # print(output.shape)
    #计算loss
    loss=lossfunc(output,train_y)
    #梯度反向传播
    loss.backward()
    #参数更新
    optim.step()
    #梯度清0
    optim.zero_grad()
   # print("iteration:{},loss:{}".format(iteration,loss))
    if iteration%20==0:
        mask=output.gt(0.5).float().squeeze()
       # print(mask.size())
        correct=(mask==train_y).sum()  #计算正确预测样本个数
        acc=correct.item()/train_x.size(0)   #item计算tensor的值
        plt.scatter(x0.data.numpy()[:,0],x0.data.numpy()[:,1],c='r',label='class 0')
        plt.scatter(x1.data.numpy()[:,0],x1.data.numpy()[:,1],c='b',label='class 1')

       # 设置图片大小
       # fig=plt.figure(figsize=(20, 8), dpi=80)
        """ figure图形图标的意思，在这里指的是我们画的图
        通过实例化一个figure并且传递参数，能够在后台自动使用该figure实例
        在图像模糊的时候可以传入d即i参数，让图片更加清晰。
         """
        w0,w1=LR.features.weight[0]
        w0,w1=float(w0.item()),float(w1.item())
        plot_b=float(LR.features.bias[0].item())
        plot_x=np.arange(-6,6,0.1)
        plot_y=(-w0*plot_x-plot_b)/w1

        plt.xlim(-7,7)   #x轴范围
        plt.ylim(-7,7)
        plt.plot(plot_x,plot_y)

        plt.title("Iteration:{},accracy:{:.2f}".format(iteration,acc))
        plt.text(-5,7,"loss=%.2f"%loss.data.numpy(),fontdict={'size': 20, 'color': 'red'})  #文字描述
        plt.legend()   #绘制图例，labels已设置

        plt.show()
        #plt.pause(0.5)
        time.sleep(0.5)

        if(acc>0.9):
            break
