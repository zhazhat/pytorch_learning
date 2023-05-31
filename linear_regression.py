import torch

# 线性回归，拟合y=2*x+5

x=torch.rand(100,1)*10
y=x*2+(5+torch.randn(100,1) )#y增加了一些噪声
#print(x)

w=torch.rand((1),requires_grad=True)
b=torch.zeros((1),requires_grad=True)

lr_rate=0.01
print("before updata:w={},b={}".format(w.data,b.data))
for iteration in range(1000):
    wx=torch.mul(w,x)          #逐元素相乘 等价于wx=w*x
    y_pred=torch.add(wx,b)      #逐元素相加

    loss=(0.5*(y_pred-y)**2).mean()  #平方差损失

    print("iteration:{},loss={}".format(iteration,loss))

    #反向传播
    loss.backward()
    #梯度下降更新参数
    w.data.sub_(lr_rate*w.grad.data)
    b.data.sub_(lr_rate*b.grad.data)
    #梯度清0
    w.grad.data.zero_()
    b.grad.data.zero_()

print("end w={},b={}".format(w.data,b.data))
