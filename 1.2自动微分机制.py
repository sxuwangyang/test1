"""一、利用backward方法求导数"""
# 1.标量的反向传播
import torch

# f(x)=a*x**2+b*x+c的导数

x = torch.tensor(0.0, requires_grad=True)  # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a * torch.pow(x, 2) + b * x + c

# 判断x是否计算梯度
print(x.requires_grad)  # 结果为True
print(a.requires_grad)  # 结果为False
y.backward()
dy_dx = x.grad
print(dy_dx)
print(a.grad)

# 2.非标量的反向传播

# f(x)=a*x**2+b*x+c的导数
x = torch.tensor([1.0, 2.0], requires_grad=True)  # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a * torch.pow(x, 2) + b * x + c
y.backward()  # 会报错 grad can be implicitly created only for scalar outputs

gradient = torch.tensor([1.0, 1.0])  # 系数

print("x:\n", x)
print("y:\n", y)
y.backward(gradient=gradient)
x_grad = x.grad
print("x_grad:\n", x_grad)

# 3.非标量的反向传播可以用标量的反向传播实现

# f(x)=a*x**2+b*x+c的导数
x = torch.tensor([1.0, 2.0], requires_grad=True)  # x 需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a * torch.pow(x, 2) + b * x + c

gradient = torch.tensor([1.0, 1.0])
z = torch.sum(y)

print("x:", x)
print("y:", y)
z.backward()
x_grad = x.grad
print("x_grad:\n", x_grad)

# 二、利用autograd.grad()方法求导数

# f(x)=a*x**2+b*x+c的导数

x = torch.tensor(0.0, requires_grad=True)  # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a * torch.pow(x, 2) + b * x + c

# create_graph设置为True将允许创建更高阶的导数
dy_dx = torch.autograd.grad(outputs=y, inputs=x, create_graph=True)[0]
print(dy_dx.data)

# 求二阶导数
dy2_dx2 = torch.autograd.grad(outputs=dy_dx, inputs=x)[0]
print(dy2_dx2.data)

# 三、利用自动微分和优化器求最小值


# f(x)=a*x**2+b*x+c的导数

x = torch.tensor(0.0, requires_grad=True)  # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x], lr=0.01)


def f(x):
    result = a * torch.pow(x, 2) + b * x + c
    return result


for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()

print("y=", f(x).data, ";", "x=", x.data)
