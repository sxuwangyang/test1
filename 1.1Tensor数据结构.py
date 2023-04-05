# 一、张量的数据类型
# 一般神经网络建模使用的都是torch.float32类型,并且默认是32浮点型和64位整型
# tensor有整型、浮点型、布尔型三种数据类型
# 整型
# torch.int64(torch.long)
# torch.int32(torch.int)
# torch.int16
# torch.int8
# 浮点型
# torch.float64(torch.double)
# torch.float32(torch.float)
# torch.float16
# 布尔型
# torch.bool

import torch

a = torch.tensor(1)
print(a)
print(type(a))
print(a.dtype)

b = torch.tensor(2.0)
print(b.dtype)

c = torch.tensor(True)
print(c.dtype)

# 指定数据类型
a = torch.tensor(1)
print(a.dtype)  # 默认数据类型int64

a = torch.tensor(1, dtype=torch.int32)
print(a.dtype)

a = torch.tensor(1, dtype=torch.long)
print(a.dtype)

b = torch.tensor(2.0)
print(b.dtype)

b = torch.tensor(2.0, dtype=torch.float64)
print(b.dtype)

b = torch.tensor(2.0, dtype=torch.double)
print(b.dtype)

c = torch.tensor(2, dtype=torch.float)
print(c.dtype)

# 不同类型进行转化
a = torch.tensor(1)
b = a.float()  # 调用float方法转换成浮点类型
print(b.dtype)

a = torch.tensor(1.0)
b = a.int()  # 调用int方法转换成整型
print(b.dtype)

a = torch.tensor(1)
c = a.type(torch.float)  # 使用type()函数转化成浮点型，最全面
print(c.dtype)

a = torch.tensor(1.0)
c = a.type(torch.int64)  # 使用type()函数转化成浮点型，最全面
print(c.dtype)

a = torch.tensor(1)
d = a.type_as(b)  # 根据给定的张量tensor的类型,按照这个类型返回当前张量,如果当前张量的类型已经满足要求,那么不做任何操作
print(d.dtype)

# 二、张量的维度
# 标量是0维张量、向量是1维张量、矩阵是2维张量
# a.dim() 使用dim()方法查看维度
# 标量
a = torch.tensor(1)
print(a)
print(a.dtype)
print(a.dim())
print(a.ndim)

a = torch.tensor(True)  # 布尔型数据是标量，0维张量
print(a)
print(a.dtype)
print(a.dim())
print(a.ndim)

# 向量
a = torch.tensor([1, 2, 3, 4.0])
print(a)
print(a.dtype)
print(a.dim())
print(a.ndim)

# 矩阵
a = torch.tensor([[1, 2], [3, 4], [5, 6.0]])
print(a)
print(a.dtype)
print(a.dim())
print(a.ndim)

# 三、张量的尺寸
# 可以用shape属性或者size()方法查看张量在每一维度的长度
# 可以使用view()方法改变张量的尺寸
# 如果view()方法改变尺寸失败，可以使用reshape()方法
a = torch.tensor(1)
print(a.shape)
print(a.size())

a = torch.tensor(True)
print(a.shape)
print(a.size())

a = torch.tensor([1, 2, 3, 4])
print(a.shape)
print(a.size())
print(a.size(dim=0))
print(a.size(dim=1))  # 会报错
print(a.size(dim=-1))

a = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(a.shape)
print(a.size())
print(a.size(dim=0))
print(a.size(dim=1))
print(a.size(dim=2))  # 会报错
print(a.size(dim=-2))
print(a.size(dim=-1))
print(a.size(dim=-3))  # 会报错

"""三、修改尺寸"""
# 使用view()方法可以改变张量的尺寸
a = torch.arange(0, 12)
print(a)
print(a.shape)
b = a.view(3, 4)
print(b)
print(b.shape)
# 使用reshape()方法改变张量尺寸
a = torch.arange(0, 12)
print(a)
b = a.reshape(3, 4)
print(b)
print(b.shape)
# view()和reshape()区别
# 当张量不连续时无法用view()，而reshape在张量连续或不连续时都可以使用
# 什么叫做连续，什么叫做不连续呢？
# tensor连续（contiguous）是指tensor的storage元素排列顺序与其按行优先时的元素排列顺序相同
a = torch.arange(9)
print(a)
print(a.storage())
b = a.view(3, 3)
print(b)
print(b.storage())
c = b.t()
print(c)
print(c.storage())
# 可以验证某个张量是不是连续的
a = torch.arange(0, 12)
print(a.is_contiguous())  # 是否连续

b = a.view(3, 4)
print(b.is_contiguous())

c = b.t()  # 创造一个不连续的张量
print(c)
print(c.is_contiguous())  # 结果显示不连续

d = c.view(3, 4)  # 会报错

e = c.contiguous()
print(e.is_contiguous())

f = e.view(3, 4)
print(f)

e2 = c.reshape(3, 4)
print(e2)

# 说明对于不连续的张量，reshape=contiguous+view
# 但还有一个问题，改变维度前后的张量是共享内存地址吗？换句话说，修改一个，另外一个会跟着一起修改吗？


"""四、张量的内存存储结果"""
# tensor在内存中的数据结构包括两部分
# 头信息区Tensor：保存张量的形状size，步长stride，数据类型等信息
# 存储区Storage：保存真正的数据
# 头信息区Tensor的占用内存较小，主要的占用内存是Storage
# 每一个tensor都有着对应的storage,一般不同的tensor的头信息可能不同，但是却可能使用相同的storage。
# （这里就是之前共享内存的view、reshape方法，虽然头信息的张量形状size发生了改变，但是其实存储的数据都是同一个storage）

a = torch.arange(6)
print(a.storage())

b = a.view(2, 3)
print(b.storage())

a[1] = 100
print(a)
print(b)

print(a.storage)  # <bound method Tensor.storage of tensor([0, 1, 2, 3, 4, 5])>
id(a) == id(b)  # 获取对象的内存地址，结果是False
id(a.storage) == id(b.storage)  # a和b的存储区的地址是一样的
# 可以发现，其实a和b虽然存储区是相同的，但是其实a和b整体式不同的。
# 自然，这个不同就不同在头信息区，应该是尺寸size改变了。这也就是头信息区不同，但是存储区相同，从而节省大量内存

# 我们更进一步，假设对tensor切片了，那么切片后的数据是否共享内存，切片后的数据的storage是什么样子的呢？
a = torch.arange(6)
b = a[2]
print(id(a.storage) == id(b.storage))
# 没错，就算切片之后，两个tensor依然使用同一个存储区，所以相比也是共享内存的，修改一个另一个也会变化。
# .data_ptr(),返回tensor首个元素的内存地址。
print(a.data_ptr(), b.data_ptr())
print(b.data_ptr() - a.data_ptr())  # 结果为16，这是因为b的第一个元素和a的第一个元素内存地址相差了16个字节，因为默认的tesnor是int64，也就是8个字节一个元素，所以这里相差了2个整形元素

torch.zeros(size=(2, 3))
torch.ones(size=(2, 3))
torch.arange(start=1, end=5, step=1)
torch.range(start=1, end=5, step=1)
torch.linspace(start=1, end=5, steps=5)


