# 小白深度学习Day1

## 1、自动微分

深度学习框架通过自动计算导数，即*自动微分*（automatic differentiation）来加快求导。 实际中，根据设计好的模型，系统会构建一个*计算图*（computational graph）， 来跟踪计算是哪些数据通过哪些操作组合起来产生输出。 自动微分使系统能够随后反向传播梯度。 这里，*反向传播*（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。

### 1.1举个例子

我们想对函数$y=2x^Tx$关于列向量$x$求导。首先，我们创建变量`x`并为其分配一个初始值。

在我们计算$y$关于$x$的梯度之前，需要一个地方来存储梯度。 重要的是，我们不会在每次对一个参数求导时都分配新的内存。 因为我们经常会成千上万次地更新相同的参数，每次都分配新的内存可能很快就会将内存耗尽。 注意，**一个标量函数关于向量$x$的梯度是向量**，并且与$x$具有相同的形状。

```python
import torch
x = torch.arange(4.0) ##tensor([0., 1., 2., 3.])
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)

y = 2 * torch.dot(x, x) #计算y,tensor(28., grad_fn=<MulBackward0>)
```

`x`是一个长度为4的向量，计算`x`和`x`的点积，得到了我们赋值给`y`的标量输出。 接下来，通过调用反向传播函数来自动计算`y`关于`x`每个分量的梯度，并打印这些梯度。

```python
y.backward()
x.grad #tensor([ 0.,  4.,  8., 12.])
```

在计算`x`的另一个函数之前，必须使用`x.grad.zero_()`来清楚`x`之前的梯度。下面继续举个例子

```python
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad #tensor([1., 1., 1., 1.])
```

### 1.2非标量变量的反向传播

当`y`不是标量时，向量`y`关于向量`x`的导数的最自然解释是一个矩阵。 对于高阶和高维的`y`和`x`，求导的结果可以是一个高阶张量。

然而，虽然这些更奇特的对象确实出现在高级机器学习中（包括深度学习中）， 但当调用向量的反向计算时，我们通常会试图计算一批训练样本中每个组成部分的损失函数的导数。 这里，我们的目的不是计算微分矩阵，而是**单独计算批量中每个样本的偏导数之和**。

```python
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad #tensor([0., 2., 4., 6.])
```

### 1.3分离计算

有时，我们希望将某些计算移动到记录的计算图之外。 例如，假设`y`是作为`x`的函数计算的，而`z`则是作为`y`和`x`的函数计算的。 想象一下，我们想计算`z`关于`x`的梯度，但由于某种原因，希望将`y`视为一个常数， 并且只考虑到`x`在`y`被计算后发挥的作用。

~~(我不明白这样做意义在哪，既然y要当成常数引入y干嘛？)~~

这里可以分离`y`来返回一个新变量`u`，该变量与`y`具有相同的值， 但丢弃计算图中如何计算`y`的任何信息。 换句话说，梯度不会向后流经`u`到`x`。 因此，下面的反向传播函数计算`z=u*x`关于`x`的偏导数，同时将`u`作为常数处理， 而不是`z=x*x*x`关于`x`的偏导数。

```python
x.grad.zero_()
y = x * x
u = y.detach() #核心步骤了，u的值与y相同，但不参与梯度计算
z = u * x

z.sum().backward()
x.grad == u #True
```

由于记录了`y`的计算结果，我们可以随后在`y`上调用反向传播， 得到`y=x*x`关于的`x`的导数，即`2*x`。

```python
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

## 2、线性回归(第一个深度学习项目)

### 2.1从零开始实现(使用李沐老师的DL包)

#### 2.11生成数据集

我们将根据带有噪声的线性模型构造一个人造数据集。 我们的任务是使用这个有限样本的数据集来恢复这个模型的参数。 我们将使用低维数据，这样可以很容易地将其可视化。 在下面的代码中，我们生成一个包含1000个样本的数据集， 每个样本包含从标准正态分布中采样的2个特征。也就是说，**训练集的feature维度是$R^{1000*2}$**。

```python
%matplotlib inline
import random
import torch
import torch as d2l

def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    #0是均值，1是方差，size是(num_examples, len(w))。
    #本题的num_examples是1000，len(w)是2
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    #加上噪声
    return X, y.reshape((-1, 1)) 
	#把y转化成一行
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

#### 2.12读取数据集

训练模型时要对数据集进行遍历，每次抽取一小批量样本，并使用它们来更新我们的模型。 由于这个过程是训练机器学习算法的基础，所以有必要定义一个函数， 该函数能打乱数据集中的样本并以小批量方式获取数据。

在下面的代码中，我们定义一个`data_iter`函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为`batch_size`的小批量。 每个小批量包含一组特征和标签。

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)#打乱顺序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        #其实就是顺序取出batch_size个样本
        yield features[batch_indices], labels[batch_indices]
```

直观感受一下小批量运算：读取第一个小批量数据样本并打印。 每个批量的特征维度显示批量大小和输入特征数。 同样的，批量的标签形状与`batch_size`相等。如下面代码所示：

```python
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
    
[[-0.2917712  -1.5321481 ]
 [-0.41116217 -0.5758122 ]
 [-0.580725    0.0188676 ]
 [-0.33994374  0.21589722]
 [-0.01192594  0.04896761]
 [-0.63423014 -0.18605877]
 [ 1.7412642  -0.76409566]
 [-0.6550841  -0.02194321]
 [-0.37621465 -0.27568716]
 [-0.5693811   1.0725546 ]]
 [[ 8.831048 ]
 [ 5.3464904]
 [ 2.9789436]
 [ 2.76835  ]
 [ 4.0068398]
 [ 3.562237 ]
 [10.282065 ]
 [ 2.9639275]
 [ 4.3731804]
 [-0.5902057]]
```

当我们运行迭代时，我们会连续地获得不同的小批量，直至遍历完整个数据集。 上面实现的迭代对教学来说很好，但它的执行效率很低，可能会在实际问题上陷入麻烦。 例如，它要求我们将所有数据加载到内存中，并执行大量的随机内存访问。 在深度学习框架中实现的内置迭代器效率要高得多， 它可以处理存储在文件中的数据和数据流提供的数据。

#### 2.13初始化模型参数

`w`和`b`需要初始化，很普通的做法就是`w`从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重；`b`初始化为0。

```python
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

#### 2.14定义模型

接下来，我们必须定义模型，将模型的输入和参数同模型的输出关联起来。这里使用到了`python`的广播机制。

```python
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b
```

#### 2.15定义损失函数

使用的是平方损失函数，也很简单了。

```python
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

#### 2.16定义优化算法

在每一步中，使用从数据集中随机抽取的一个小批量，然后根据参数计算损失的梯度。 接下来，朝着减少损失的方向更新我们的参数。 下面的函数实现小批量随机梯度下降更新。 该函数接受模型参数集合、学习速率和批量大小作为输入。每 一步更新的大小由学习速率`lr`决定。 因为我们计算的损失是一个批量样本的总和，所以我们用批量大小（`batch_size`） 来规范化步长，这样步长大小就不会取决于我们对批量大小的选择。

```python
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():#这里很重要！在更新参数的时候不能计算梯度
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

#### 2.17训练

现在我们已经准备好了模型训练所有需要的要素，可以实现主要的训练过程部分了。 理解这段代码至关重要，因为从事深度学习后， 相同的训练过程几乎一遍又一遍地出现。 在每次迭代中，我们读取一小批量训练样本，并通过我们的模型来获得一组预测。 计算完损失后，我们开始反向传播，存储每个参数的梯度。 最后，我们调用优化算法`sgd`来更新模型参数。

```python
lr = 0.03
#学习率
num_epochs = 3
#扫描数据集的次数
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  
        # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

'''
epoch 1, loss 0.043705
epoch 2, loss 0.000172
epoch 3, loss 0.000047
'''
```

因为我们使用的是自己合成的数据集，所以我们知道真正的参数是什么。 因此，我们可以通过比较真实参数和通过训练学到的参数来评估训练的成功程度。 事实上，真实参数和通过训练学到的参数确实非常接近。

```python
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

'''
w的估计误差: tensor([ 0.0003, -0.0002], grad_fn=<SubBackward0>)
b的估计误差: tensor([0.0010], grad_fn=<RsubBackward1>)
'''
```

### 2.2使用pytorch框架直接实现

#### 2.21生成数据集

跟上面的一样，不再赘述了。

#### 2.22读取数据集

我们可以调用框架中现有的API来读取数据。 我们将`features`和`labels`作为API的参数传递，并通过数据迭代器指定`batch_size`。 此外，布尔值`is_train`表示是否希望数据迭代器对象在每个迭代周期内打乱数据。

```python
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

使用`data_iter`的方法与上面一小节的一致。展示一下使用python的`iter`即迭代器对象来访问的方法。

```python
next(iter(data_iter))
'''
[tensor([[ 0.1554, -0.2034],
         [-0.2140,  1.0352],
         [-0.4209,  0.0428],
         [ 0.1887,  0.6141],
         [ 0.4987, -0.2314],
         [ 0.0653,  1.6406],
         [-1.1881,  0.2900],
         [-0.2824,  0.5910],
         [ 0.9963, -0.1816],
         [-1.6830, -1.3963]]),
 tensor([[ 5.2116],
         [ 0.2479],
         [ 3.2188],
         [ 2.4845],
         [ 5.9884],
         [-1.2453],
         [ 0.8441],
         [ 1.6217],
         [ 6.8072],
         [ 5.5692]])]
'''
```

#### 2.23定义模型

对于标准深度学习模型，我们可以使用框架的预定义好的层。这使我们只需关注使用哪些层来构造模型，而不必关注层的实现细节。

首先定义一个模型变量`net`，它是一个`Sequential`类的实例。 `Sequential`类将多个层串联在一起。 当给定输入数据时，`Sequential`实例将数据传入到第一层， **然后将第一层的输出作为第二层的输入，以此类推**。 在下面的例子中，我们的模型只包含一个层，因此实际上不需要`Sequential`。 但是由于以后几乎所有的模型都是多层的，在这里使用`Sequential`会让你熟悉“标准的流水线”。

在PyTorch中，全连接层在`Linear`类中定义。 值得注意的是，我们将两个参数传递到`nn.Linear`中。 **第一个指定输入特征形状**，即2，**第二个指定输出特征形状**，输出特征形状为单个标量，因此为1。

```python
from torch import nn
# nn是神经网络的缩写

net = nn.Sequential(nn.Linear(2, 1))
```

#### 2.24初始化模型参数

正如我们在构造`nn.Linear`时指定输入和输出尺寸一样， 现在我们能直接访问参数以设定它们的初始值。 我们通过`net[0]`选择网络中的第一个图层， 然后使用`weight.data`和`bias.data`方法访问参数。 我们还可以使用替换方法`normal_`和`fill_`来重写参数值。

```python
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

#### 2.25定义损失函数

计算均方误差使用的是`MSELoss`类，也称为平方$L_2$范数。默认情况下，它返回所有样本损失的平均值。直接用`nn.MSELoss()`就可以定义了。~~(pytorch真的是打工仔的福利啊)~~

```python
loss = nn.MSELoss()
```

#### 2.26定义优化算法

小批量随机梯度下降算法是一种优化神经网络的标准工具， PyTorch在`optim`模块中实现了该算法的许多变种。 当我们实例化一个`SGD`实例时，我们要指定优化的参数 （可通过`net.parameters()`从我们的模型中获得）以及优化算法所需的超参数字典。 小批量随机梯度下降只需要设置`lr`值，这里设置为0.03。

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

~~简洁程度真的有点震撼到我了，pytorch牛逼！~~

#### 2.27训练

通过深度学习框架的高级API来实现我们的模型只需要相对较少的代码。 我们**不必单独分配参数、不必定义我们的损失函数，也不必手动实现小批量随机梯度下降**。 当我们需要更复杂的模型时，高级API的优势将大大增加。 当我们有了所有的基本组件，训练过程代码与我们从零开始实现时所做的非常相似。

回顾一下：在每个迭代周期里，我们将完整遍历一次数据集（`train_data`）， 不停地从中获取一个小批量的输入和相应的标签。 对于每一个小批量，我们会进行以下步骤:

- 通过调用`net(X)`生成预测并计算损失`l`（前向传播）。
- 通过进行反向传播来计算梯度。
- 通过调用优化器来更新模型参数。

为了更好的衡量训练效果，我们计算每个迭代周期后的损失，并打印它来监控训练过程。

```python
num_epochs = 3
#遍历数据集次数
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        #一个批次的损失
        trainer.zero_grad()
        #梯度清零
        l.backward()
        #求解这一个批次的样本的导数和
        trainer.step()
        # 以求得的导数和，结合优化器，更新参数w和b，然后进行下一批次的训练
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

'''
epoch 1, loss 0.000183
epoch 2, loss 0.000101
epoch 3, loss 0.000101
'''
```

下面我们比较生成数据集的真实参数和通过有限数据训练获得的模型参数。 要访问参数，我们首先从`net`访问所需的层，然后读取该层的权重和偏置。 正如在从零开始实现中一样，我们估计得到的参数与生成数据的真实参数非常接近。

```python
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)

'''
w的估计误差： tensor([-0.0003, -0.0002])
b的估计误差： tensor([8.1062e-06])
'''
```

