# 小白深度学习Day2

## 1、SoftMax前导知识

### 1.1 分类问题

主要就是引入独热编码，比如猫、鸡、狗三类，分别对应的三维向量是：

$(1,0,0),(0,1,0),(0,0,1)$

### 1.2 网络架构

为了估计所有可能类别的条件概率，我们需要一个有多个输出的模型，每个类别对应一个输出。 为了解决线性模型的分类问题，我们需要和输出一样多的*仿射函数*（affine function）。 每个输出对应于它自己的仿射函数。 在我们的例子中，由于我们有4个特征和3个可能的输出类别，我们将需要12个标量来表示权重(带下标的$w$), 3个标量来表示偏置（带下标的$b$）。下面我们为每个输入计算三个*未规范化的预测*（logit）:$o_1,o_2,o_3$

$\begin{matrix}o_1=x_1w_{11}+x_2w_{12}+x_3w_{13}+x_4w_{14}+b_1,\\ o_2=x_1w_{21}+x_2w_{22}+x_3w_{23}+x_4w_{24}+b_2,\\ o_3=x_1w_{31}+x_2w_{32}+x_3w_{33}+x_4w_{34}+b_3.\end{matrix}$

![../_images/softmaxreg.svg](https://zh-v2.d2l.ai/_images/softmaxreg.svg)

为了更简洁地表达模型，我们仍然使用线性代数符号。 通过向量形式表达为$o=Wx+b$

这是一种更适合数学和编写代码的形式。 由此，我们已经将所有权重放到一个3×4矩阵中。 对于给定数据样本的特征$x$,我们的输出是由权重与输入特征进行矩阵-向量乘法再加上偏置b得到的。

### 1.3全连接层的参数开销

全连接层无处不在。 然而，顾名思义，全连接层是“完全”连接的，可能有很多可学习的参数。 具体来说，对于任何具有$d$个输入，$q$个输出的全连接层，参数开销为$O(dq)$,这个数字在实践中可能高得令人望而却步。幸运的是，将$d$个输入转换为$q$个输出成本可以减少到$O(dq/n)$，其中超参数$n$可以由我们灵活指定，以在实际应用中平衡参数节约和模型有效性。

### 1.4 SoftMax运算

$ \widehat{y}=softmax\left( o \right) \,\,\widehat{y_j}=\frac{e^{o_j}}{\sum_k{exp\left( o_k \right)}} $

这里，对于所有的$j$总有$ 0\leqslant \widehat{y_j}\leqslant 1 $.softmax运算不会改变未规范化的预测$o$之间的大小次序，只会确定分配给每个类别的概率。 因此，在预测过程中，我们仍然可以用下式来选择最有可能的类别。即：
$$
\underset{j}{arg\max}\widehat{y_j}=\underset{j}{arg\max}o_j
$$
尽管softmax是一个非线性函数，但softmax回归的输出仍然由输入特征的仿射变换决定。 因此，softmax回归是一个*线性模型*（linear model）。

### 1.5 小批量样本的矢量化

为了提高计算效率并且充分利用GPU，我们通常会对小批量样本的数据执行矢量计算。 假设我们读取了一个批量的样本$X$,其中特征维度（输入数量）为$d$,批量大小为$n$,此外，假设我们在输出中有$q$个类别。 那么小批量样本的特征为$ X\in R^{n\times d} $(n个样本，每个样本d个特征),权重为$ W\in R^{d\times q} $(每个样本d个特征，共有q个类别)，偏置为$ b\in R^{1\times q} $。softmax回归的矢量计算表达式为：
$$
O=XW+b
\\
\widehat{Y}=softmax\left( O \right)
$$
相对于一次处理一个样本， 小批量样本的矢量化加快了$X$和$W$的矩阵-向量乘法。 由于$X$中的每一行代表一个数据样本， 那么softmax运算可以*按行*（rowwise）执行：对于$O$的每一行，我们先对所有项进行幂运算，然后通过求和对它们进行标准化。

### 1.6损失函数

接下来，我们需要一个损失函数来度量预测的效果。 我们将使用最大似然估计。对于任何标签$y$和模型预测$
\widehat{y}
$，损失函数为：
$$
l\left( y,\widehat{y} \right) =-\sum_{j=1}^q{y_j\log}\widehat{y_j}
$$
这就是所说的交叉熵损失。

将$
\widehat{y_j}
$的定义式带进去，可以进一步得到：
$$
l\left( y,\widehat{y} \right) =-\sum_{j=1}^q{\begin{array}{c}
	\begin{array}{c}
	y_j\log \frac{\exp \left( o_j \right)}{\sum_{k=1}^q{\exp \left( o_k \right)}}\\
\end{array}\\
\end{array}}
\\
=\sum_{j=1}^q{y_j\log \sum_{k=1}^q{\exp \left( o_k \right)}}-\sum_{j=1}^q{y_jo_j}
\\
=\log \sum_{k=1}^q{\exp \left( o_k \right)}-\sum_{j=1}^q{y_jo_j}
$$
对于$o_j$取对数，我们可以得到：
$$
\partial _{o_j}l\left( y,\widehat{y} \right) =\frac{\exp \left( o_j \right)}{\sum_{k=1}^q{\exp \left( o_k \right)}}-y_j=soft\max \left( o \right) _j-y_j
$$
换句话说，导数是我们softmax模型分配的概率与实际发生的情况（由独热标签向量表示）之间的差异。 从这个意义上讲，这与我们在回归中看到的非常相似， 其中梯度是观测值y和估计值y^之间的差异。 这不是巧合，在任何指数族分布模型中， 对数似然的梯度正是由此得出的。 这使梯度计算在实践中变得容易很多。~~(数学家牛逼！)~~

## 2、图像分类数据集准备

使用到的是`Fashion-MNIST`数据集。

下载数据集：

```python
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

可视化样本

```python
'''可视化样本'''
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```



## 3、从0开始实现softmax回归

### 3.1初始化模型参数

原始数据集中的每个样本都是$28×28$的图像。 本节将展平每个图像，把它们看作长度为784的向量。

回想一下，在softmax回归中，我们的输出与类别一样多。 因为我们的数据集有10个类别，所以网络输出维度为10。 因此，权重将构成一个$784×10$的矩阵， 偏置将构成一个$1×10$的行向量。 与线性回归一样，我们将使用正态分布初始化我们的权重`W`，偏置初始化为0。

```python
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

### 3.2 定义SoftMax操作

回想一下，实现softmax由三个步骤组成：

- 对每个项求幂（使用`exp`）；
- 对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；
- 将每一行除以其规范化常数，确保结果的和为1。

ok,所以可以编写以下的代码：

```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
```

正如上述代码，对于任何随机输入，我们将每个元素变成一个非负数。 此外，依据概率原理，每行总和为1。

### 3.3 定义模型

定义softmax操作后，我们可以实现softmax回归模型。 下面的代码定义了输入如何通过网络映射到输出。 注意，将数据传递到模型之前，我们使用`reshape`函数将每张原始图像展平为向量。

```python
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

### 3.4 定义损失函数

引入交叉熵损失函数即可。回顾一下，交叉熵采用真实标签的预测概率的负对数似然。 这里我们不使用Python的for循环迭代预测（这往往是低效的）， 而是通过一个运算符选择所有元素。 下面，我们创建一个数据样本`y_hat`，其中包含2个样本在3个类别的预测概率， 以及它们对应的标签`y`。 有了`y`，我们知道在第一个样本中，第一类是正确的预测； 而在第二个样本中，第三类是正确的预测。 然后使用`y`作为`y_hat`中概率的索引， 我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率。

```python
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]

'''tensor([0.1000, 0.5000])'''
```

然后就可以计算交叉熵损失函数了：

```python
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
'''tensor([2.3026, 0.6931])'''
```

### 3.5 分类精度

当预测与标签分类`y`一致时，即是正确的。 分类精度即正确预测数量与总预测数量之比。 虽然直接优化精度可能很困难（因为精度的计算不可导）， 但精度通常是我们最关心的性能衡量标准，我们在训练分类器时几乎总会关注它。

为了计算精度，我们执行以下操作。 首先，如果`y_hat`是矩阵，那么假定第二个维度存储每个类的预测分数。 我们使用`argmax`获得每行中最大元素的索引来获得预测类别。 然后我们将预测类别与真实`y`元素进行比较。 由于等式运算符“`==`”对数据类型很敏感， 因此我们将`y_hat`的数据类型转换为与`y`的数据类型一致。 结果是一个包含0（错）和1（对）的张量。 最后，我们求和会得到正确预测的数量。

```python
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
```

我们将继续使用之前定义的变量`y_hat`和`y`分别作为预测的概率分布和标签。 可以看到，第一个样本的预测类别是2（该行的最大元素为0.6，索引为2），这与实际标签0不一致。 第二个样本的预测类别是2（该行的最大元素为0.5，索引为2），这与实际标签2一致。 因此，这两个样本的分类精度率为0.5。

```python
accuracy(y_hat, y) / len(y)
'''0.5'''
```

这里定义一个实用程序类`Accumulator`，用于对多个变量进行累加。 在上面的`evaluate_accuracy`函数中， 我们在`Accumulator`实例中创建了2个变量， 分别用于存储正确预测的数量和预测的总数量。 当我们遍历数据集时，两者都将随着时间的推移而累加。

```python
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

对于任意数据迭代器`data_iter`可访问的数据集， 我们可以评估在任意模型`net`的精度。

```python
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

由于我们使用随机权重初始化`net`模型， 因此该模型的精度应接近于随机猜测。 例如在有10个类别情况下的精度为0.1。

```python
evaluate_accuracy(net, test_iter)
#0.0598
```

### 3.6 训练(代码有误，跑不通)

```python
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
```

## 4、softmax回归的简洁实现

### 4.1 加载数据集

```python
import torch
from torch import nn

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
```

### 4.2 初始化模型参数

softmax回归的输出层是一个全连接层。 因此，为了实现我们的模型， 我们只需在`Sequential`中添加一个带有10个输出的全连接层。 同样，在这里`Sequential`并不是必要的， 但它是实现深度模型的基础。 我们仍然以均值0和标准差0.01随机初始化权重。

```python
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

> nn.init.normal_函数解释：
>
> torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
>
> - **tensor**-一个n维`torch.Tensor`
> - **mean**-正态分布的平均值
> - **std**-正态分布的标准差
>
> 使用从正态分布 N(mean,std) 中提取的值填充输入张量。

>net.apply函数解释：
>
>pytorch中的`model.apply(fn)`会递归地将函数`fn`应用到父模块的每个子模块`submodule`，也包括`model`这个父模块自身。

### 4.3 定义损失函数

```python
loss = nn.CrossEntropyLoss(reduction='none')
```

### 4.4 模型的训练(可以跑通)

1、计算预测正确的数量:

```python
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
         y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
```

2、计算在指定数据集上模型的精度:

```python
def evaluate_accuracy(net,data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net,torch.nn.Module):
        # 将模型设置为评估模式
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y), y.numel())
    return metric[0] / metric[1]
```

3、每次训练的流程

```python
def epoch_train(net,train_iter,loss,updater):
    # 将模型设置为训练模式
    if isinstance(net,torch.nn.Module):
        net.train()

    # 指标：训练损失总和，训练准确度总和，样本数
    metric = Accumulator(3)
    for X,y in train_iter:
        # 计算梯度，并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)

        if isinstance(updater, torch.optim.Optimizer):
            # 使用pytorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])

        metric.add(
            float(l.sum()),
            accuracy(y_hat,y),
            y.numel()
        )

    # 返回训练损失和训练精度
    return (metric[0] / metric[2],metric[1] / metric[2])
```

