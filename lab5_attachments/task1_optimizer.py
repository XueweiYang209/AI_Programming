"""
本文件我们尝试实现一个Optimizer类，用于优化一个简单的双层Linear Network
本次作业主要的内容将会在opti_epoch内对于一个epoch的参数进行优化
分为SGD_epoch和Adam_epoch两个函数，分别对应SGD和Adam两种优化器
其余函数为辅助函数，也请一并填写
和大作业的要求一致，我们不对数据处理和读取做任何要求
因此你可以引入任何的库来帮你进行数据处理和读取
理论上我们也不需要依赖lab5的内容，如果你需要的话，你可以将lab5对应代码copy到对应位置
"""
# from task0_autodiff import *
# from task0_operators import *
import numpy as np
import torch
from torchvision import datasets, transforms

def parse_mnist():
    """
    读取MNIST数据集，并进行简单的处理，如归一化
    你可以可以引入任何的库来帮你进行数据处理和读取
    所以不会规定你的输入的格式
    但需要使得输出包括X_tr, y_tr和X_te, y_te
    """
    ## 请于此填写你的代码

    # 加载MNIST数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)

    X_tr = train_dataset.data.numpy().reshape(-1, 28*28).astype(np.float32)
    y_tr = train_dataset.targets.numpy()
    
    X_te = test_dataset.data.numpy().reshape(-1, 28*28).astype(np.float32)
    y_te = test_dataset.targets.numpy()

    X_tr /= 255.0
    X_te /= 255.0
    return X_tr, y_tr, X_te, y_te

def set_structure(n, hidden_dim, k):
    """
    定义你的网络结构，并进行简单的初始化
    一个简单的网络结构为两个Linear层，中间加上ReLU
    Args:
        n: input dimension of the data.
        hidden_dim: hidden dimension of the network.
        k: output dimension of the network, which is the number of classes.
    Returns:
        List of Weights matrix.
    Example:
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    return list(W1, W2)
    """

    ## 请于此填写你的代码
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    return [W1,W2]

def forward(X, weights):
    """
    使用你的网络结构，来计算给定输入X的输出
    Args:
        X : 2D input array of size (num_examples, input_dim).
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
    Returns:
        Logits calculated by your network structure.
    Example:
    W1 = weights[0]
    W2 = weights[1]
    return np.maximum(X@W1,0)@W2
    """
    ## 请于此填写你的代码
    W1 = weights[0]
    W2 = weights[1]
    return np.maximum(X@W1,0)@W2

def softmax_loss(Z, y):
    """ 
    一个写了很多遍的Softmax loss...

    Args:
        Z : 2D numpy array of shape (batch_size, num_classes), 
        containing the logit predictions for each class.
        y : 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ## 请于此填写你的代码
    batch_size = Z.shape[0]

    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # 为了数值稳定性，减去最大值
    softmax_probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    log_probs = np.log(softmax_probs)
    correct_log_probs = log_probs[np.arange(batch_size), y]

    loss = -np.mean(correct_log_probs)

    return loss

def opti_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """
    优化一个epoch
    具体请参考SGD_epoch 和 Adam_epoch的代码
    """
    if using_adam:
        Adam_epoch(X, y, weights, lr = lr, batch=batch, beta1=beta1, beta2=beta2)
    else:
        SGD_epoch(X, y, weights, lr = lr, batch=batch)

def SGD_epoch(X, y, weights, lr = 0.1, batch=100):
    """ 
    SGD优化一个List of Weights
    本函数应该inplace地修改Weights矩阵来进行优化
    用学习率简单更新Weights

    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ## 请于此填写你的代码
    num_examples = X.shape[0]
    
    indices = np.random.permutation(num_examples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # 每次迭代对minibatch个数据做梯度下降
    for i in range(0, num_examples, batch):
        X_batch = X_shuffled[i:i+batch]
        y_batch = y_shuffled[i:i+batch]
        
        Z = forward(X_batch, weights)
        
        # 反向传播计算梯度
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        softmax_probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        dZ = softmax_probs
        dZ[np.arange(batch), y_batch] -= 1
        dZ /= batch
        
        dW2 = np.maximum(X_batch @ weights[0], 0).T @ dZ
        
        dA = dZ @ weights[1].T
        dZ1 = dA * (X_batch @ weights[0] > 0)
        
        dW1 = X_batch.T @ dZ1
        
        # 更新参数
        weights[1] -= lr * dW2
        weights[0] -= lr * dW1

def Adam_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999):
    """ 
    ADAM优化一个
    本函数应该inplace地修改Weights矩阵来进行优化
    使用Adaptive Moment Estimation来进行更新Weights
    具体步骤可以是：
    1. 增加时间步 $t$。
    2. 计算当前梯度 $g$。
    3. 更新一阶矩向量：$m = \beta_1 \cdot m + (1 - \beta_1) \cdot g$。
    4. 更新二阶矩向量：$v = \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$。
    5. 计算偏差校正后的一阶和二阶矩估计：$\hat{m} = m / (1 - \beta_1^t)$ 和 $\hat{v} = v / (1 - \beta_2^t)$。
    6. 更新参数：$\theta = \theta - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$。
    其中$\eta$表示学习率，$\beta_1$和$\beta_2$是平滑参数，
    $t$表示时间步，$\epsilon$是为了维持数值稳定性而添加的常数，如1e-8。
    
    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
        beta1 (float): smoothing parameter for first order momentum
        beta2 (float): smoothing parameter for second order momentum

    Returns:
        None
    """
    ## 请于此填写你的代码
    num_examples = X.shape[0]
    epsilon = 1e-8
    
    # 初始化m,v
    m1 = np.zeros_like(weights[0])
    m2 = np.zeros_like(weights[1])
    v1 = np.zeros_like(weights[0])
    v2 = np.zeros_like(weights[1])
    
    t = 0
    
    indices = np.random.permutation(num_examples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # 每次迭代对minibatch个数据做优化
    for i in range(0, num_examples, batch):
        X_batch = X_shuffled[i:i+batch]
        y_batch = y_shuffled[i:i+batch]
        
        Z = forward(X_batch, weights)
        
        # 反向传播计算优化参数
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        softmax_probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        dZ = softmax_probs
        dZ[np.arange(batch), y_batch] -= 1
        dZ /= batch
        
        dW2 = np.maximum(X_batch @ weights[0], 0).T @ dZ
        
        dA = dZ @ weights[1].T
        dZ1 = dA * (X_batch @ weights[0] > 0)
        
        dW1 = X_batch.T @ dZ1
        
        t += 1
        
        m1 = beta1 * m1 + (1 - beta1) * dW1
        m2 = beta2 * m2 + (1 - beta2) * dW2
        
        v1 = beta2 * v1 + (1 - beta2) * dW1**2
        v2 = beta2 * v2 + (1 - beta2) * dW2**2
        
        m1_hat = m1 / (1 - beta1**t)
        m2_hat = m2 / (1 - beta2**t)
        
        v1_hat = v1 / (1 - beta2**t)
        v2_hat = v2 / (1 - beta2**t)
        
        # Update weights using Adam
        weights[0] -= lr * m1_hat / (np.sqrt(v1_hat) + epsilon)
        weights[1] -= lr * m2_hat / (np.sqrt(v2_hat) + epsilon)


def loss_err(h,y):
    """ 
    计算给定预测结果h和真实标签y的loss和error
    """
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """ 
    训练过程
    """
    n, k = X_tr.shape[1], y_tr.max() + 1
    weights = set_structure(n, hidden_dim, k)
    np.random.seed(0)
    

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        opti_epoch(X_tr, y_tr, weights, lr=lr, batch=batch, beta1=beta1, beta2=beta2, using_adam=using_adam)
        train_loss, train_err = loss_err(forward(X_tr, weights), y_tr)
        test_loss, test_err = loss_err(forward(X_te, weights), y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = parse_mnist() 
    weights = set_structure(X_tr.shape[1], 100, y_tr.max() + 1)
    ## using SGD optimizer 
    train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=20, lr = 0.001, batch=100, beta1=0.9, beta2=0.999, using_adam=False)
    ## using Adam optimizer
    # train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=20, lr = 0.001, batch=100, beta1=0.9, beta2=0.999, using_adam=True)
    