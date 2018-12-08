import numpy as np
import matplotlib.pyplot as plt

#输入数据
X = np.array([[1,0,0,0,0,0],[1,0,1,0,0,1],[1,1,0,1,0,0],[1,1,1,1,1,1]])
#标签
Y = np.array([-1,1,1,-1])
#权值初始化，1行3列，取值范围-1到1
W = (np.random.random(6)-0.5)*2
print(W)
#学习率设置
lr = 0.11
#计算迭代次数
n = 0
#神经网络输出
O = 0

def update():
    global X,Y,W,lr,n
    n+=1
    O = np.dot(X,W.T)
    W_C = lr*((Y-O.T).dot(X))/int(X.shape[0])
    W = W + W_C


# 因为学习规则公式的收敛性，通过不断循环，就能试验出方程的的解（权值w就是解，有了权值w就能完成分类公式，从而获取神经网络）
# 因为数据的局限性，所以解并不唯一，这也像我们平时学习一样，学习的数据越多，分类越准确
for _ in range(1000):
    update()

# 画图

# 正样本
x1 = [0, 1]
y1 = [1, 0]

# 负样本
x2 = [0, 1]
y2 = [0, 1]


def calculate(x, root):
    a = W[5]
    b = W[2] + x * W[4]
    c = W[0] + x * W[1] + x * x * W[3]
    if root == 1:
        return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    if root == 2:
        return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)


# 计算分界线的斜率以及截距
k = -W[1] / W[2]
d = -W[0] / W[2]
xdata = np.linspace(-1, 2)

plt.figure()
plt.plot(xdata, calculate(xdata, 1), "r")
plt.plot(xdata, calculate(xdata, 2), "r")
plt.plot(x1, y1, "bo")
plt.plot(x2, y2, "yo")
plt.show()

