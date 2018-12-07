import numpy as np
import matplotlib.pyplot as plt

#输入数据
X = np.array([[1,3,3],[1,4,3],[1,1,1]])
#标签
Y = np.array([1,1,-1])
#权值初始化，1行3列，取值范围-1到1
W = (np.random.random(3)-0.5)*2
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
    O = np.sign(np.dot(X,W.T))
    W_C = lr*((Y-O.T).dot(X))/int(X.shape[0])
    W = W + W_C


# 因为学习规则公式的收敛性，通过不断循环，就能试验出方程的的解（权值w就是解，有了权值w就能完成分类公式，从而获取神经网络）
# 因为数据的局限性，所以解并不唯一，这也像我们平时学习一样，学习的数据越多，分类越准确
for _ in range(100):
    update()
    print(W)
    print(n)
    O = np.sign(np.dot(X, W.T))
    if (O == Y).all():
        print("sussceful")
        break

# 画图

# 正样本
x1 = [3, 4]
y1 = [3, 3]

# 负样本
x2 = [1]
y2 = [1]

# 计算分界线的斜率以及截距
k = -W[1] / W[2]
d = -W[0] / W[2]
xdata = np.linspace(0, 5)

plt.figure()
plt.plot(xdata, xdata * k + d, "r")
plt.plot(x1, y1, "bo")
plt.plot(x2, y2, "yo")
plt.show()

