from sklearn.datasets import load_digits
import pylab as pl

digits = load_digits()#载入数据集
print(digits.data.shape)

pl.gray()#灰度化图片
pl.matshow(digits.images[2])
pl.show()