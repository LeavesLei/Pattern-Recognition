from sklearn import datasets
import numpy as np

#导入IRIS数据集
iris = datasets.load_iris()
x = iris.data
y = iris.target

#返回类别值
a = np.unique(y)

#划分训练集和测试集，各占50%
#x_tr为训练集，x_te为测试集
from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size=0.5,random_state=0)

#数据标准化，加快收敛速度
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(x_tr)#计算方差
x_tr_s = ss.transform(x_tr)
x_te_s = ss.transform(x_te)

#导入sklearn中的感知器,进行建模
from sklearn.linear_model import Perceptron
ptn = Perceptron(n_iter=100,eta0=0.1,random_state=0)#迭代200次，learning rate为0.1
ptn.fit(x_tr_s,y_tr)

#预测并计算分类准确率
y_predict = ptn.predict(x_te_s)

from sklearn.metrics import accuracy_score
print('准确率：%.3f' %accuracy_score(y_te,y_predict))
