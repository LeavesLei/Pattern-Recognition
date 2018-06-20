import numpy as np	
import math
from sklearn.datasets import load_iris

#找出list中匹配字符的最大index
def maxIndex(num,numList):
    max_index = 0
    for i in range(len(numList)):
        if(num==numList[i]):
            max_index = i
    return max_index
#找出片段序列中的最大值和最小值
def maxAndMinValue(numberList,indexList):
    max_value = numberList[indexList[0]]
    min_value = numberList[indexList[0]]
    for i in indexList:
        if(numberList[i]>max_value):
            max_value = numberList[i]
        if(numberList[i]<min_value):
            min_value = numberList[i]
    return max_value,min_value

#当对数输入为0是，输出为0
def log(m,n):
    if(m==0):
        return 0
    else:
        return math.log(m,n)
#熵的计算,m为某种结果出现的次数，n为所有结果出现的次数
def entropy(n,m_0,m_1,m_2):
	#结果出现概率 p = m/n
    p_0 = float(m_0/n) 
    p_1 = float(m_1/n)
    p_2 = float(m_2/n)
    e = - (p_0 * log(p_0,2) + p_1 * log(p_1,2) + p_2 * log(p_2,2))
    return e

#计算出某一类中最小的熵，从而得到最大的information gain
def computing_min_entropy(label_name,include_index,step_width):
    order = labels.index(label_name)
    classed_attributes = attributes[:,order]
    min_entropy = 1000 #存储最小的熵
    max_value,min_value = maxAndMinValue(classed_attributes,include_index)
    ar_1 = np.arange(min_value+0.01,max_value-0.01,step_width)
    #存储分成两类的索引值
    branch_1_index = [];
    branch_2_index = [];
    #划分的临界值
    threshold = 0
    #分成两类的索引值
    branch1 = []
    branch2 = []
    matrix = [[0,0,0,0],[0,0,0,0]] 
    for i in ar_1:
        #存储分类后各项指标
        computing_matrix = [[0,0,0,0],[0,0,0,0]] 
        
        #存储分成两类的索引值
        branch_1_index = [];
        branch_2_index = [];
        for j in include_index:
            if(classed_attributes[j]<=i):
                branch_1_index.append(j)
                computing_matrix[0][0] +=1
                if(target[j]==0):
                	computing_matrix[0][1] +=1
                elif(target[j]==1):
                	computing_matrix[0][2] +=1
                else:
                	computing_matrix[0][3] +=1
            else:
                branch_2_index.append(j)
                computing_matrix[1][0] +=1
                if(target[j]==0):
                	computing_matrix[1][1] +=1
                elif(target[j]==1):
                	computing_matrix[1][2] +=1
                else:
                	computing_matrix[1][3] +=1
        #属于第一类与第二类的个数
        class1_num = computing_matrix[0][0]
        class2_num = computing_matrix[1][0]

        e1 = class1_num * entropy(computing_matrix[0][0],computing_matrix[0][1],
        	computing_matrix[0][2],computing_matrix[0][3])
        e2 = class2_num * entropy(computing_matrix[1][0],computing_matrix[1][1],
        	computing_matrix[1][2],computing_matrix[1][3])
     
        if((e1+e2)<min_entropy):
            min_entropy = e1 + e2
            #对阈值进行一个优化处理，使其有最大间隙
            m = 20
            for k in include_index:
            	if(classed_attributes[k]>i and classed_attributes[k]<m):
            		m = classed_attributes[k]
            threshold = (i-0.01+m)/2 #存储最佳分割阈值
            matrix = computing_matrix
            branch1 = branch_1_index
            branch2 = branch_2_index
   
    return matrix,label_name,threshold,min_entropy,branch1,branch2

#计算所有类的熵
def computingAllClass(labels,include_index):
    sepal_length = computing_min_entropy(labels[0],include_index,0.1)
    sepal_width = computing_min_entropy(labels[1],include_index,0.1)
    petal_length = computing_min_entropy(labels[2],include_index,0.1)
    petal_width = computing_min_entropy(labels[3],include_index,0.1)
    
    allClass = [sepal_length,sepal_width,petal_length,petal_width]
    allClassEntropy = [sepal_length[3],sepal_width[3],
                       petal_length[3],petal_width[3]]

    min_entropy = min(allClassEntropy)
    min_order = maxIndex(min_entropy,allClassEntropy)

    return allClass[min_order]

if __name__ == '__main__':
	#导入iris数据集
    iris=load_iris()  
    attributes=iris.data  #获取属性数据
    target=iris.target  #获取类别数据
    labels=iris.feature_names #获取列属性值
    #分类结果
    result = [[],[],[],[],[]]
    result[0].append(computingAllClass(labels,range(150)))
    
    for i in range(4):
        for j in range(len(result[i])):
            matrix = result[i][j][0]
   
            #检测分类后树枝中是否只剩下一类or为空
            if(matrix[0].count(0)>1):
                result[i+1].append([[[0,0],[0,0]]])
            else:
                result[i+1].append(computingAllClass(labels,result[i][j][-2]))
            if(matrix[1].count(0)>1):
                result[i+1].append([[[0,0],[0,0]]])
            else:
                result[i+1].append(computingAllClass(labels,result[i][j][-1]))
    #结果显示
    for i in range(5):
    	print("==========================")
    	for j in range(len(result[i])):
    		print("--------------------------")
    		if(result[i][j][0]!=[[0,0],[0,0]]):
    			print(result[i][j][1])
    			print(result[i][j][2])
    			print(result[i][j][-2])
    			print(result[i][j][-1])
    		else:
    			print("This branch is over")