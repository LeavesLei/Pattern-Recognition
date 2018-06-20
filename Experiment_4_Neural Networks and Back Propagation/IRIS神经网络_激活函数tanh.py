import numpy as np	
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

#前项传播计算函数
def layerComputing(inputNum,weightNum,biaseNum):
	outputNum = tanh(np.dot(inputNum,weightNum) + biaseNum)
	return outputNum

#sigmoid激活函数
def sigmoid(n):
	for i in range(np.size(n)):
		n[0,i] = 1/(1+np.exp(-n[0,i]))
	return n

#ReLU激活函数
def ReLU(n):
	for i in range(np.size(n)):
		if(n[0,i]<=0):
			n[0,i] = 0
	return n

#tanh激活函数
def tanh(n):
	for i in range(np.size(n)):
		n[0,i] = np.tanh(n[0,i])
	return n
	
#损失函数
def Loss(output,target):
	loss = target - output
	
	loss = loss * loss
	n = 0
	for i in range(np.size(loss)):
		n = n + loss[0,i]
	n = n/2
	return n

#信号函数
def sign(n):
	for i in range(np.size(n)):
		if(n[0,i]<=0):
			n[0,i] = 0
		else:
			n[0,i] = 1
	return n

#将输出转化为属于某一类的形式
def out2class(n):
	#找出最大数的索引值
	order = np.argmax(n)
	return order

#独热码计算函数
def oneHot(n,outputNum):
	m = np.zeros((1,outputNum))
	m[0,n] = 1
	return m	

if __name__ == '__main__':
	#导入iris数据集
	iris=load_iris()  
	attributes=iris.data  #获取属性数据
	target=iris.target  #获取类别数据
	labels=iris.feature_names #获取列属性值
	#划分数据集和测试集
	test_proportion = 0.3#测试集所占比例
	train_proportion = 1 - test_proportion #训练集所占比例
	#X_train,Y_train分别是数据集的data和target，X_test，Y_target分别是测试集的data和target
	X_train, X_test, Y_train, Y_test = train_test_split(attributes, target, test_size=test_proportion)
	#定义步长
	step = 0.01
	#定义可变步长
	step1 = 0.008
	#输入层神经元数目
	input_num = 4
	#输出层神经元数目
	output_num = 3
	#调整矩阵分布
	distribution = 0.1
	#定义第一层神经元
	layer1_num = 10
	layer1 = (2*np.random.randn(1,layer1_num)-1)*distribution
	layer1_weight = (2*np.random.randn(input_num,layer1_num)-1)*distribution
	layer1_biase = (2*np.random.randn(1,layer1_num)-1)*distribution
    
	#定义第二层神经元
	layer2_num = 10
	layer2 =  (2*np.random.randn(1,layer2_num)-1)*distribution
	layer2_weight = (2*np.random.randn(layer1_num,layer2_num)-1)*distribution
	layer2_biase = (2*np.random.randn(1,layer2_num)-1)*distribution
    
	#定义输出层神经元
	layer_out = (2*np.random.randn(1,output_num)-1)*distribution
	out_weight = (2*np.random.randn(layer2_num,output_num)-1)*distribution
	out_biase = (2*np.random.randn(1,output_num)-1)*distribution
	
	#定义一个数组实现可变步长
	change_past = [0,0,0,0,0,0]
	#change_now = [0,0,0,0,0,0]
	
	#train
	for i in range(50):
		for j in range((int)(150*train_proportion)):
			layer_in  = np.zeros((1,input_num))
			for k in range(input_num):
				layer_in[0,k] = X_train[j,k]
				#layer_in[0,k] = 1
			
			#正向计算
			layer1 = layerComputing(layer_in,layer1_weight,layer1_biase)
			layer2 = layerComputing(layer1,layer2_weight,layer2_biase)
			layer_out = layerComputing(layer2,out_weight,out_biase)

			#计算损失函数
			one_hot_target = oneHot(Y_train[j],output_num)
			loss = Loss(layer_out,one_hot_target)
		
			out_loss = (1 - layer_out * layer_out) * (layer_out - one_hot_target)
			layer2_loss = np.dot(out_loss,out_weight.T) * (1 - layer2 * layer2)
			layer1_loss = np.dot(layer2_loss,layer2_weight.T) * (1- layer1 * layer1)
            
			#计算权重的偏导数
			layer2_W_out_loss = np.dot(layer2.T,out_loss)
			layer1_W_layer2_loss = np.dot(layer1.T,layer2_loss)
			in_W_layer1_loss = np.dot(layer_in.T,layer1_loss)
			#计算biase的偏导数
			layer2_B_out_loss = out_loss
			layer1_B_layer2_loss = layer2_loss
			in_B_layer1_loss = layer1_loss
			#更新weight和biase
			layer1_weight = layer1_weight - step * in_W_layer1_loss - step1 * change_past[0]
			layer2_weight = layer2_weight - step * layer1_W_layer2_loss - step1 * change_past[1]
			out_weight = out_weight - step * layer2_W_out_loss - step1 * change_past[2]

			layer1_biase = layer1_biase - step * in_B_layer1_loss - step1 * change_past[3]
			layer2_biase = layer2_biase - step * layer1_B_layer2_loss - step1 * change_past[4]
			out_biase = out_biase - step * layer2_B_out_loss - step1 * change_past[5]
			#更新可变步长调整值
			change_past[0] = in_W_layer1_loss
			change_past[1] = layer1_W_layer2_loss
			change_past[2] = layer2_W_out_loss
			change_past[3] = in_B_layer1_loss
			change_past[4] = layer1_B_layer2_loss
			change_past[5] = layer2_B_out_loss
	#test
	pass_num = 0
	all_num = 150*test_proportion

	#正向计算
	for j in range((int)(150*test_proportion)):
		layer_in  = np.zeros((1,input_num))
		for k in range(input_num):
			layer_in[0,k] = X_test[j,k]
		layer1 = layerComputing(layer_in,layer1_weight,layer1_biase)
		layer2 = layerComputing(layer1,layer2_weight,layer2_biase)
		layer_out = layerComputing(layer2,out_weight,out_biase)
		class_num = out2class(layer_out)
		if(class_num==Y_test[j]):
			pass_num = pass_num + 1

	accuracy = pass_num / all_num
	print("accuracy:",accuracy)			
