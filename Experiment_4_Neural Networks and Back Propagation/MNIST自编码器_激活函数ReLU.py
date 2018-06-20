import numpy as np    
import struct    
     
def loadImageSet(filename):    
    print("load image set",filename)   
    binfile= open(filename, 'rb')    
    buffers = binfile.read()    
     
    head = struct.unpack_from('>IIII' , buffers ,0)    
    print("head,",head)  
     
    offset = struct.calcsize('>IIII')    
    imgNum = head[1]    
    width = head[2]    
    height = head[3]    
    #[60000]*28*28    
    bits = imgNum * width * height    
    bitsString = '>' + str(bits) + 'B' #like '>47040000B'    
     
    imgs = struct.unpack_from(bitsString,buffers,offset)    
     
    binfile.close()    
    imgs = np.reshape(imgs,[imgNum,1,width*height])    
    print("load imgs finished")   
    return imgs    
     
def loadLabelSet(filename):    
     
    print("load label set",filename)  
    binfile = open(filename, 'rb')    
    buffers = binfile.read()    
     
    head = struct.unpack_from('>II' , buffers ,0)    
    print("head,",head)   
    imgNum=head[1]    
     
    offset = struct.calcsize('>II')    
    numString = '>'+str(imgNum)+"B"    
    labels = struct.unpack_from(numString , buffers , offset)    
    binfile.close()    
    labels = np.reshape(labels,[imgNum,1])    
     
    print('load label finished')   
    return labels    
     
#前项传播计算函数  
def layerComputing(inputNum,weightNum,biaseNum):  
    outputNum = ReLU(np.dot(inputNum,weightNum) + biaseNum)  
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
  
if __name__ == '__main__':  
    #导入MNIST数据集  
    imgs = loadImageSet("train-images-idx3-ubyte")    
    labels = loadLabelSet("train-labels-idx1-ubyte")   
    test_imgs = loadImageSet("t10k-images-idx3-ubyte")  
    test_labels = loadLabelSet("t10k-labels-idx1-ubyte")  
    #定义步长  
    step = 0.01  
    #训练集以及测试集的个数  
    train_num = 60000  
    test_num = 10000  
    #输入层神经元数目  
    input_num = 784  
    #输出层神经元数目  
    output_num = 784  
    #调整矩阵分布  
    distribution = 0.008  
    #定义隐藏层神经元  
    layer1_num = 64  
    layer1 = (2*np.random.randn(1,layer1_num)-1)*distribution  
    layer1_weight = (2*np.random.randn(input_num,layer1_num)-1)*distribution  
    layer1_biase = (2*np.random.randn(1,layer1_num)-1)*distribution  
      
      
    #定义输出层神经元  
    layer_out = (2*np.random.randn(1,output_num)-1)*distribution  
    out_weight = (2*np.random.randn(layer1_num,output_num)-1)*distribution  
    out_biase = (2*np.random.randn(1,output_num)-1)*distribution  
      
      
    #train  
    for i in range(3):  
        for j in range(train_num):  
            layer_in  = np.zeros((1,input_num))  
            for k in range(input_num):  
                layer_in[0,k] = imgs[j,0,k]  
              
            #正向计算  
            layer1 = layerComputing(layer_in,layer1_weight,layer1_biase)   
            layer_out = layerComputing(layer1,out_weight,out_biase)  
  
            #计算损失函数   
            loss = Loss(layer_out,layer_in)  
          
            out_loss = sign(layer_out) * (layer_out - layer_in)    
            layer1_loss = np.dot(out_loss,out_weight.T) *sign(layer1)  
            #in_loss = np.dot(layer1_loss,layer1_weight.T) * sign(layer_in)  
              
            #计算权重的偏导数   
            layer1_W_out_loss = np.dot(layer1.T,out_loss)  
            in_W_layer1_loss = np.dot(layer_in.T,layer1_loss)  
            #计算biase的偏导数  
            layer1_B_out_loss = out_loss  
            in_B_layer1_loss = layer1_loss  
            #更新weight和biase  
              
            layer1_weight = layer1_weight - step * in_W_layer1_loss     
            out_weight = out_weight - step * layer1_W_out_loss   
  
            layer1_biase = layer1_biase - step * in_B_layer1_loss    
            out_biase = out_biase - step * layer1_B_out_loss   
    
    #自编码器输出
    #选择测试集中的第1000副图片输出
    n = 1000
    layer_in  = np.zeros((1,input_num))  
    for k in range(input_num):  
        layer_in[0,k] = test_imgs[n,0,k]  
    layer1 = layerComputing(layer_in,layer1_weight,layer1_biase)  
    layer_out = layerComputing(layer1,out_weight,out_biase)  
    image_in = layer_in
    image_out = layer_out
    print(image_in)
    print(image_out)

