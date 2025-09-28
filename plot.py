from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

#下载数据并划分数据，放在data文件夹里

#向量下载数据
train_data=FashionMNIST(root='./data',
                        train=True,
                        transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),   #改变图片大小归一化处理
                        download=True)
#把数据打成64个一捆并打乱
train_loader=data.DataLoader(dataset=train_data,
                             batch_size=64,
                             shuffle=True,
                             num_workers=0)

#获得一个Batch(一捆数据)数据
for step,(b_x,b_y) in enumerate(train_loader):
    if step>0:
        break
batch_x=b_x.squeeze().numpy()  #移除最后一维(通道数)，转换成Numpy数组,方便画图
batch_y=b_y.numpy()   #将张量转化成一个Numpy数组
class_label=train_data.classes   #训练集标签
print(class_label)
print("The size of batch in train data: ",batch_x.shape)

#可视化一个Batch的图像
plt.figure(figsize=(12,5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4,16,ii+1)
    plt.imshow(batch_x[ii,:,:],cmap='gray')
    plt.title(class_label[batch_y[ii]],size=10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05)
plt.show()