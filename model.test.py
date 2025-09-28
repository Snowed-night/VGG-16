import torch
import torch.utils.data as data
from torchvision import  transforms
from torchvision.datasets import  FashionMNIST
from model import VGG16

#处理训练集和验证集
def test_data_process():
    #加载数据
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                             download=True)
    test_dataloader=data.DataLoader(dataset=test_data,
                                    batch_size=128,
                                    shuffle=False,
                                    num_workers=8)

    return test_dataloader

#模型测试
def test_model_process(model, test_dataloader):
    device="cuda" if torch.cuda.is_available() else "cpu"
    #将模型放入设备
    model=model.to(device)
    #初始化参数
    test_corrects=0    #预测正确数
    test_num=0         #样本数
    #将梯度置为0，表示只进行前向传播
    with torch.no_grad():
        #一个一个获取
        for test_data_x,test_data_y in test_dataloader:
            test_data_x=test_data_x.to(device)    #数据
            test_data_y=test_data_y.to(device)    #标签
            #设置为评估模式
            model.eval()
            output=model(test_data_x)
            pre_lab=torch.argmax(output,dim=1)   #模型算出最大概率下标

            test_corrects+=torch.sum(pre_lab==test_data_y.data)
            test_num+=test_data_x.size(0)
        #计算测试正确率
        test_acc=test_corrects.double().item() / test_num
        print("测试的准确率为：",test_acc)

if __name__=='__main__':
    #加载模型,训练好的参数,和测试的数据集
    model = VGG16()
    model.load_state_dict(torch.load("best_model.pth"))
    test_dataloader=test_data_process()
    #计算整体测试准确率
    test_model_process(model,test_dataloader)

    #处理具体结果返回
    device="cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    classes=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            model.eval()
            output=model(b_x)
            #找最大下标
            pre_lab=torch.argmax(output,dim=1)
            #遍历每个样本
            for i in range(len(b_y)):
                result=pre_lab[i].item()
                label=b_y[i].item()
                print("预测值：",classes[result],"  -----  真实值：",classes[label])