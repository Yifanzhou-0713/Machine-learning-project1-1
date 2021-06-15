import torch.nn as nn
import torch.nn.functional as F


class basic_block_resnet18_1(nn.Module):
	'''  第一类基础块  '''
	def __init__(self,in_channels):
		super(basic_block_resnet18_1,self).__init__()
		self.layer12 = nn.Sequential(
			nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(),
			nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(in_channels)
			)

	def forward(self,x):
		out = self.layer12(x)
		out = F.relu(out + x)
		return out


class basic_block_resnet18_2(nn.Module):
	'''第二类基础块'''
	def  __init__(self,in_channels,out_channels):
		super(basic_block_resnet18_2,self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=2),
			nn.BatchNorm2d(out_channels)
			)
		self.layer23 = nn.Sequential(
			nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
			nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(out_channels)
			) 
		
	def forward(self,x):
		out1 = self.layer1(x)
		out2 = self.layer23(x)
		out = F.relu(out1+out2)
		return out



class bottleneck_resnet50_1(nn.Module):
	def __init__(self,in_channels,channels,stride=1):
		super(bottleneck_resnet50_1,self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels,channels[0],kernel_size=1,stride=1),
			nn.BatchNorm2d(channels[0]),
			nn.ReLU(),
			nn.Conv2d(channels[0],channels[1],kernel_size=3,stride=stride,padding=1),
			nn.BatchNorm2d(channels[1]),
			nn.ReLU(),
			nn.Conv2d(channels[1],channels[2],kernel_size=1,stride=1),
			nn.BatchNorm2d(channels[2])
			)

		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels,channels[2],kernel_size=1,stride=stride),
			nn.BatchNorm2d(channels[2])
			)

	def forward(self,x):
		out1 = self.layer2(x)
		out2 = self.layer1(x)
		out = F.relu(out1+out2)
		return out

class bottleneck_resnet50_2(nn.Module):
	def __init__(self,in_channels,channels):
		super(bottleneck_resnet50_2,self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels,channels[0],kernel_size=1,stride=1),
			nn.BatchNorm2d(channels[0]),
			nn.ReLU(),
			nn.Conv2d(channels[0],channels[1],kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(channels[1]),
			nn.ReLU(),
			nn.Conv2d(channels[1],channels[2],kernel_size=1,stride=1),
			nn.BatchNorm2d(channels[2])
			)

	def forward(self,x):
		out = self.layer1(x)
		out = F.relu(out + x)
		return out


class Resnet_18(nn.Module):
	'''18 resnet'''
	def __init__(self, inchannels, numclass):
		super(Resnet_18,self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(inchannels,64,kernel_size=7,stride=2,padding=3),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
			)
		self.reslayer = nn.Sequential(
			basic_block_resnet18_1(64),
			basic_block_resnet18_1(64),
			basic_block_resnet18_2(64,128),
			basic_block_resnet18_1(128) ,
			basic_block_resnet18_2(128,256),
			basic_block_resnet18_1(256),
			basic_block_resnet18_2(256,512),
			basic_block_resnet18_1(512)
			)

		self.avgpool = nn.AvgPool2d(2,2,padding=1)
		self.connect = nn.Linear(512,numclass)

	def forward(self,x):
		out = self.layer1(x)
		out = self.reslayer(out)
		out = self.avgpool(F.relu(out))
		out = out.view(out.size(0),-1)
		out = self.connect(out)
		return out


class Resnet_50(nn.Module):
	def __init__(self, inchannels, numclass):
		super(Resnet_50,self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(inchannels,64,kernel_size=7,stride=2,padding=3),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
			)

		self.resnet1 = nn.Sequential(
			bottleneck_resnet50_1(64,[64,64,256],1),
			bottleneck_resnet50_2(256,[64,64,256]),
			bottleneck_resnet50_2(256,[64,64,256])
			)
		self.resnet2 = nn.Sequential(
			bottleneck_resnet50_1(256,[128,128,512],2),
			bottleneck_resnet50_2(512,[128,128,512]),
			bottleneck_resnet50_2(512,[128,128,512]),
			bottleneck_resnet50_2(512,[128,128,512])
			)
		self.resnet3 = nn.Sequential(
			bottleneck_resnet50_1(512,[256,256,1024],2),
			bottleneck_resnet50_2(1024,[256,256,1024]),
			bottleneck_resnet50_2(1024,[256,256,1024]),
			bottleneck_resnet50_2(1024,[256,256,1024]),
			bottleneck_resnet50_2(1024,[256,256,1024]),
			bottleneck_resnet50_2(1024,[256,256,1024])
			)
		self.resnet4 = nn.Sequential(
			bottleneck_resnet50_1(1024,[512,512,2048],2),
			bottleneck_resnet50_2(2048,[512,512,2048]),
			bottleneck_resnet50_2(2048,[512,512,2048])
			)
		self.avgpool = nn.AvgPool2d(2,2,padding=1)
		self.connect = nn.Linear(2048,numclass)

	def forward(self,x):
		out = self.layer1(x)
		out = self.resnet1(out)
		out = self.resnet2(out)
		out = self.resnet3(out)
		out = self.resnet4(out)
		out = self.avgpool(F.relu(out))
		out = out.view(out.size(0),-1)
		out = self.connect(out)
		return out



def ResNet18(in_channels, num_classes):
    return Resnet_18(in_channels, num_classes)


def ResNet50(in_channels, num_classes):
    return Resnet_50(in_channels, num_classes)