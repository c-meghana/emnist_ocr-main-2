from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1,6,kernel_size = 5, stride = 1, padding=2)
        self.conv2 = nn.Conv2d(6,16,kernel_size = 5, stride = 1, padding=0)
        self.conv3 = nn.Conv2d(16,120,kernel_size = 5, stride = 1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self,x):
        x = F.relu(self.conv1(x.view(-1,1,28,28)))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        #x = F.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(x.size(0),-1)
        return x

    def predict_class(self,xb):
        with torch.no_grad():
            return torch.argmax(self.forward(xb),axis=1)





class LeNet_NClasses(nn.Module):
    def __init__(self,nclasses):
        super(LeNet_NClasses,self).__init__()
        self.conv1 = nn.Conv2d(1,6,kernel_size = 5, stride = 1, padding=2)
        self.conv2 = nn.Conv2d(6,16,kernel_size = 5, stride = 1, padding=0)
        self.conv3 = nn.Conv2d(16,120,kernel_size = 5, stride = 1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=nclasses)

    def forward(self,x):
        x = F.relu(self.conv1(x.view(-1,1,28,28)))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        #x = F.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(x.size(0),-1)
        return x

class LeNet_NClasses_batchnorm(nn.Module):
    def __init__(self,nclasses):
        super(LeNet_NClasses_batchnorm,self).__init__()
        self.conv1 = nn.Conv2d(1,6,kernel_size = 5, stride = 1, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6,16,kernel_size = 5, stride = 1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(in_features=84, out_features=nclasses)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x.view(-1,1,28,28))))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = x.view(x.size(0),-1)
        return x
