import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset,ConcatDataset
import torchvision


from nets import *


def train_nn_mbgd(nepochs,net,train_dl,lr):
    """
    nepochs : int
        number of epochs
    net : torch.nn.Module
        the neural network
    dl : torch.utils.data.DataLoader
        training data
    lr : float
        learning rate
    returns : None
    """
    loss_fct = F.cross_entropy
    opt = optim.SGD(net.parameters(),lr=lr)
    print("learning rate = " + str(lr))
    for epoch in range(nepochs):
        print("epoch "+str(epoch+1)+"/"+str(nepochs)+" ... ",end="")
        net.train()
        for ii,(xb,yb) in enumerate(train_dl):
            loss = loss_fct(net(xb),yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            #print("batch "+str(ii+1))

        net.eval()

        with torch.no_grad():
            train_loss = sum(loss_fct(net(xb),yb) for xb,yb in train_dl)
            train_loss /= len(train_dl)
            print("training loss = "+str(train_loss.numpy()))



def accuracy(net,dl):
    """
    Calculates the accuracy of a classifier for a given dataset.
    net : torch.nn.Module
    dl : torch.utils.data.DataLoader
    returns : float
    """
    n_correct = 0
    n_total = 0
    net.eval()
    with torch.no_grad():
        for x_batch,y_batch in dl:
            pred = net(x_batch)
            pred = torch.argmax(pred,dim=1)
            n_correct += (pred == y_batch).float().sum().numpy()
            n_total += x_batch.size(0)

    return n_correct/n_total





def discard_none_targets(dataset):
    """
    Obtain the indices of dataset samples whose class labels are not None.
    dataset : torch.utils.data.DataSet
    """
    indices = []
    for (ii,sample) in enumerate(dataset):
        target = sample[1]
        if target is not None:
            indices.append(ii)

    return Subset(dataset,indices)

#The images need to be rotated by 90° in clock-wise order and flipped along the vertical axis
emnist_img_transform = torchvision.transforms.Compose(
    [
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Lambda(lambda x: x.transpose(1,2))
    ]
)

def load_mnist_digits_bg(batch_size = 128):
    """
    Construct a dataset that consists of MNIST handwritten digits and a subset of
    EMNIST "Letters" as negative examples (non digits):
    The letters "G,I,L,O,Q" (both caps and lower) are excluded since they might be confused
    with the digits "(6,9),1,1,0,(0,9)" respectively.
    batch_size : int, default 128
    returns: (DataLoader,DataLoader)
        DataLoaders for for train and test set.
    """

    def relabel_letter_class(class_idx):
        excluded_letters_idx = [6,8,11,14,16]
        if class_idx in excluded_letters_idx:
            return None
        if class_idx >= 10:
            return 10



    background_train = torchvision.datasets.EMNIST(root='./data',
                                       train=True,
                                       download=True,
                                       split = 'letters',
                                       transform = emnist_img_transform,
                                       target_transform = relabel_letter_class)



    background_test = torchvision.datasets.EMNIST(root='./data',
                                           train=False,
                                           download=True,
                                           split = 'letters',
                                           transform = emnist_img_transform,
                                           target_transform = relabel_letter_class)


    mnist_train = torchvision.datasets.EMNIST(root='./data',
                                       train=True,
                                       download=True,
                                       split = 'mnist',
                                       transform = emnist_img_transform)



    mnist_test = torchvision.datasets.EMNIST(root='./data',
                                           train=False,
                                           download=True,
                                           split = 'mnist',
                                           transform = emnist_img_transform)

    # Discard unwanted letters from the background data
    background_train = discard_none_targets(background_train)
    background_test = discard_none_targets(background_test)

    # merge background data and digits data into a new data set
    train_ds = ConcatDataset([mnist_train,background_train])
    test_ds = ConcatDataset([mnist_test,background_test])


    # create data loaders and shuffle everything...
    train_dl = torch.utils.data.DataLoader(train_ds,
                                          batch_size=batch_size,
                                          shuffle=True)

    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size=batch_size,
                                          shuffle=True)

    return train_dl,test_dl



def load_caps(batch_size = 128):
    """
    Load all uppercase letters from the EMNIST "By_Class" dataset.
    batch_size : int, default 128
    returns: (DataLoader,DataLoader)
        DataLoaders for for train and test set.
    """
    def relabel_classes(class_idx):
        if 10 <= class_idx <= 35:
            return class_idx - 10
        return None



    train_ds = torchvision.datasets.EMNIST(root='./data',
                                           train=True,
                                           download=True,
                                           split = 'byclass',
                                           transform = emnist_img_transform,
                                           target_transform = relabel_classes)



    test_ds = torchvision.datasets.EMNIST(root='./data',
                                           train=False,
                                           download=True,
                                           split = 'byclass',
                                           transform = emnist_img_transform,
                                           target_transform = relabel_classes)

    # Discard unwanted characters from the background data
    train_ds = discard_none_targets(train_ds)
    test_ds = discard_none_targets(test_ds)


    # create data loaders and shuffle everything...
    train_dl = torch.utils.data.DataLoader(train_ds,
                                          batch_size=batch_size,
                                          shuffle=True)

    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size=batch_size,
                                          shuffle=True)

    return train_dl,test_dl




def train_caps(state_dict_path = "lenet_caps_parameters.pth"):
    print("LeNet for upper case letters")
    print("loading data ... ",end="")
    caps_train_dl,caps_test_dl = load_caps()
    print("done")

    torch.manual_seed(1)
    cnn_caps = LeNet_NClasses_batchnorm(26)

    train_nn_mbgd(nepochs=20,
                 net = cnn_caps,
                 train_dl = caps_train_dl,
                 lr=0.1)

    print("evaluating...")
    print("training accuracy: " + str(accuracy(cnn_caps,caps_train_dl)))
    print("test accuracy: " + str(accuracy(cnn_caps,caps_test_dl)))

    torch.save(cnn_caps.state_dict(),state_dict_path)



def train_digits_bg(state_dict_path = "lenet_digits_bg_parameters.pth"):
    print("LeNet for MNIST digits + negative examples")
    print("loading data ... ",end="")
    digits_train_dl,digits_test_dl = load_mnist_digits_bg(batch_size = 128)
    print("done")

    torch.manual_seed(1)
    cnn_digits_bg = LeNet_NClasses_batchnorm(11)

    train_nn_mbgd(nepochs=10,
                 net = cnn_digits_bg,
                 train_dl = digits_train_dl,
                 lr=0.1)

    train_nn_mbgd(nepochs=10,
                 net = cnn_digits_bg,
                 train_dl = digits_train_dl,
                 lr=0.01)

    print("evaluating...")
    print("training accuracy: " + str(accuracy(cnn_digits_bg,digits_train_dl)))
    print("test accuracy: " + str(accuracy(cnn_digits_bg,digits_test_dl)))

    torch.save(cnn_digits_bg.state_dict(),state_dict_path)


if __name__ == "__main__":

    train_caps()
