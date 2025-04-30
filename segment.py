import numpy as np
import cv2

import torch
from torch import nn
import torch.nn.functional as F

def binarize(img,filter_size = 21):
    """
    Binarizes an image (Gaussian blur followd by Otsu's algorithm)
    and might flip the color, such that it is predominantely black.
    img : 3d-array(int)
        numpy array holding the image (3 channel RBG)
    filter_size : int, optional
        size of the Gaussian filter
    returns : 2d-array(int)
        binarized image. Pixel Values are either 0 or 255
    """
    img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img,(21,21),0)
    _,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY )

    # if the image is predominately white, flip the color
    whitefrac = np.sum(img)/(255*img.shape[0]*img.shape[0])
    if whitefrac > .5:
        img = 255 - img
    return img


def pad2square(img,l_target,padding=0):
    """
    Zero-pad a rectangular image so that it becomes square shaped
    and then rescales it to a desired target size l_target.
    myimg : 2d-array(int)
        numpy array holding the image
    l_target : int
        desired target size in number of pixels
    padding : int, optional
        extra zero padding. It is ensured, that the final image has
        at least "padding" zero padding pixels around whatever "img" has been rescaled to.
    """
    h,w = img.shape
    lmax = np.max(img.shape)
    #rescaling factor
    fresc = (l_target - 2*padding)/lmax

    #zero pad shorter side to obtain a square image
    if h > w:
        img_pad = cv2.copyMakeBorder(img,0,0,(h-w)//2,h-w - (h-w)//2,cv2.BORDER_CONSTANT)
    else:
        img_pad = cv2.copyMakeBorder(img,(w-h)//2,w-h - (w-h)//2,0,0,cv2.BORDER_CONSTANT)

    # add padding to all sides
    if padding > 0:
        p_resc = int(np.round(lmax*padding/(l_target-2*padding)))
        img_pad = cv2.copyMakeBorder(img_pad,p_resc,p_resc,p_resc,p_resc,cv2.BORDER_CONSTANT)

    #rescale to target size
    img_pad = cv2.resize(img_pad,(l_target,l_target))
    return img_pad




def segment_characters(myimg,minsize=512,l_target = 28,padding=2):
    """
    Pipeline for detecting single-contour characters in a binarized image:
    Returns a list of padded and rescaled single-character subimages
    as well as their corresponding bounding boxes.
    myimg : 2d-array(int)
        numpy array holding the binarized image
    minsize = int, optional
        minimum number of pixels of a detected bounding box.
        Discarded if smaller
    l_target : int
        target size of the extracted subimages
    padding : int, optional
        padding added to the extracted character images, see pad2square
    returns: list of (tuple,2d-array(int))
        (bounding bbox, character image)
    """
    #find contours
    contours,_ = cv2.findContours(myimg.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

    segments = []

    for c in contours:
        bbox = cv2.boundingRect(c)
        if bbox[2]*bbox[3] >= minsize:
            subimg = myimg[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
            subimg = pad2square(subimg,l_target=28,padding=padding)
            segments.append((bbox,subimg))

    return segments



def predict_class(mymodel,img):
    """
    Predicted class and probablility of an image for a given network.
    mymodel : torch.nn.Module
        trained model
    img : 2d-array(int)
        image
    returns : (int,float)
        predicted class (highest probability) and its probability
    """
    # reshape, normalize and convert to torch.tensor
    #x = np.reshape(img,(28*28,1))
    x = img.copy()
    x = x.astype(float)/255.0
    x = torch.tensor(x,dtype=torch.float32)

    #feed into the network
    with torch.no_grad():
        mymodel = mymodel.eval()
        probs = F.softmax(mymodel(x),dim=1).numpy().squeeze()

    c = np.argmax(probs).astype(int)

    return c,probs[c]



def detectandlabel(mymodel,img,labels,thresh = 0.0):
    """
    Detects characters (contours) in an image and classifies and lables them.
    mymodel : torch.nn.Module
        a multi-class classifier
    img : 3d-array
        an RGB image
    labels : list of str
        class labels
    thresh : float, default 0.0
        examples are ignored if the highest softmax probability from the classifier
        is less than thesh
    returns: 3d-array
        the labelled image
    """
    img = img.copy()
    for bbox,subimg in segment_characters(binarize(img)):
        c,prob = predict_class(mymodel,subimg)
        if c >= len(labels) or prob < thresh:
            continue
        label = labels[c]
        img = cv2.rectangle(img, bbox,(255,0,0), 5)
        cv2.putText(img,label,(bbox[0],bbox[1]-50),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),7)
    return img

def detectanddrawbb(img):
    img = img.copy()
    for bbox,subimg in segment_characters(binarize(img)):
        img = cv2.rectangle(img, bbox,(255,0,0), 5)
    return img
