import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from collections import OrderedDict
import json
import train_def
from train_def import transform_data, build_classifier, test_model
import PIL
from PIL import Image
import numpy as np
import os


def load_model(model_name,checkpoint, device):
    """Builds a model and restores a checkpoint to that model

    Parameters
    ----------
    model_name : str
        The name of the model architecture
    checkpoint : str
        The file path of the checkpoint
    device : str
        The choice of device to be used to build the classifier. 

    Returns
    -------
    model
        A model of 'model_name' architecture with weights and class to index mapping from the 'checkpoint'
    data_dir
        The directory containing training, testing and validation images used to build the classifier.
    """
    cp = torch.load(checkpoint)
    hidden_units = int(cp['hidden_units'])
    learning_rate = float(cp['learning_rate'])
    data_dir = cp['data_dir']
    train_loader, test_loader, valid_loader, train_dataset = transform_data(data_dir, print_diagnostics=False)
    model, criterion, optimizer, scheduler, hidden_units, learning_rate = build_classifier(device, model_name, hidden_units, learning_rate)    
    model.load_state_dict(cp['model_state_dict'])
    model.class_to_idx = cp['class_to_idx']
    test_model(model, criterion, device, test_loader)
    return model, data_dir

def process_image_alt(image):
    """Transforms an input image to a tensor before prediction (used for testing.)
    
    Parameters
    ----------
    image : PIL image
        A PIL image object to be transformed to a tensor.

    Returns
    -------
    image_tensor
        An image transformed and converted to a tensor
    """
    t_means = [0.485, 0.456, 0.406]
    t_stds = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(t_means, t_stds)
    ])
    image_tensor = transform(image)
    return image_tensor

def process_image(image):
    """Transforms an input image to a tensor before prediction
    
    Parameters
    ----------
    image : PIL image
        A PIL image object to be transformed to a tensor.

    Returns
    -------
    image_tensor
        An image transformed and converted to a tensor
    """
    #set variables
    resize_to = 256
    crop_to = 224
    
    width, height = image.size
    ratio = 0
    
    if(width > height):
        ratio = height/width
        size = int(resize_to + (resize_to * ratio)), resize_to
    elif(width < height):
        ratio = width/height
        size = resize_to, int(resize_to + (resize_to * ratio))
    else:
        size = resize_to, resize_to
    
    diff = resize_to - crop_to
    margin = diff/2
    box = (margin, margin, resize_to - margin, resize_to - margin)
    
    t_means = np.array([0.485, 0.456, 0.406])
    t_stds = np.array([0.229, 0.224, 0.225])    

    #print(size)
    #print(width, height)
    #resize and crop
    image.thumbnail(size)
    image = image.crop(box)
    
    #convert color channel to interval [0,1]
    np_image = np.array(image)/255
    #print(np_image.shape)
    #normalize image
    np_image = (np_image - t_means)/t_stds
    np_image = np_image.transpose((2, 0, 1))
    
    #convert to tensor and return image
    py_image_tensor = torch.tensor(np_image)
    return py_image_tensor

def predict(model, device, image_path, topk = 5):
    """Makes a prediction from an input image and returns the top 'topk' classes
    
    Parameters
    ----------
    model: model
        A pretrained model to be used for making the prediction.
    device: str
        The name of the device to be used for making the prediction.
    image_path: str
        The file path of the image.
    topk: int
        The number of classes to be returned.

    Returns
    -------
    probs
        A list of 'topk' probabilities corresponding to each class predicted
    classes
        A list of 'topk' classes predicted
    """
    with Image.open(image_path) as image:
        image_tensor = process_image(image)
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.type(torch.FloatTensor)        
        image_tensor = image_tensor.to(device)
        model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
        
        with torch.no_grad():
            model.eval()
            output = torch.exp(model(image_tensor))
            probs, indices = output.topk(topk)
            probs = probs.squeeze()
            indices = indices.squeeze()
            classes = [model.idx_to_class[idx] for idx in indices.tolist()]
        return probs, classes
        
def check_sanity(data_dir, model, device, topk):
    """Makes predictions from input images inside 'data_dir' to establish sanity
    
    Parameters
    ----------
    data_dir: string
        A directory containing the 'validation images' folder.
    model: model
        A model to be used to make the predictions.
    device: str
        The name of the device to be used to make the predictions.
    topk: int
        The number of classes to be returned for each prediction.

    Returns
    -------
    
    """
    valid_dir = data_dir + '/valid'
    correct = 0
    for i in model.class_to_idx:
        path = f'{valid_dir}/{str(i)}'
        #print(path)
        files = np.random.choice(os.listdir(path), size = 1)
        for file in files:
            image_path = f'{path}/{file}'
            #print(image_path)
            top_probs, top_classes = predict(model, device, image_path, topk)
            if str(i) in top_classes[0]:
                correct += 1
            print(image_path,top_classes, str(i))
    print(str(correct/len(model.class_to_idx)))