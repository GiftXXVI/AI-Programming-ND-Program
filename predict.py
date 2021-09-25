"""Prediction Script

This script allows the user to load a checkpoint of a 
pretrained model of vgg16, densenet121 or alexnet architecture
and use it to make predictions from input images.

This script accepts parameters such as; 
    --> the file path of the input fie,
    --> the name of the model architecture,
    --> the file path of the class to name mappings file
    --> the number of class predictions to return
    --> the file path of the checkpoint,
    --> whether to use the gpu for making the prediction.
"""

#imports
import argparse
#import os
#import sys
import torch
import numpy as np
import train_def
from train_def import get_labelmap
import predict_def
from predict_def import load_model, predict, check_sanity

parser = argparse.ArgumentParser(prog='predict.py', description='Make prediction from or check the sanity of a checkpointed model.')

parser.add_argument('input', metavar='input_file', type=str, help='The location of the input image file.')
parser.add_argument('checkpoint', metavar='check_point', type=str, help='The location of the checkpoint file to be loaded.')
parser.add_argument('--arch', type=str, help='The model architecture to train.')
parser.add_argument('--category_names', type=str, help='The location for the files containing the class to name mapping.')
parser.add_argument('--topk', type=int, help='The number of class predictions to be returned.')
parser.add_argument('--gpu', action='store_true', help='To train or make inferences using the GPU.')
parser.add_argument('--check_sanity', action='store_true', help='To check sanity using validation set along with the prediction.')


args = parser.parse_args()
input_file = 'default' if not args.input else args.input
category_names = 'default' if not args.category_names else args.category_names
checkpoint = 'checkpoint-densenet121.pth' if not args.checkpoint else args.checkpoint
model_name = 'densenet121' if not args.arch else args.arch
topk = 5 if not args.topk else args.topk
device = torch.device("cuda:0" if (args.gpu and torch.cuda.is_available()) else "cpu")

model, data_dir = load_model(model_name, checkpoint, device)
if not args.check_sanity:
    if(category_names == 'default'):
        cat_to_name = get_labelmap()
    else:
        cat_to_name = get_labelmap(category_names)
    label = cat_to_name[str(input_file.split('/')[2])]
    probs, classes = predict(model, device, input_file, topk = 5)
    n_probs = np.array(probs)
    cats = [cat_to_name[str(cl)] for cl in classes]
    print(f'Label: {label}') 
    print(f'Top {topk} Predictions:')
    for cat, prob in zip(cats, n_probs):
        print(f'{cat}: {prob * 100}')
else:
    check_sanity(data_dir, model, device, topk)
