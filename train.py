"""Model Training Script

This script allows the user to create, train and save a checkpoint of a 
pretrained model of vgg16, densenet121 or alexnet architecture.

This script accepts parameters such as; 
    --> the name of the directory containing training images,
    --> the name of the model architecture,
    --> model training parameters such as; 
        --> number of hidden units, 
        --> learning rate, 
        --> number of epochs,
        --> number of batches for aggregation.
    --> the name of the directory for saving checkpoints,
    --> whether to use the gpu for training the model.
"""

#Imports
import argparse
#import os
#import sys
import torch
import train_def
from train_def import transform_data, get_labelmap, build_classifier, train_model, test_model, save_model

parser = argparse.ArgumentParser(prog='train.py', description='Create and train a deep learning model.')
parser.add_argument('Dir', metavar='data_dir', type=str, help='The directory containing the images for training, testing and validation.')
parser.add_argument('--arch', type=str, help='The model architecture to train.')
parser.add_argument('--hidden_units', type=int, help='The number of hidden units for the model.')
parser.add_argument('--learning_rate', type=float, help='The learning rate to be used when training the model.')
parser.add_argument('--epochs', type=int, help='The number of epochs for training the model.')
parser.add_argument('--batch_aggregate', type=int, help='The number of batches over which to aggregate accuracy and loss during training.')
parser.add_argument('--save_dir', type=str, help='The location for saving checkpoints.')
parser.add_argument('--gpu', action='store_true', help='To train or make inferences using the GPU.')

args = parser.parse_args()
data_dir = 'flowers' if not args.Dir else args.Dir
model_name = 'densenet121' if not args.arch else args.arch
hidden_units = 0 if not args.hidden_units else args.hidden_units
learning_rate = 0.003 if not args.learning_rate else args.learning_rate
epochs = 3 if not args.epochs else args.epochs
batch_aggregate = 13 if not args.batch_aggregate else args.batch_aggregate
save_dir = 'checkpoints' if not args.save_dir else args.save_dir
device = torch.device("cuda:0" if (args.gpu and torch.cuda.is_available()) else "cpu")

if(args.Dir):
    train_loader, test_loader, valid_loader, train_dataset = transform_data(data_dir)
    model, criterion, optimizer, scheduler, hidden_units, learning_rate = build_classifier(device, model_name, hidden_units, learning_rate)
    cat_to_name = get_labelmap()
    train_model(model, criterion, optimizer, scheduler, device, train_loader, valid_loader, epochs, batch_aggregate)
    test_model(model, criterion, device, test_loader)
    save_model(model, save_dir, train_dataset, model_name, hidden_units, learning_rate, data_dir)


        
        
            




    
    
