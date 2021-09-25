import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from collections import OrderedDict
import json

def get_directories(data_dir = 'flowers'):
    """Get directory paths for training, testing and validation data.
    
    Parameters
    ----------
    data_dir : str
        A file path to the root directory.

    Returns
    -------
    train_dir
        A path to the training data directory.
    test_dir
        A path to the test data directory.
    valid_dir
        A path to the validation data directory.
    """
    #define directories    
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'
    return train_dir, test_dir, valid_dir

def transform_data(data_dir = 'flowers', print_diagnostics = True):
    """Transforms data inside the root directory to prepare it for training, testing and validation.
    
    Parameters
    ----------
    data_dir : str
        A file path to the root directory.
    print_diagnotics : bool
        A boolean that determines whether diagnostic information should be printed or not.

    Returns
    -------
    train_loader
        A dataloader for training data.
    test_loader
        A dataloader for test data.
    valid_loader
        A dataloader for validation data.
    train_dataset
        A dataset object for training data, to be used to get the class_to_idx mapping if needed later.
    """
    #define constants
    t_means = [0.485, 0.456, 0.406]
    t_stds = [0.229, 0.224, 0.225]
    batch_size = 63
    shuffle = True
    
    #define transforms
    train_transform = transforms.Compose([transforms.Resize(256), transforms.RandomRotation(30), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(t_means, t_stds)])
    
    eval_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(t_means, t_stds)])
    #get directories
    train_dir, test_dir, valid_dir = get_directories(data_dir)
    
    #define datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=eval_transform)
    
    #load data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    
    if print_diagnostics:
        #print diagnostic information
        print('Training Image Batches', len(train_loader))
        print('Testing Image Batches', len(test_loader))
        print('Validation Image Batches', len(valid_loader))
    
    return train_loader, test_loader, valid_loader, train_dataset

def get_labelmap(input_file = 'cat_to_name.json'):
    """Get the category to name mapping from the provided input file.
    
    Parameters
    ----------
    input_file : str
        A file path to the mapping file.

    Returns
    -------
    cat_to_name
        A dictionary mapping categories to names.
    """
    with open(input_file, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name


def build_classifier(device, model_name='vgg16', hidden_units = 0, learning_rate = 0.003):
    """Create a pretrained model, replace the classifier to adapt it for the required number of classes.
    
    Parameters
    ----------
    device : str
        The name of the device to be used to initialize the model and optimizer and later train the model.
    model_name : str
        The name of the model architecture to load.
    hidden_units : int
        The number of hidden units or the number used to calculate hidden units inside the classifier.
    learning_rate : float
        The learning rate to be used to initialize the optimizer before training.

    Returns
    -------
    model
        The model to be used for training.    
    criterion
        The loss function to be used for training.        
    optimizer
        The optimizer to adjust the weights during the gradient descent phase of training.        
    scheduler
        The scheduler to aadjust the learning rate during the training epochs.    
    hidden_units
        The hidden units used when building the model, to be saved to a checkpoint.    
    learning_rate
        The learning rate used when building the model, to be saved to a checkpoint.
    """
    #dropout_prob = 0.5
    cats = get_labelmap()
    
    #load selected model
    if model_name == 'densenet121':
        #load the model
        model = models.densenet121(pretrained=True)
        #define the new classifier
        if hidden_units == 0:
            hidden_units = 1000
        in_features = model.classifier.in_features
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features, hidden_units)),
                                                ('batchnorm1', nn.BatchNorm1d(hidden_units, affine=False)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(hidden_units, len(cats))),
                                                ('output', nn.LogSoftmax(dim=1))
                                               ]))
        model.classifier = classifier
    elif model_name == 'alexnet':
        #load the model
        model = models.alexnet(pretrained=True)
        if hidden_units == 0:
            hidden_units = 4096
        in_features = model.classifier[1].in_features
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features, hidden_units)),
                                                ('batchnorm1', nn.BatchNorm1d(hidden_units, affine=False)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(hidden_units, len(cats))),
                                                ('output', nn.LogSoftmax(dim=1))
                                                ]))
        model.classifier = classifier
    elif model_name == 'vgg16':
        #load the model
        model = models.vgg16(pretrained=True)
        if hidden_units == 0:
            hidden_units = 4096
        in_features = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features, hidden_units)),
                                                ('batchnorm1', nn.BatchNorm1d(hidden_units, affine=False)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(hidden_units, int(hidden_units/8))),
                                                ('batchnorm2', nn.BatchNorm1d(int(hidden_units/8), affine=False)),
                                                ('relu2', nn.ReLU()),
                                                ('fc3', nn.Linear(int(hidden_units/8), len(cats))),
                                                ('output', nn.LogSoftmax(dim=1))
                                               ]))
        model.classifier = classifier
    
    #freeze parameters
    for param in model.features.parameters():
        param.requires_grad = False
    
    criterion = nn.NLLLoss()
    model.to(device)
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1,last_epoch=-1)
    
    return model, criterion, optimizer, scheduler, hidden_units, learning_rate

def train_model(model, criterion, optimizer, scheduler, device, train_loader, valid_loader, epochs = 3, batch_aggregate = 13):
    """Get directory paths for training, testing and validation data.
    
    Parameters
    ----------
    model : model
        The model to be used for training.
    criterion : loss_function
        The loss function to be used during the training.
    optimizer : optimizer
        The optimizer to be used to update weights during the training.
    scheduler : scheduler
        The scheduler to be used to adjust the learning rate during the training.
    device : str
        The device to be used during the training.
    train_loader : dataloader
        A dataloader for the data to be used to train the model.
    valid_loader : dataloader
        A dataloader for the data to be used to validate the model during training.
    epochs : int
        The number of training loops, with the entire training data being used for learning in each loop.
    batch_aggregate : int
        The number of batches of training data over which to aggregate accuracy and loss during training.

    Returns
    -------
    """
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        train_loss = 0.0
        batch_no = 0
        scheduler.step()
        for train_inputs, train_labels in train_loader:
            #clear previous loop gradients
            optimizer.zero_grad()
            
            #switch to training mode
            model.train()
            
            #increment batch no.
            batch_no += 1
            
            #move batch to current device
            train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
            
            #feedforward and evaluate error
            train_outputs = model(train_inputs)
            train_error = criterion(train_outputs, train_labels)
            train_loss += train_error.item()
            
            #backpropagate
            train_error.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            #evaluate error using entire test dataset every 'batch_aggregate' loops
            if (batch_no % batch_aggregate) == 0:
                valid_loss = 0.0
                valid_accuracy = 0.0
                with torch.no_grad():
                    model.eval()
                    for valid_inputs, valid_labels in valid_loader:
                        valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)
                        
                        #feedforward and evaluate test error
                        valid_outputs = model(valid_inputs)
                        valid_error = criterion(valid_outputs, valid_labels)
                        valid_loss += valid_error.item()
                        
                        #evaluate test accuracy
                        valid_ps = torch.exp(valid_outputs)
                        top_prob, top_class = valid_ps.topk(1, dim=1)
                        equal = top_class == valid_labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
                print('Epoch #{}, Batch #{}, Training-Loss: {}, Validation-Loss: {}, Validation-Accuracy: {}'.format(str(epoch + 1), str(batch_no), str(round(train_loss/batch_aggregate, 4)), 
                                                                                                                     str(round(valid_loss/len(valid_loader), 4)), 
                                                                                                      str(round(valid_accuracy/len(valid_loader), 4))))
                train_loss = 0.0
                model.train()
                
def test_model(model, criterion, device, test_loader):
    """Get the category to name mapping from the provided input file.
    
    Parameters
    ----------
    model : model
        The model to be used for validation.
    criterion : loss_function
        The loss function to be used during the validation.
    device : str
        The device to be used during the validation.
    test_loader : dataloader
        The dataloader for test data.

    Returns
    -------
    """
    loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        model.eval()        
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            error = criterion(outputs, labels)
            loss += error.item()
            
            ps = torch.exp(outputs)
            top_prob, top_class = ps.topk(1, dim=1)
            equal = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
    print('Test-Loss: {}, Test-Accuracy: {}'.format(str(round(loss/len(test_loader), 4)), str(round(accuracy/len(test_loader), 4))))
    model.train()

def save_model(model, save_dir, train_dataset, model_name, learning_rate, hidden_units, data_dir):
    """Get the category to name mapping from the provided input file.
    
    Parameters
    ----------
    model : model
        The model to be used for validation.
    save_dir : str
        The folder path where the checkpoint will be saved.
    train_dataset : dataset
        The dataset object containing the class to index mapping used for training.
    model_name : str
        The model architecture of the checkpoint.
    learning_rate : float
        The learning rate used during training.
    hidden_units : int
        The value for hidden units used during the training.
    data_dir : str
        The file path to the root directory containing the data for training, testing and validation.
    
    Returns
    -------
    """
    model.class_to_idx = train_dataset.class_to_idx
    torch.save({'learning_rate': learning_rate,
                'hidden_units': hidden_units,
                'data_dir': data_dir,
                'class_to_idx': model.class_to_idx, 
                'model_state_dict': model.state_dict()}, 
               f'{save_dir}/checkpoint-{model_name}.pth')