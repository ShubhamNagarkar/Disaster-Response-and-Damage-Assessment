import numpy as np
import torch
import argparse
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchsummary import summary
from PIL import Image
import os
from collections import Counter
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix

# set up GPU
DEVICE = ("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
PATIENCE = 5
N_WORKERS = 8
IMAGE_SIZE = (224, 224)
CLASSES = {'affected_individuals': 0, 'infrastructure_and_utility_damage': 1, 'not_humanitarian': 2, 'other_relevant_information': 3, 'rescue_volunteering_or_donation_effort': 4}

REL = os.getcwd()

PATH_TRAIN = REL + '/dataset/images/Train/'
PATH_TEST =  REL + '/dataset/images/Test/'
PATH_VAL = REL + '/dataset/images/Val/'

CHECKPOINT = REL + '/model_checkpoints/resnet50/'
VISUALIZATION = REL + '/visualization/resnet50/'

# {0: 71, 1: 612, 2: 3252, 3: 1279, 4: 912} counts



class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=6, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta


    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        saves the current best version of the model if there is decrease in validation loss
        """
        torch.save(model.state_dict(), CHECKPOINT + 'early_stopping_resnet50.pth')
        self.vall_loss_min = val_loss


class Resnet50(torch.nn.Module):
  def __init__(self):
    super(Resnet50,self).__init__()

    self.resnet = torchvision.models.resnet50(pretrained=True)
    modules = list(self.resnet.children())[:-1]
    self.resnet = torch.nn.Sequential(*modules)
  
    for params in self.resnet.parameters():
      params.requires_grad = False
    

    self.head = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=2048, out_features=512, bias=True),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, 5, bias=True))
    

  def forward(self, x):
    feat = self.resnet(x)
    output = self.head(feat)
    return feat, output


def transforms_train():
    """
    Returns transformations on training dataset.
    params: mean - channel-wise mean of data
            std - channel-wise std of data
    """
    transfrms = []
    p = np.random.uniform(0, 1)

    transfrms.append(torchvision.transforms.Resize(IMAGE_SIZE, interpolation=Image.BILINEAR))
    transfrms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
    if p >= 0.4 and p <=0.6:
        transfrms.append(torchvision.transforms.ColorJitter(0.2,0.1,0.2))
  
    transfrms.append(torchvision.transforms.ToTensor())  
    # transfrms.append(torchvision.transforms.Normalize(MEAN, STD))
    
    return torchvision.transforms.Compose(transfrms)
    

def transforms_test():
    """
    Returns transformations on testing dataset.
    params: mean - channel-wise mean of data
            std - channel-wise std of data
    """
    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(IMAGE_SIZE, interpolation=Image.BILINEAR),
                                                 torchvision.transforms.ToTensor()])
                                                 #torchvision.transforms.Normalize(MEAN, STD)])
    return test_transform


def find_mean_std(train_set):
    """
    returns mean and std of the entire dataset
    :params: train_set - train data loader
             test_set - test data loader
    """
    mean = 0.
    var = 0.
    nb_samples = 0.
    
    for data,label in train_set:
        batch_samples = data.size(0) # Batch size
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        var += data.var(2).sum(0)
        nb_samples += batch_samples
   
    mean /= nb_samples
    var /= nb_samples
    std = torch.sqrt(var)
    
    return mean, std


def get_dataloaders(train_transform=None, test_transform=None):
    """
    returns train, validation and test dataloafer objects
    params: train_transform - Augmentation for trainset
            test_transform - Augmentation for testset
            batch_size - size of batch
            n_workers - number of workers
    """
    training = torchvision.datasets.ImageFolder(root=PATH_TRAIN, transform=train_transform)
    validation = torchvision.datasets.ImageFolder(root=PATH_VAL, transform=test_transform)
    testing = torchvision.datasets.ImageFolder(root=PATH_TEST, transform=test_transform)

    train_set = torch.utils.data.DataLoader(training, batch_size = BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
    val_set = torch.utils.data.DataLoader(validation, batch_size = BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
    test_set = torch.utils.data.DataLoader(testing, batch_size = BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

    return train_set, val_set, test_set



def train_loop(model, t_dataset, v_dataset, criterion, optimizer):
    """
    returns loss and accuracy of the model for 1 epoch.
    params: model -  resnet50
          dataset - train or val dataset
          flag - "train" for training, "val" for validation
          criterion - loss function
          optimizer - Adam optimizer
    """
    total = 0
    correct = 0
    epoch_t_loss = 0
    epoch_v_loss = 0
    model.train()
    
    for ind, (image, label) in enumerate(t_dataset):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        _, output = model(image)
        # print(output)
        loss = criterion(output, label)
     
        epoch_t_loss += loss.item()
        predicted = torch.argmax(output, dim=1)
        total += label.size(0)
        correct += (predicted==label).sum().item()
        
        loss.backward()
        optimizer.step()
 
    epoch_t_accuracy = correct/total
    epoch_t_loss = epoch_t_loss/len(t_dataset)
    
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for ind, (image, label) in enumerate(v_dataset):

            image = image.to(DEVICE)
            label = label.to(DEVICE)

            _, output = model(image)

            loss = criterion(output, label)
            epoch_v_loss += loss.item()
            predicted = torch.argmax(output, dim=1)
            total += label.size(0)
            correct += (predicted==label).sum().item()


    epoch_v_accuracy = correct/total
    epoch_v_loss = epoch_v_loss/len(v_dataset)
    
    return epoch_t_loss, epoch_t_accuracy, epoch_v_loss, epoch_v_accuracy

     

def train(train_loader, val_loader, model, optimizer, criterion):
    """
    returns train and validation losses of the model over complete training.
    params: train_loader - train dataset
          val_loader - validation dataset
          optimizer - optimizer for training
          criterion - loss function
    """
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    print("Training start...")
    
    early_stop = EarlyStopping(patience=PATIENCE)
    used_early_stopping = False

    for epoch in range(EPOCHS):

        print("Running Epoch {}".format(epoch+1))

        epoch_train_loss, train_accuracy, epoch_val_loss, val_accuracy = train_loop( model, train_loader, val_loader, criterion, optimizer)
        train_losses.append(epoch_train_loss)
        train_acc.append(train_accuracy)
        val_losses.append(epoch_val_loss)
        val_acc.append(val_accuracy)
        
        if (epoch+1)%25==0:
            ckpt_path = CHECKPOINT + 'resnet50_epoch_{}.pth'.format(epoch+1)
            torch.save(model.state_dict(), ckpt_path)

        early_stop(epoch_val_loss, model)

        if early_stop.early_stop:
            print("Early stopping")
            used_early_stopping  = True
            break

        print("Training loss: {0:.4f}  Train Accuracy: {1:0.2f}".format(epoch_train_loss, train_accuracy))
        print("Validation loss: {0:.4f}  Validation Accuracy: {1:0.2f}".format(epoch_val_loss, val_accuracy))
        print("--------------------------------------------------------")

    print("Training done...")
    print("Model saved!")

    if used_early_stopping:
      return train_losses, train_acc, val_losses, val_acc, epoch+1

    final_ckpt = CHECKPOINT + 'resnet50_final_model.pth'
    torch.save(model.state_dict(), final_ckpt)

    return train_losses, train_acc, val_losses, val_acc, EPOCHS


def test(model, test):
  """
  returns output probabilites and prediction classes
  params: model - model for testing
          test - test dataset
  """
  correct = 0
  total = 0
  predicted_prob = []
  actual_class = []
  predicted_class = []
  model.eval()
  with torch.no_grad():  
    for image, label in test:

        image = image.to(DEVICE)
        label = label.type(torch.float).to(DEVICE)
        _, output_prob = model(image)
        predicted = torch.argmax(output_prob, dim=1)
        actual_class.extend(label.cpu().tolist())
        predicted_prob.extend(output_prob.tolist())
        predicted_class.extend(predicted.tolist())
        total += label.size(0)
        correct += (predicted == label).sum().item()
    
    test_accuracy = correct/total
    print('Test Accuracy: %f %%' %(test_accuracy))

  return actual_class, predicted_class



def draw_training_curves(train_losses, test_losses, curve_name, epoch):
    plt.clf()
    max_y = 2.0
    if curve_name == "Accuracy":
        max_y = 1.0
        plt.ylim([0,max_y])
        
    plt.xlim([0,epoch])
    plt.xticks(np.arange(0, epoch, 3))
    plt.plot(train_losses, label='Training {}'.format(curve_name))
    plt.plot(test_losses, label='Testing {}'.format(curve_name))
    plt.ylabel(curve_name)
    plt.xlabel('Epoch')
    plt.legend(frameon=False)
    plt.savefig(VISUALIZATION + "/{}_curve_resnet50.png".format(curve_name))
    plt.close()


def plot_cm(lab, pred):

    target_names = ['A', 'I', 'N', 'O', 'R' ]

    cm = confusion_matrix(lab, pred)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names, cmap = "YlGnBu")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout(h_pad=5.0)
    plt.savefig(VISUALIZATION + "confusion_matrix.png")
 
    plt.close()


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Arguments for training/testing')
  parser.add_argument('--mode', type=str, default="train", help=' train or test model')
  opt = parser.parse_args()


  train_transform, test_transform = transforms_train(), transforms_test()
  train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_transform, test_transform)


  nSamples = [71, 612, 3252, 1279, 912]
  normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
  normedWeights = torch.FloatTensor(normedWeights).to(DEVICE)

  if opt.mode== "train":

    # Initialize model
    model = Resnet50().to(DEVICE)

    #Optimizer initialization
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))

    #Loss function initialization
    criterion = torch.nn.CrossEntropyLoss()

    train_loss, train_acc, val_loss, val_acc, epoch = train(train_dataloader, val_dataloader, model, optimizer, criterion)

    curve_name = 'Loss'
    draw_training_curves(train_loss, val_loss, curve_name, epoch)

    curve_name = 'Accuracy'
    draw_training_curves(train_acc, val_acc, curve_name, epoch)

    actual_class, predicted_class = test(model, test_dataloader)

    print(classification_report(actual_class, predicted_class))

    plot_cm(actual_class, predicted_class)
  else:

    # load specific model for test
    test_model = Resnet50().to(DEVICE)
    test_model.load_state_dict(torch.load(CHECKPOINT + 'model.pth', map_location ='cuda:0'))

    output_prob, output_class = test(test_model, test_dataloader)