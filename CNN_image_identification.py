# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:49:03 2019

@author: Renchang Lu

This code including CNN, SVM and Random Forest.

"""

import matplotlib.pyplot as plt
import numpy as np
## Import Pytorch
import torch
from torch import nn
from torchvision import datasets, transforms
## Import sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV



###################################################################################################################################
############################################################ CNN ##################################################################
###################################################################################################################################

class View_(nn.Module):

    def __init__(self, *args):
        super(View_, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view( *self.shape)


class LeNet(nn.Module):
    
    ## Difine neutral network parameters
    def __init__(self):
      super().__init__()
      
      self.features = nn.Sequential(
                                      nn.Conv2d(3, 64, 3, 1),
                                      nn.ReLU(),
                                      nn.AdaptiveMaxPool2d(128),
                                      nn.Conv2d(64, 128, 3, 1), 
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, 3, 1),	
                                      nn.ReLU(), 
                                      nn.Conv2d(128, 128, 3, 1),	
                                      nn.ReLU(),
                                      nn.AdaptiveMaxPool2d(24),
                                      nn.Conv2d(128, 256, 3, 1),	
                                      nn.AdaptiveMaxPool2d(11),
                                      View_(-1, 256*11*11),
                                      nn.Linear(256*11*11, 512),
                                      nn.Dropout(0.5)
                                   )

      self.fc2 = nn.Linear(512, 2)
      
    ## Forward function  
    def forward(self, x):
        
      x = self.features(x)
      x = self.fc2(x)
      
      return x
      
def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image


def train(model, device, training_loader, criterion, optimizer, last_epoch):
    model.train()
    train_extract_feature = np.zeros((1,100))
    trian_label_output = np.zeros(1)
    running_loss = 0.0
    running_corrects = 0.0
    num = 0

    for inputs, labels in training_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)       
        num += 1
        print('This is training loop: ' + str(num))
        
        ## Record the features and labels at the last epoch.
        if (last_epoch == True):
            train_feature_output = model.features(inputs).detach().numpy()
            train_extract_feature = np.r_[train_extract_feature, train_feature_output]
            trian_label_output = np.r_[trian_label_output, labels.detach().numpy()]
        else:
            pass
    
    else:        
        epoch_loss = running_loss/len(training_loader.dataset)
        epoch_acc = running_corrects.float()/ len(training_loader.dataset)

        print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                epoch_loss, running_corrects.float(), len(training_loader.dataset),
                100. * epoch_acc))
        
        return train_extract_feature, trian_label_output, epoch_loss, epoch_acc






def test(model, device, test_loader, criterion, last_epoch):
    test_extract_feature = np.zeros((1,100))
    test_label_output = np.zeros(1)
    model.eval()
    test_running_loss = 0.0
    test_running_corrects = 0.0
    
    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_labels)
        
            _, test_preds = torch.max(test_outputs, 1)
            test_running_loss += test_loss.item()
            test_running_corrects += test_preds.eq(test_labels.view_as(test_preds)).sum().item()
            
            ## Extract features at the last epoch
            if (last_epoch == True):
                test_feature_output = model.features(test_inputs).detach().numpy()
                ## Combine all the features
                test_extract_feature = np.r_[test_extract_feature, test_feature_output]
                ## Combine all the labels
                test_label_output = np.r_[test_label_output, test_labels.detach().numpy()]
                
                dataiter = iter(test_loader)
                images, labels = dataiter.next()
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                _, preds = torch.max(output, 1)

                fig = plt.figure(figsize=(25, 4))

                for idx in np.arange(20):
                    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
                    plt.imshow(im_convert(images[idx]))
                    ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())), color=("green" if preds[idx]==labels[idx] else "red"))               
            else:
                pass
        
    test_epoch_loss = test_running_loss/len(test_loader.dataset)
    test_epoch_acc = test_running_corrects / len(test_loader.dataset)
            
            
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_epoch_loss, test_running_corrects, len(test_loader.dataset),
        100. * test_epoch_acc))
    
    return test_extract_feature, test_label_output, test_epoch_loss, test_epoch_acc



## Judge if we have cuda or not
use_cuda = torch.cuda.is_available()
print("if cuda:",use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
    
## Resize the graph to 250 * 250 * 3
transform = transforms.Compose([transforms.Resize((258,258)),                                   
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
## Load trainning set
training_dataset = datasets.ImageFolder(              
        root='./train',
        transform=transform)

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True)
    
## Load testing set
test_dataset = datasets.ImageFolder(
        root='./test',
        transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    
## Difine the model
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

## Set the epochs
epochs = int(input("Please enter an epoch: "))        

running_loss_history = []
running_corrects_history = []
test_running_loss_history = []
test_running_corrects_history = []

for e in range(epochs):
    print('laoding...')
    print('epoch :', (e+1))
    if ((e+1) != epochs):
        train_extract_feature, trian_label_output, epoch_loss, epoch_acc = train(model, device, training_loader, criterion, optimizer, last_epoch = False)
        test_extract_feature, test_label_output, test_epoch_loss, test_epoch_acc = test(model, device, test_loader, criterion, last_epoch = False)   
    else:
        train_extract_feature, trian_label_output, epoch_loss, epoch_acc = train(model, device, training_loader, criterion, optimizer, last_epoch = True)
        test_extract_feature, test_label_output, test_epoch_loss, test_epoch_acc = test(model, device, test_loader, criterion, last_epoch = True)
    
    running_loss_history.append(epoch_loss)
    running_corrects_history.append(epoch_acc)

    test_running_loss_history.append(test_epoch_loss)
    test_running_corrects_history.append(test_epoch_acc)


plt.figure(num=1, figsize=(16.4, 14.8))
    
plt.plot(running_loss_history, label='Training loss')
plt.plot(test_running_loss_history, label='Test loss')
plt.title("Loss of Every Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

    
plt.figure(num=2, figsize=(16.4, 14.8))
plt.plot(running_corrects_history, label='Training accuracy')
plt.plot(test_running_corrects_history, label='Test accuracy')
plt.legend()
plt.title("Accuracy of Every Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")


print("The CNN is done, let's begin SVM")
print()
###################################################################################################################################
############################################################ SVM ##################################################################
###################################################################################################################################  
x_train = train_extract_feature[1:]
y_train = trian_label_output[1:]
x_test = test_extract_feature[1:]
y_test = test_label_output[1:]

torch.save(model.state_dict(), 'trial2.pkl')
np.savetxt("x_train2.txt", x_train, fmt="%s")
np.savetxt("y_train2.txt", y_train, fmt="%s")
np.savetxt("x_test2.txt", x_test, fmt="%s")
np.savetxt("y_test2.txt", y_test, fmt="%s")




# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                   ]
## Use GridSearchCV to choose svm hyper-parameters
print("# Tuning SVM hyper-parameters")
print()

clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5)
clf.fit(x_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

## Let user to enter the best hype-parameter
Kernel_best = input("Please enter the best kernel: ")
C_best = float(input("Please enter the best C: "))
gamma_best = float(input("Please enter the best gamma: "))


# Use SVM
def SVM_Sk(x_train, x_test, y_train, y_test):   
    # Define model
    clf = svm.SVC(C=C_best, kernel=Kernel_best, gamma=gamma_best, random_state=123)
    # Training model
    clf.fit(x_train, y_train)
    # Predict
    y_pred = clf.predict(x_test)
    # Calculate accuracy
    print('SVM Accuracy = {:0.2f}%.'.format(100 * metrics.accuracy_score(y_test, y_pred)))

SVM_Sk(x_train, x_test, y_train, y_test)

print("The SVM is done, let's begin Random Forest")
print()

###################################################################################################################################
######################################################## Random Forest ############################################################
###################################################################################################################################

def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print('Random Forest Accuracy = {:0.2f}%.'.format(100 * metrics.accuracy_score(y_test, y_pred)))

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(x_train, y_train)
best_grid = grid_search.best_estimator_
evaluate(best_grid, x_test, y_test)

print()


print( "done!")






"""
def main():
    ## Judge if we have cuda or not
    use_cuda = torch.cuda.is_available()
    print("if cuda:",use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    ## Resize the graph to 250 * 250 * 3
    transform = transforms.Compose([transforms.Resize((250,250)),                                   
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    ## Load trainning set
    training_dataset = datasets.ImageFolder(        
            root='./train',
            transform=transform)

    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True)
    
    ## Load testing set
    test_dataset = datasets.ImageFolder(
            root='./test',
            transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    ## Difine the model
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        
    epochs = 2

    for e in range(epochs):
        print('laoding...')
        print('epoch :', (e+1))
        train_extract_feature, trian_label_output, running_loss_history, running_corrects_history = train(model, device, training_loader, criterion, optimizer, last_epoch = True)
        test_extract_feature, test_label_output, test_running_loss_history, test_running_corrects_history = test(model, device, test_loader, criterion, last_epoch = True)
 
    
    x_train = train_extract_feature[1:]
    y_train = trian_label_output[1:]
    x_test = test_extract_feature[1:]
    y_test = test_label_output[1:]
    
    SVM_Sk(x_train, x_test, y_train, y_test)
    
    fig1, ax1 = plt.subplots()
    
    ax1.plot(running_loss_history, label='Training loss')
    ax1.plot(test_running_loss_history, label='Test loss')
    ax1.legend()
    
    fig2, ax2 = plt.subplots()
    ax2.plot(running_corrects_history, label='Training accuracy')
    ax2.plot(test_running_corrects_history, label='Test accuracy')
    ax2.legend()

    print( "done!")

if __name__ == '__main__':

    main()
"""
  
"""  
dataiter = iter(validation_loader)
images, labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)
output = model(images)
_, preds = torch.max(output, 1)

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[idx]))
  ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())), color=("green" if preds[idx]==labels[idx] else "red"))
"""

"""
torch.save(model.state_dict(), '3rd_trial.pkl')

np.savetxt("feature_extract.txt", feature_array, fmt="%s")

for param_tensor in model1.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())



model1 = torch.load('First_trial.pkl')

"""


"""
# Split the data to validation set
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=109)

"""

"""
def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image


dataiter = iter(training_loader)
images, labels = dataiter.next()
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[idx]))
  ax.set_title([labels[idx].item()])

"""

"""
  else:
    with torch.no_grad():
      for val_inputs, val_labels in validation_loader:
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_labels)
        
        _, val_preds = torch.max(val_outputs, 1)
        val_running_loss += val_loss.item()
        val_running_corrects += torch.sum(val_preds == val_labels.data)
"""


    
"""
    val_epoch_loss = val_running_loss/len(validation_loader)
    val_epoch_acc = val_running_corrects.float()/ len(validation_loader)
    val_running_loss_history.append(val_epoch_loss)
    val_running_corrects_history.append(val_epoch_acc)

    print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))
"""


