import os
import copy
import time
import warnings
import random
warnings.filterwarnings("ignore")
import pickle
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
from copy import deepcopy
import torch.nn.functional as F
from torch import nn
from options import arg_parser
from torch.utils.data import DataLoader
from models import MLP, MLP_colored
from utils import get_dataset, exp_details

class Client(object):
    def __init__(self, args, train_loader, test_loader):
        self.args = args
        self.trainloader, self.testloader = train_loader, test_loader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.BCEWithLogitsLoss()

    def update_weights(self, model):
        # set mode to train model
        model.train()

        # set optimizer for local updates - must be set each time bc we have diff parameters than we left off on
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
      
        for iter in range(self.args.local_ep):
            batch_loss = []

            batch_correct, batch_total = 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images.to(torch.float32))
                loss = self.criterion(outputs.ravel(), labels.to(torch.float32))
                loss.backward()
                optimizer.step()

                # prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                batch_correct += torch.sum(torch.eq(pred_labels, labels)).item()
                batch_total += len(labels)

                batch_loss.append(loss.item())

        return model.state_dict(), sum(batch_loss)/len(batch_loss), batch_correct/batch_total

def test_inference(model, testloaders): #, test_loaders):
    """
    Returns the test accuracy loss, and uncertainty values
    """
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.BCEWithLogitsLoss()
 
    all_loss, all_accs = [], []

    for t in testloaders:
        loss, total, correct = 0.0, 0.0, 0.0
       
        for _, (images, labels) in enumerate(t):
            images, labels = images.to(device), labels.to(device)

            # inference
            outputs = model(images.to(torch.float32))
            
            batch_loss = criterion(outputs.ravel(), labels.to(torch.float32))
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(F.softmax(outputs), 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        all_accs.append(correct/total)
        all_loss.append(loss/len(t))

    return all_loss, all_accs


def average_weights(w, amount_data):
    w_avg = deepcopy(w[0])

    weights = [a/sum(amount_data) for a in amount_data]

    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
            w_avg[key] += w[i][key]*weights[i]
    return w_avg, weights


def main(r): 
    if r == 0:
        seed = 42
    elif r == 1:
        seed = 123
    elif r == 2:
        seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = arg_parser()
    args.round = r
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loaders, test_loaders, trainloader, testloader, shape, full_adult_train, full_adult_test, train_df, test_df, train_df_rr, possible, train_pandas_torch = get_dataset(args)

    # schl_dicts = {i: [0 for _ in range(11)] for i in range(\args.num_users+1)}
    num_labels = {i: [0, 0] for i in range(args.num_users+1)}

    for j, t in enumerate(train_loaders):
        for i, (x, y) in enumerate(t):
            # print(x, y)
            # for x_ in x[:, 6]:
            #     schl_dicts[j][int(x_)] += 1
            for y_ in y:
                num_labels[j][int(y_)] += 1

    for i, (x, y) in enumerate(trainloader):
        for y_ in y:
            num_labels[args.num_users][int(y_)] += 1
    
    # print(schl_dicts)
    print(full_adult_train)
    train_y = full_adult_train['PINCP']
    train_x = full_adult_train.drop('PINCP', axis=1)
    test_y = full_adult_test['PINCP']
    test_x = full_adult_test.drop('PINCP', axis=1)

    transformer = StandardScaler().fit(train_x)
    train_x = transformer.transform(train_x)
    test_x = transformer.transform(test_x)

    logreg = LogisticRegression(random_state=r, fit_intercept=False).fit(train_x, train_y)
    preds = logreg.predict(test_x)
    acc = accuracy_score(test_y, preds)
    print(acc)
 
    # with open(f'data_files/acsincome/train_{args.round}.pkl', 'rb') as f:  
    #     # pickle.dump(train_loaders, f)
    #     train_loaders = pickle.load(f)

    # with open(f'data_files/acsincome/test_{args.round}.pkl', 'rb') as f:  
    #     # pickle.dump(test_loaders, f)
    #     test_loaders = pickle.load(f)

    # with open(f'data_files/acsincome/trainloader_{args.round}.pkl', 'rb') as f:  
    #     # pickle.dump(trainloader, f)
    #     trainloader = pickle.load(f)

    # with open(f'data_files/acsincome/testloader_{args.round}.pkl', 'rb') as f:  
    #     # pickle.dump(testloader, f)
    #     testloader = pickle.load(f)

    # Build model
    model = MLP(args, dim_in=shape[0], dim_hidden=None, dim_out=1)


    final_accs = []
    final_full_accs = []
   

    print('-----------------------------------------------------------------------')
    for i in range(args.num_users+1):
        print('\n\n')

        if i == 0:
            current_client = Client(args=args, train_loader=trainloader, test_loader=testloader)
        else:
        # get train and test loaders
            current_client = Client(args=args, train_loader=train_loaders[i-1], test_loader=test_loaders[i-1])
        ctrainloader, ctestloader = current_client.trainloader, current_client.testloader
        
        global_model = copy.deepcopy(model)
        # set the model to train and send to device
        global_model.to(device)
        global_model.train()

        # Training
        # Set optimizer and criterion
        # sgd for dirtymnist and adam for curetsr
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=args.momentum)
        criterion = nn.BCEWithLogitsLoss()
        epoch_loss_train = []
        epoch_acc_train = []
        epoch_loss_test = []
        epoch_acc_test = []
        gepoch_loss_test = []
        gepoch_acc_test = []


        for epoch in range(args.epochs):
            batch_loss = []
            batch_correct, batch_total = 0.0, 0.0
            
            for _, (images, labels) in enumerate(ctrainloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = global_model(images.to(torch.float32))
                loss = criterion(outputs.ravel(), labels.to(torch.float32))
                # print(loss)
                loss.backward()
                #for curetsr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
          

                # prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                batch_correct += torch.sum(torch.eq(pred_labels, labels)).item()
                batch_total += len(labels)

                batch_loss.append(loss.item())
           
            train_loss, train_acc= test_inference(global_model, [ctrainloader])
            # print("\nEpoch {}".format(epoch+1), "\nTrain Accuracy: {:.3f}%".format(train_acc[0]*100), "Train Loss: {:.6f}".format(train_loss[0]))
            epoch_loss_train.append(train_loss[0])
            epoch_acc_train.append(train_acc[0])


            # testing
            test_loss, test_acc= test_inference(global_model, [ctestloader])
            # print("Test Accuracy: {:.2f}%".format(100*test_acc[0]), "Test Loss: {:.6f}".format(test_loss[0]))
            epoch_loss_test.append(test_loss[0])
            epoch_acc_test.append(test_acc[0])

            # testing
            test_loss, test_acc= test_inference(global_model, [testloader])
            # print("Full Test Accuracy: {:.2f}%".format(100*test_acc[0]), "Full Test Loss: {:.6f}".format(test_loss[0]))
            gepoch_loss_test.append(test_loss[0])
            gepoch_acc_test.append(test_acc[0])

        print(f'Client {i}')
        print(f'Training Loss : {epoch_loss_train[-1]}')
        print(f'Train Accuracy: {epoch_acc_train[-1]}')
        print(f'Test Loss : { epoch_loss_test[-1]}')
        print(f'Test Accuracy: {epoch_acc_test[-1]}')
        print(f'Full Test Loss : {gepoch_loss_test[-1]}')
        print(f'Full Test Accuracy: {gepoch_acc_test[-1]}')

        final_accs.append(epoch_acc_test[-1])
        final_full_accs.append(gepoch_acc_test[-1])

    return  final_accs, final_full_accs

if __name__ == '__main__':
    all_test_accs = [[] for _ in range(5)]
    all_full_test_accs = [[] for _ in range(5)]

    for i in range(3):
        test_accs, full_test_accs = main(i)
        for j in range(5):
            all_test_accs[j].append(test_accs[j])
            all_full_test_accs[j].append(full_test_accs[j])

    avg_client_accs = [round(100*np.average(a),2) for a in all_test_accs]
    avg_client_full_test_accs = [100*round(np.average(a),2) for a in all_full_test_accs]

    std_client_accs = [round(np.std(a),3) for a in all_test_accs]
    std_client_full_test_accs = [round(np.std(a),3) for a in all_full_test_accs]

    client_std = np.std([round(np.average(a),4) for a in all_test_accs])

    print('\n\n\n-----------------------------------------------------------------------')
    print('-----------------------------------------------------------------------')
    print('Average/std over three rounds:')
    print('------------------------------')
    print('Client Test Acc:', avg_client_accs, std_client_accs)
    print('Client Full Test Acc:', avg_client_full_test_accs, std_client_full_test_accs)

    print('Client STD:', client_std)
