import os
import copy
import time
import warnings
import random
warnings.filterwarnings("ignore")
import pickle
import numpy as np
from tqdm import tqdm

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
        self.criterion = nn.CrossEntropyLoss().to(self.device)

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
                loss = self.criterion(outputs, labels.to(torch.int64))
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
    criterion = nn.CrossEntropyLoss().to(device)
 
    all_loss, all_accs = [], []

    for t in testloaders:
        loss, total, correct = 0.0, 0.0, 0.0
       
        for _, (images, labels) in enumerate(t):
            images, labels = images.to(device), labels.to(device)

            # inference
            outputs = model(images.to(torch.float32))
            
            batch_loss = criterion(outputs, labels.to(torch.int64))
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
    print('-----------------------------------------------------------------------')
    seed = 42
    args = arg_parser()
    args.round = r
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    
    train_loaders, test_loaders, trainloader, testloader, shape, full_adult_train, full_adult_test, train_df, test_df = get_dataset(args)

    # schl_dicts = {i: [0 for _ in range(11)] for i in range(args.num_users)}
    # num_labels = {i: [0, 0] for i in range(args.num_users)}
    # for j, t in enumerate(train_loaders):
    #     for i, (x, y) in enumerate(t):
    #         for x_ in x[:, 6]:
    #             schl_dicts[j][int(x_)] += 1
    #         for y_ in y:
    #             num_labels[j][int(y_)] += 1
         
    # # print(schl_dicts)
    # # print(num_labels)
  
    # print(shape)
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

    # print(a)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    amount_data = []

    for t in train_loaders:
        amount_data.append(len(t.dataset))

    print(amount_data)

    # for acsincome num_features = 10
    global_model = MLP(args, dim_in=shape[0], dim_hidden=None, dim_out=2)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()

   
    global_loss, global_acc = [], []
    after_agg_loss, after_agg_acc = {i: [] for i in range(args.num_users)}, {i: [] for i in range(args.num_users)}
    round_weights = {i: [] for i in range(args.num_users)}

    for epoch in range(args.epochs):
        local_weights, local_losses, local_accuracy = [], [], []

        global_model.train()
        
        for idx in range(args.num_users):
            local_model = Client(args=args, train_loader=train_loaders[idx], test_loader=test_loaders[idx])
            w, loss, accuracy = local_model.update_weights(model=copy.deepcopy(global_model))
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(round(loss,3))
            local_accuracy.append(round(accuracy,3))
     
        # update global weights
        global_weights, client_weights = average_weights(local_weights, amount_data)
        global_model.load_state_dict(global_weights)

        test_loss_g, test_acc_g = test_inference(global_model, [testloader])
        after_agg_test_loss, after_agg_test_acc = test_inference(global_model, test_loaders)

        global_loss.append(round(test_loss_g[0],3))
        global_acc.append(round(test_acc_g[0],3))

        for i in range(args.num_users):
            after_agg_acc[i].append(round(after_agg_test_acc[i],3))
            after_agg_loss[i].append(round(after_agg_test_loss[i],3))

        print(f' \n Results after {epoch+1} global rounds of training:')
        print(f'After Agg Loss: {[round(after_agg_test_loss[i],3) for i in range(args.num_users)]}')
        print(f'After Agg Accuracy: {[round(after_agg_test_acc[i],3) for i in range(args.num_users)]}')
        print(f'Global Test Loss: {test_loss_g[0]:.4f}')
        print(f'Global Test Accuracy: {100.*test_acc_g[0]:.3f}')

    
    final_client_test_accuracies = []
    final_global_model_accuracy = []
    
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    print('--------------------------------------------------------------------------')
    for i in range(args.num_users):
        print(f'Client {i+1}')
        print(f'After Agg Loss: {after_agg_loss[i]}')
        print(f'After Agg Accuracy: {after_agg_acc[i]}')
        final_client_test_accuracies.append(after_agg_acc[i][-1])
        print(f'Weight: {round_weights[i]}')

    print(f'Global Test Loss: {global_loss}')
    print(f'Global Test Accuracy: {global_acc}')
    final_global_model_accuracy.append(global_acc[-1])

    return final_client_test_accuracies, final_global_model_accuracy


if __name__ == '__main__':
    # main(0)
    all_test_accs = [[] for _ in range(5)]
    final_global_accs = []

    for i in range(3):
        test_accs, global_acc = main(i)
        for j in range(5):
            all_test_accs[j].append(test_accs[j])
            final_global_accs.append(global_acc)

    avg_client_accs = [round(100*np.average(a),4) for a in all_test_accs]
    avg_client_full_test_accs = 100*round(np.average(final_global_accs),4)

    std_client_accs = [round(np.std(a),4) for a in all_test_accs]
    std_client_full_test_accs = round(np.std(final_global_accs),4)

    client_std = np.std([round(np.average(a),4) for a in all_test_accs])

    print('\n\n\n-----------------------------------------------------------------------')
    print('-----------------------------------------------------------------------')
    print('Average/std over three rounds:')
    print('------------------------------')
    print('Client Test Acc:', avg_client_accs, std_client_accs)
    print('Global Acc:', avg_client_full_test_accs, std_client_full_test_accs)
    print('Client Acc STD:', round(client_std, 4))