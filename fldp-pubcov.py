import math
import copy
import time
import torch
import pickle
import random
import warnings
import itertools

import numpy as np
import pandas as pd
import torch.utils.data as data
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from copy import deepcopy
from torch.autograd import grad
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from models import MLP
from options import arg_parser
from utils import get_dataset, GRR_Client, PandasDataset
warnings.filterwarnings("ignore")

E = math.e

class Client(object):
    def __init__(self, args, train_loader, test_loader):
        self.args = args
        self.trainloader, self.testloader = train_loader, test_loader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
    def update_weights(self, model):
        
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        for iter in range(self.args.local_ep):
            batch_loss = []

            batch_correct, batch_total = 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images.to(torch.float32))
                loss = self.criterion(torch.atleast_2d(outputs), labels.to(torch.int64))
                loss.backward()
                optimizer.step()

                # prediction
                _, pred_labels = torch.max(F.softmax(outputs), 1)
                pred_labels = pred_labels.view(-1)
                batch_correct += torch.sum(torch.eq(pred_labels, labels)).item()
                batch_total += len(labels)

                batch_loss.append(loss.item())

        return model.state_dict(), sum(batch_loss)/len(batch_loss), batch_correct/batch_total


def test_inference(model, testloaders): 
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
 
    all_loss, all_accs = [], []

    for t in testloaders:
        loss, total, correct = 0.0, 0.0, 0.0
       
        for _, (images, labels) in enumerate(t):
            images, labels = images.to(device), labels.to(device)

            # inference
            outputs = model(images.to(torch.float32))
            
            batch_loss =criterion(torch.atleast_2d(outputs), labels.to(torch.int64))
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(F.softmax(outputs), 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        all_accs.append(correct/total)
        all_loss.append(loss/len(t))

    return all_loss, all_accs

def average_weights(w, weights):
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
            w_avg[key] += w[i][key]*weights[i]
    return w_avg

def calc_influence(args, device, model, train_data_rr, train_data_rr_df, train_data, test_data, weights, possible, epsilon, which_client):
    # train_data_rr = [train_data_rr_x, train_data_rr_y]
    # test_data = [test_x, test_y]
    # train_data is a pandas dataframe with all features and label
    # train_data_rr_df is a pandas dataframe with all features and label

    est_hess = -explicit_hess(model, train_data_rr, nn.CrossEntropyLoss(reduction="mean"))
    # grad_train_rr = do_grad(train_data_rr, model, nn.CrossEntropyLoss(reduction='sum'))
    # influence_removal = torch.mm(est_hess.to(device), grad_train_rr[0].T) * weights[which_client] * (1/len(train_data_rr[0]))
    removal_model = deepcopy(model)

    # with torch.no_grad():
    #     for i, param in enumerate(removal_model.parameters()):
    #         param.data.add_(influence_removal.T)
    

    # influence of perturbation on parameters
    grad_randomized = pert_grad(args, device, model, possible, epsilon, train_data, train_data_rr_df)
    # grad_change_epsilon = grad_randomized[0] - grad_train_rr[0]
    influence_randomize = torch.mm(est_hess.to(device), grad_randomized[0].T) * weights[which_client] * (1/len(train_data_rr[0]))
    randomized_model = deepcopy(model)

    with torch.no_grad():
        for i, param in enumerate(randomized_model.parameters()):
            param.data.add_(influence_randomize.T)

    # influence of perturbation on loss
    grad_test = do_grad(test_data, model, nn.CrossEntropyLoss(reduction="mean"))
    ihvp = torch.mm(grad_test[0], est_hess.to(device))
    influence_loss = torch.dot(ihvp.flatten(), grad_randomized[0].flatten()) * (1/len(train_data_rr[0])) * weights[which_client]

    return removal_model, randomized_model, influence_loss.cpu().detach().numpy(), influence_randomize


def explicit_hess(model, train_data_rr, criterion):
    model.eval()
    logits = model(train_data_rr[0])
    loss = criterion(torch.atleast_2d(logits), train_data_rr[1].to(torch.int64))

    grads = grad(loss, model.parameters(), retain_graph=True, create_graph=True)
    hess_params = torch.zeros(len(model.fc1.weight[0]), len(model.fc1.weight[0]))
    
    for i in range(len(model.fc1.weight[0])):
        hess_params_ = grad(grads[0][0][i], model.parameters(), retain_graph=True)[0][0]
        for j, hp in enumerate(hess_params_):
            hess_params[i,j] = hp

    inv_hess = torch.linalg.inv(hess_params)

    return inv_hess

def do_grad(data, model, criterion):
    # data = [data_x, data_y]
    model.eval()

    logits = model(data[0])
    loss = criterion(torch.atleast_2d(logits), data[1].to(torch.int64))
    grads = grad(loss, model.parameters())
    return grads

def pert_grad(args, device, model, possible, epsilon, train_df, train_df_rr):
    # train df is a pandas dataframe with all features and label

    criterion = nn.CrossEntropyLoss(reduce="mean")
    model.eval()
    sensitive_columns = ['AGEP', 'SEX', 'RAC1P', 'PUBCOV']

    possible_list = [possible[s] for s in sensitive_columns]
    cartesian = [list(item) for item in itertools.product(*possible_list)]

    full_agg_loss = 0
    probability_same = [float((E**epsilon)/(len(c) - 1 + (E**epsilon))) for c in possible_list]
    probility_different = [float(1/(len(c) - 1 + (E**epsilon))) for c in possible_list]
    

    for (index, row), (index_rr, row_rr) in zip(train_df.iterrows(), train_df_rr.iterrows()):
        record_loss_agg = 0

        original_randomized_data = copy.deepcopy(row_rr)
        randomized_data = copy.deepcopy(row)

        original_randomized_label = torch.FloatTensor([original_randomized_data["PUBCOV"]]).to(device).to(torch.int64)
        original_randomized_features = torch.FloatTensor(list(original_randomized_data)[:-1]).to(device)
        original_randomized_logits = model(original_randomized_features)
        original_randomized_loss = criterion(torch.atleast_2d(original_randomized_logits), original_randomized_label.to(torch.int64))
    

        original_combination = ''
        for s in range(len(sensitive_columns)):
            original_combination += str(row[sensitive_columns[s]])

        counting_p = 0
        for i, combo in enumerate(cartesian):
            str_combo = ''.join(str(c) for c in combo)
           
            different_locations = [int(a!=b) for a, b in zip(original_combination, str_combo)]
 
            prob_combo = 1
            for j, d in enumerate(different_locations):
                if d == 1:
                    prob_combo *= probility_different[j]
                else:
                    prob_combo *= probability_same[j]
            counting_p += prob_combo

            for c, col in enumerate(sensitive_columns):
                randomized_data.at[col] = cartesian[i][c]
                
            randomized_label = torch.FloatTensor([randomized_data["PUBCOV"]]).to(device).to(torch.int64)
            randomized_features = torch.FloatTensor(list(randomized_data)[:-1]).to(device)
        
            randomized_logits = model(randomized_features)
            randomized_loss = criterion(torch.atleast_2d(randomized_logits), randomized_label)
            
            record_loss_agg += (prob_combo*randomized_loss)
       
        # full_agg_loss += ((record_loss_agg/len(cartesian)) - original_randomized_loss)
        full_agg_loss += (record_loss_agg - original_randomized_loss)

       
    randomized_grad = grad(full_agg_loss, model.parameters())
    return randomized_grad


def do_influence(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(f'/home/ancarey/influence/final_data/{args.dataset}/{args.dataset}_data_{args.epsilon}_{args.round}.pkl', 'rb') as f:  
        all_data = pickle.load(f)

    train_loaders, test_loaders, trainloader, testloader, shape, full_train_rr_df, full_test_df, train_df_clients, test_df_clients, train_df_rr_clients, possible, train_pandasdataset =  deepcopy(all_data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    amount_data = []
    for t in train_loaders:
        amount_data.append(len(t.dataset))
    weights = [a/sum(amount_data) for a in amount_data]

    global_model = torch.load(f'/home/ancarey/influence/models/{args.dataset}/{args.dataset}_global_model_epsilon_{args.epsilon}_{args.round}.pt', weights_only=False)
    global_model.to(device)
    global_model.eval()
    inf_random = []

    for k in range(args.how_many):
        print(k)
        train_df_rr_y = torch.FloatTensor(train_df_rr_clients[k]['PUBCOV'].values).to(device).to(torch.int64)
        train_df_rr_x = torch.FloatTensor(train_df_rr_clients[k].drop(['PUBCOV'], axis=1).values).to(device)
        full_test_y = torch.FloatTensor(full_test_df['PUBCOV'].values).to(device).to(torch.int64)
        full_test_x = torch.FloatTensor(full_test_df.drop(['PUBCOV'], axis=1).values).to(device)
            
        removal_model, randomized_model_est, randomized_influence_loss, influence_randomize = calc_influence(args, device, deepcopy(global_model), [train_df_rr_x, train_df_rr_y], train_df_rr_clients[k], train_df_clients[k], [full_test_x, full_test_y], weights, possible, args.change_epsilon_value, k)
        inf_random.append(influence_randomize)

    randomized_model = deepcopy(global_model)

    for rand_param in inf_random:
        with torch.no_grad():
            for i, param in enumerate(randomized_model.parameters()):
                param.data.add_(rand_param.T)

    # train_df_rr_y = torch.FloatTensor(train_df_rr_clients[0]['PUBCOV'].values).to(device).to(torch.int64)
    # train_df_rr_x = torch.FloatTensor(train_df_rr_clients[0].drop(['PUBCOV'], axis=1).values).to(device)
    # full_test_y = torch.FloatTensor(full_test_df['PUBCOV'].values).to(device).to(torch.int64)
    # full_test_x = torch.FloatTensor(full_test_df.drop(['PUBCOV'], axis=1).values).to(device)
        
    # removal_model, randomized_model_est, randomized_influence_loss, influence_randomize = calc_influence(args, device, deepcopy(global_model), [train_df_rr_x, train_df_rr_y], train_df_rr_clients[0], train_df_clients[0], [full_test_x, full_test_y], weights, possible, args.change_epsilon_value, 0)
    # removal_model, randomized_model_est, randomized_influence_loss, influence_randomize = calc_influence(args, device, deepcopy(global_model), [train_df_rr_x, train_df_rr_y], train_df_rr_clients[0], train_df_clients[0], [full_test_x, full_test_y], weights, possible, args.change_epsilon_value, 0)

    test_loss, test_acc = test_inference(randomized_model, [testloader])

    return randomized_model.state_dict(), test_loss, test_acc, randomized_influence_loss

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.save == "yes":
        all_data = get_dataset(args)
        with open(f'/home/ancarey/influence/{args.dataset}_data_{args.epsilon}_{args.round}.pkl', 'wb') as f:  
            pickle.dump(all_data, f)
    else:
        with open(f'/home/ancarey/influence/final_data/{args.dataset}/{args.dataset}_data_{args.epsilon}_{args.round}.pkl', 'rb') as f:  
            all_data = pickle.load(f)

    train_loaders, test_loaders, trainloader, testloader, shape, full_train_rr_df, full_test_df, train_df_clients, test_df_clients, train_df_rr_clients, possible, train_pandasdataset =  deepcopy(all_data)

    # If doing retraining
    if args.do_diff_rr == "yes":
        for k in range(args.how_many):
            train_df_to_randomize = copy.deepcopy(train_df_clients[k])
            p = np.exp(args.change_epsilon_value) / (np.exp(args.change_epsilon_value) + 1)
            sensitive_columns = ['AGEP', 'SEX', 'RAC1P', 'PUBCOV']

            randomized_train = []
            for col in train_df_to_randomize.columns:
                column_values = train_df_to_randomize[col].tolist()
                if col in sensitive_columns:
                    randomized_col = pd.DataFrame([int(GRR_Client(val, deepcopy(possible[col]), p)) for val in column_values], columns=[col])
                    randomized_train.append(randomized_col)
                else:
                    duplicate_col = pd.DataFrame([int(val) for val in column_values], columns=[col])
                    randomized_train.append(duplicate_col)
            
            randomized_train_df = pd.concat(randomized_train, axis=1)
            train_df_rr_clients[k] = randomized_train_df
            randomized_train_pd = PandasDataset(randomized_train_df, 'pubcov')
            randomized_trainloader = DataLoader(randomized_train_pd, batch_size=args.local_bs, shuffle=True)
            train_loaders[k] = randomized_trainloader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amount_data = []
    for t in train_loaders:
        amount_data.append(len(t.dataset))
    print(amount_data)
    # weights = [a/sum(amount_data) for a in amount_data]
    weights = [a/sum(amount_data) for a in amount_data[1:]]

    # Set the model to train and send it to device.
    global_model = MLP(args, dim_in=shape[0], dim_hidden=None, dim_out=2)
    global_model.to(device)
    global_model.train()

    global_loss, global_acc = [], []
    
    for epoch in range(args.epochs):
        local_weights = []
        for idx in range(args.num_users):
            if idx == 0:
                continue
            local_model = Client(args=args, train_loader=train_loaders[idx], test_loader=test_loaders[idx])
            w, _, _ = local_model.update_weights(model=copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))

        global_weights = average_weights(local_weights, weights)
        global_model.load_state_dict(global_weights)
        test_loss, test_acc = test_inference(global_model, [testloader])
        global_loss.append(round(test_loss[0], 6))
        global_acc.append(round(test_acc[0]*100, 4))

    # if args.save == "yes":
    #     torch.save(global_model, f'/home/ancarey/influence/{args.dataset}_global_model_epsilon_{args.epsilon}_{args.round}_multiple_{args.how_many}.pt')

    return global_model.state_dict(), global_loss, global_acc


if __name__ == '__main__':
    args = arg_parser()
    num_to_do = 6
    args.dataset = 'pubcov'
    args.lr = 0.00001
    args.epochs = 50
    args.multiple = 'yes'
    
    # for i in range(200):
    #     print(i)
    #     try:
            # works: 0, 1, 
    seeds =  [42, 1, 65, 67, 68, 75]
    # seeds = [39, 1, 42, 19, 22, 33]

    # normal training
    # args.epochs = 50
    # args.do_diff_rr = "no"
    # args.save = "yes"
    
    normal_losses, normal_accs, normal_params = [], [], []
    # for r in range(num_to_do):
    #     args.seed = seeds[r]
    #     args.round = r
    #     n_p, n_l, n_a = main(args)
    #     normal_params.append(list(n_p.values())[0].detach().cpu().numpy())
    #     normal_accs.append(n_a[-1])
    #     normal_losses.append(n_l[-1])
    

    change_epsilon_values = {10: [.1, .5, 1, 2, 3, 4, 5], 
                                5: [.1, .5, 1, 2, 3, 4],
                                4: [.1, .5, 1, 2, 3],
                                3: [.1, .5, 1, 2],
                                2: [.1, .5, 1],
                                1: [.1, .5]}
    
    # change_epsilon_values = {10: [1], 
    #                         5: [1],
    #                         4: [1],
    #                         3: [1],
    #                         2: [1],
    #                         1: [1]}
    
    # influence 
    args.epochs = 1
    pert_losses, pert_accs, pert_params, pert_influence = {i:[] for i in change_epsilon_values[args.epsilon]}, {i:[] for i in change_epsilon_values[args.epsilon]}, {i:[] for i in change_epsilon_values[args.epsilon]}, {i:[] for i in change_epsilon_values[args.epsilon]}

    for r in range(num_to_do):
        args.round = r
        args.seed = seeds[r]
        print(f'Influence {r+1}')
        for ce in change_epsilon_values[args.epsilon]:
            print(f'Epsilon {ce}')
            args.change_epsilon_value = ce
            randomized_model, test_loss, test_acc, randomized_influence_loss = do_influence(args)
            pert_params[ce].append(list(randomized_model.values())[0].detach().cpu().numpy())
            pert_losses[ce].append(round(test_loss[0], 6))
            pert_accs[ce].append(round(test_acc[0]*100, 4))
            pert_influence[ce].append(randomized_influence_loss)
            print(randomized_influence_loss)

        # print('Works', i)

    # except torch._C._LinAlgError:
    #     print('Doesnt work', i)

    # changed training
    args.epochs = 50
    args.do_diff_rr = "yes"
    args.save = "no"
    
    change_losses, change_accs, change_params = {i:[] for i in change_epsilon_values[args.epsilon]}, {i:[] for i in change_epsilon_values[args.epsilon]}, {i:[] for i in change_epsilon_values[args.epsilon]}
    for r in range(num_to_do):
        args.round = r
        args.seed = seeds[r]
        print(f'Change {r+1}')
        for ce in change_epsilon_values[args.epsilon]:
            print(f'To epsilon {ce}')
            args.change_epsilon_value = ce
            c_p, c_l, c_a = main(args)
            change_params[ce].append(list(c_p.values())[0].detach().cpu().numpy())
            change_accs[ce].append(c_a[-1])
            change_losses[ce].append(c_l[-1])
            print(c_l[-1], c_a[-1])


    to_save_dict = {'normal_losses': normal_losses,
                    'normal_accs': normal_accs,
                    'normal_params': normal_params, 
                    'change_losses': change_losses,
                    'change_accs': change_accs, 
                    'change_params': change_params, 
                    'pert_losses': pert_losses, 
                    'pert_accs': pert_accs, 
                    'pert_params': pert_params, 
                    'pert_inf': pert_influence,
                }
    with open(f'/home/ancarey/influence/{args.dataset}_data_{args.epsilon}_results_redo_multiple_{args.how_many}.pkl', 'wb') as f:  
        pickle.dump(to_save_dict, f)