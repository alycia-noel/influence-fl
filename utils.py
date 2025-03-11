import random
import os
import copy
import math
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from PIL import Image
import fairlearn.datasets
import torchvision.datasets as dsets
from folktables import ACSDataSource, ACSPublicCoverage
import torchvision
import torchvision.transforms as transforms

class DatasetSplit(Dataset):
    """
    An abstract dataset class wrapped around Pytorch dataset class
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.labels = [self.dataset[int(i)][1] for i in idxs]

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class FastMNIST(MNIST):
    def __init__(self, root, train, download):
        super().__init__(root, train, download)
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(.1307).div_(0.3081)
    
    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        return img, label
        

class PandasDataset(Dataset):
    def __init__(self, dataframe, which_dataset):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        features = row[:-1]
        label = row[-1]
        return features, label

    def __len__(self):
        return len(self.dataframe)

def GRR_Client(input_data, possible, p):
    if np.random.binomial(1, p) == 1:
        return int(input_data)

    else:
        possible.remove(input_data)
        return np.random.choice(possible)

def make_adult(args):
    """
    for adult dataset: followed pre-processing https://mlr3fairness.mlr-org.com/reference/adult.html
    
    adult == acsincome
        individuals >16 who worked at least 1 hour per week and had an income of at least $100
    """
    adult_dataset = fairlearn.datasets.fetch_acs_income()['frame']

    adult_dataset = adult_dataset.drop(labels=['OCCP'], axis=1)
    adult_dataset = adult_dataset.dropna(axis=0,  how='any', ignore_index=True)
    
    adult_dataset = adult_dataset[(adult_dataset.values != '?').all(axis=1)]

    adult_dataset["POBP"] = adult_dataset["POBP"].apply(lambda x: x >= 56)

    adult_dataset["COW"] = adult_dataset["COW"].replace({1: 'Not-Gov',
                                                         2: 'Not-Gov',
                                                         3: 'Gov',
                                                         4: 'Gov',
                                                         5: 'Gov',
                                                         6: 'Not-Gov',
                                                         7: 'Not-Gov',
                                                         8: 'None',
                                                         9: 'None'})

    adult_dataset["MAR"] = adult_dataset["MAR"].replace({1: 'Married',
                                                         2: 'Not-Married',
                                                         3: 'Not-Married',
                                                         4: 'Married',
                                                         5: 'Not-Married'})

    adult_dataset["SCHL"] = adult_dataset["SCHL"].replace({1: 'None',
                                                                             2: '<GradeS',
                                                                             3: '<GradeS',
                                                                             4: '~GradeS',
                                                                             5: '~GradeS',
                                                                             6: '~GradeS',
                                                                             7: '~GradeS',
                                                                             8: 'MS',
                                                                             9: 'MS',
                                                                             10: 'JHS',
                                                                             11: 'JHS',
                                                                             12: 'nine',
                                                                             13: 'ten',
                                                                             14: 'eleven',
                                                                             15: 'tweleve',
                                                                             16: 'HS',
                                                                             17: 'GED',
                                                                             18: '~College',
                                                                             19: '~College',
                                                                             20: 'Assoc',
                                                                             21: 'BS',
                                                                             22: 'MS',
                                                                             23: 'Prof',
                                                                             24: 'PhD'})

    adult_dataset['RAC1P'] = adult_dataset["RAC1P"].replace({1: "White",
                                                             2: "BIPOC",
                                                             3: "BIPOC",
                                                             4: "BIPOC",
                                                             5: "BIPOC",
                                                             6: "AAPI",
                                                             7: "AAPI",
                                                             8: "Other",
                                                             9: "Other"})

    adult_dataset["RELP"] = adult_dataset["RELP"].replace({0: 'self',
                                                           1: 'partner',
                                                           2: 'child',
                                                           3: 'child',
                                                           4: 'stepchild',
                                                           5: 'sibling',
                                                           6: "parent",
                                                           7: 'other-child',
                                                           8: 'parent',
                                                           9: 'child',
                                                           10: 'other-family',
                                                           11: 'roommate',
                                                           12: 'roommate',
                                                           13: 'partner',
                                                           14: 'other-child',
                                                           15: 'roommate',
                                                           16: 'group',
                                                           17: 'group'})

    adult_dataset["PINCP"] = adult_dataset["PINCP"].apply(lambda x: x > 50000)
  
    adult_dataset["AGEP"] = adult_dataset["AGEP"].apply(lambda x: round(x / 10))
    adult_dataset["WKHP"] = adult_dataset["WKHP"].apply(lambda x: round(x / 10))

    for k in adult_dataset.columns:
        adult_dataset[k] = adult_dataset[k].astype('category').cat.codes

    adult_dataset = adult_dataset.drop_duplicates(ignore_index=True)
    
    possible = {}
    for k in adult_dataset.columns:
        possible[k] = pd.unique(adult_dataset[k]).tolist()

    adult_dataset = adult_dataset.groupby('PINCP').sample(n=3000)
    adult_dataset = adult_dataset.reset_index(drop=True)

    return adult_dataset, possible

def make_glioma(args):
    # https://www.kaggle.com/code/mikedelong/logistic-regression-acc-0-8869/notebook
    INFO = '/home/ancarey/influence/final_data/TCGA_InfoWithGrade.csv'
    df = pd.read_csv(filepath_or_buffer=INFO)
    # df['Grade'] = df['Grade'].map({0:'LGG', 1:'GMB'})
    df["Age_at_diagnosis"] = df["Age_at_diagnosis"].apply(lambda x: int(x >= np.mean( df["Age_at_diagnosis"])))

    df = df.drop_duplicates(ignore_index=True)

    where_to_add = len(list(df.columns))
    column_to_move = df.pop("Grade")

    # insert column with insert(location, column_name, column_value)

    df.insert(where_to_add-1, "Grade", column_to_move)

    possible = {}
    for k in df.columns:
        possible[k] = pd.unique(df[k]).tolist()

    return df, possible


def make_pubcov():
    ds = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    STATE_DATA = ds.get_data(states=["TX"], download=True)
    features, labels, _ = ACSPublicCoverage.df_to_pandas(STATE_DATA)
    
    df = pd.concat([features, labels], axis=1)
    
    df = df.drop_duplicates(keep='first', ignore_index=True)
    df = df.drop(['ANC', 'ST', 'ESP', 'FER', 'MIG', 'DEAR', 'DEYE', 'DREM'], axis=1)
        
    def numericalBinary(dataset, features):
        dataset[features] = np.where(dataset[features] >= dataset[features].mean(), 1, 0)

    def binarize(dataset, features):
        dataset[features] = np.where(df[features] == 1, 1, 0)
    
    df["PINCP"] = df["PINCP"].apply(lambda x: round(x / 1000))
  
    df["AGEP"] = df["AGEP"].apply(lambda x: round(x / 10))


    for k in df.columns:
        df[k] = df[k].astype('category').cat.codes

    df = df.drop_duplicates(ignore_index=True)
   
    where_to_add = len(list(df.columns))
    column_to_move = df.pop("PUBCOV")
    df.insert(where_to_add-1, "PUBCOV", column_to_move)
    
    possible = {}
    for k in df.columns:
        possible[k] = pd.unique(df[k]).tolist()

    df = df.groupby('PUBCOV').sample(n=3000)
    df = df.reset_index(drop=True)

    return df, possible

def get_dataset(args):
    
    if args.dataset == 'adult':
        adult_dataset, possible = make_adult(args)
        adult_dict_users = adult_noniid(args, adult_dataset['PINCP'], args.num_users)
    elif args.dataset == 'glioma':
        adult_dataset, possible = make_glioma(args)
        adult_dict_users = adult_noniid(args, adult_dataset['Grade'], args.num_users)
    elif args.dataset == 'pubcov':
        adult_dataset, possible = make_pubcov()
        adult_dict_users = adult_noniid(args, adult_dataset['PUBCOV'], args.num_users)
    
    all_test = []
    all_train = []
    train_loaders = []
    test_loaders = []
    train_df = []
    train_df_rr = []
    test_df = []

    # Create dataloaders 
    for i in range(args.num_users):
        random.shuffle(adult_dict_users[i])

        adult_idx_train = adult_dict_users[i][:int(0.8*len(adult_dict_users[i]))]
        adult_idx_test = adult_dict_users[i][int(0.8*len(adult_dict_users[i])):]

        adult_train_df = adult_dataset.iloc[adult_idx_train.tolist()]
        adult_test_df = adult_dataset.iloc[adult_idx_test.tolist()]

        # do initial RR here
        if args.do_diff_rr == 'yes':
            if i == 0:
                p = np.exp(args.change_epsilon_value) / (np.exp(args.change_epsilon_value) + 1)
            else:
                p = np.exp(args.epsilon) / (np.exp(args.epsilon) + 1)
        else:
            p = np.exp(args.epsilon) / (np.exp(args.epsilon) + 1)


        if args.dataset == 'adult':
            sensitive_columns = ['RAC1P', "PINCP", "SEX"]
        elif args.dataset == 'glioma':
            sensitive_columns = ['Age_at_diagnosis', "Race", "Grade"]
        elif args.dataset == 'pubcov':
            sensitive_columns = ['AGEP', 'SEX', 'DIS', 'PUBCOV']

        rr_adult_train = []
        for col in adult_train_df.columns:
            if col in sensitive_columns:
                col_val_list = adult_train_df[col].tolist()
                df_new_col = pd.DataFrame([int(GRR_Client(val, deepcopy(possible[col]), p)) for val in col_val_list], columns=[col])
                rr_adult_train.append(df_new_col)
            else:
                col_val_list = adult_train_df[col].tolist()
                rr_adult_train.append(pd.DataFrame([int(val) for val in col_val_list], columns=[col]))
        
        adult_train_df_rr = pd.concat(rr_adult_train, axis=1)
        train_df_rr.append(adult_train_df_rr)
        train_df.append(adult_train_df)
        test_df.append(adult_test_df)


        if args.dataset == 'adult':
            adult_train = PandasDataset(adult_train_df_rr, 'adult')
            adult_test = PandasDataset(adult_test_df, 'adult')
        elif args.dataset == 'glioma':
            adult_train = PandasDataset(adult_train_df_rr, 'glioma')
            adult_test = PandasDataset(adult_test_df, 'glioma')
        elif args.dataset == 'pubcov':
            adult_train = PandasDataset(adult_train_df_rr, 'pubcov')
            adult_test = PandasDataset(adult_test_df, 'pubcov')
       
        all_train.append(adult_train)
        all_test.append(adult_test)
        trainloader = DataLoader(adult_train, batch_size=args.local_bs, shuffle=True)
        testloader = DataLoader(adult_test, batch_size=args.local_bs, shuffle=False)
        train_loaders.append(trainloader)
        test_loaders.append(testloader)

    all_adult_test = data.ConcatDataset(all_test)
    testloader = DataLoader(all_adult_test, batch_size=args.local_bs, shuffle=False)
    all_adult_train = data.ConcatDataset(all_train)
    trainloader = DataLoader(all_adult_train, batch_size=args.local_bs, shuffle=True)

    return [train_loaders, test_loaders, trainloader, testloader, adult_train[0][0].shape, pd.concat(train_df_rr, axis=0).sample(frac=1, ignore_index=True), pd.concat(test_df, axis=0),  train_df, test_df, train_df_rr, possible, all_train]

def adult_noniid(args, data, num_users):
    """
    Sample non-IID client data from MNIST dataset
    :param dataset: torch dataset object
    :param num_users: int, how many users in the federation
    :return: dict of image index
    """
    # print(len(data.discr))
    # 4160 training images --> 208 images/shard x 20 shards
    if args.unbalanced == 'yes':
        num_shards, num_imgs = 100, math.floor(len(data)/100)
    else:
        # num_shards, num_imgs = num_users, 2400
        num_shards, num_imgs = num_users, math.floor(len(data)/num_users)
    data = np.array(data[:num_imgs*num_shards])

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i:np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)

    # sort labels
    idxs_discr = np.vstack((idxs, data))
    idxs_discr = idxs_discr[:, idxs_discr[1,:].argsort()]
    idxs = idxs_discr[0,:]

    # divide and assign a,b shards/client for both mnist and amnist
    for i in range(num_users):
        if args.unbalanced == 'yes':
            if i == 0:
                ns = int(args.percentage)
            else:
                if ((100 - int(args.percentage)) %  4) == 0:
                    ns = int((100 - int(args.percentage)) / 4)
                else:
                    a = ((100 - int(args.percentage)) - 10)/ 4
                    if i in [1, 2]:
                        ns = int(a)
                    else:
                        ns = int(a+5)
            print(ns)
        else:
            ns = 1
    
        rand_set = set(np.random.choice(idx_shard, ns, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
      
        for mrand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[mrand*num_imgs:(mrand+1)*num_imgs]), axis=0)
    return dict_users

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : SGD')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}')
    print(f'    Round     : {args.round}')
    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : 100%')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return