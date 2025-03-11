from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, args, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_out, bias=False)
        self.criterion = nn.BCEWithLogitsLoss()
        # self.fc1 = nn.Linear(dim_in, dim_out)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout()
        # self.layer_hidden = nn.utils.parametrizations.weight_norm(nn.Linear(dim_hidden, dim_out))
        # self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.view(-1, x.shape[1]*x.shape[0])
        x = self.fc1(x)
       #x = self.dropout(x)
        # x = self.relu(x)
        # x = self.layer_hidden(x)
        return x
    
class MLP_colored(nn.Module):
    def __init__(self, args, dim_in, dim_hidden, dim_out):
        super(MLP_colored, self).__init__()
        self.layer_input = nn.Linear(dim_in*3, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.dropout)
        self.layer_hidden = nn.utils.parametrizations.weight_norm(nn.Linear(dim_hidden, dim_out))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28 * 3)
        x = self.layer_input(x)
       #x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

