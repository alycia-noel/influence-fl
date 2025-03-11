import argparse

def arg_parser():
    parser = argparse.ArgumentParser()

    # federated argumnets (Notation for the arguments followed from the paper)
    parser.add_argument('--round', type=int, default=0, help='Which round to run: 0, 1, 2')
    parser.add_argument('--epochs', type=int, default=100, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: k")
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=1, help="the local batch size: B")
    parser.add_argument('--lr', type=float, default=.00001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.0, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0, help="for weight decay")
    parser.add_argument('--epsilon', type=float, default=10.0, help="for randomized response")
    parser.add_argument('--change_epsilon_epoch', type=int, default=1, help="epoch to change client epsilon")
    parser.add_argument('--change_epsilon_value', type=float, default=1.0, help='value to change epsilon to')
    parser.add_argument('--tau', type=float, default=0.5, help='threshold value')
    parser.add_argument('--do_influence', type=str, default='no')
    parser.add_argument('--do_diff_rr', type=str, default='no')
    parser.add_argument('--save', type=str, default='no')
    parser.add_argument('--skip_user', type=str, default='no')
    parser.add_argument('--unbalanced', type=str, default='no')
    parser.add_argument('--load_model', type=str, default='no')
    parser.add_argument('--percentage', type=float, default=.2, help='percent of data to give change user in unbalanced scenario')
    parser.add_argument('--multiple', type=str, default='no', help='Change multiple clients epsilon values')
    parser.add_argument('--how_many', type=int, default=1, help='how many clients to change')
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help="number or each type of kernel")
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help="comma-seperated kernel size to use for convolution")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imgs")
    parser.add_argument('--norm', type=str, default="batch_norm", help="batch_norm, layer_norm, or none")
    parser.add_argument('--num_filters', type=int, default=32, help="NUmber of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot")
    parser.add_argument('--max_pool', type=bool, default=True, help="Whether to user max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='adult', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--iid', type=bool, default=False, help="Whether to use iid data or not")
    parser.add_argument('---unequal', type=bool, default=False, help="Whether to use unequal data splits for non-iid setting")
    parser.add_argument('--stopping_rounds', type=int, default=10, help="rounds of early stopping")
    parser.add_argument('--verbose', type=bool, default=True, help="verbose")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    
    args = parser.parse_args()

    return args


# best params for acsincome federated: epochs - 100, num_users = 5, local_ep=3, local_bs=1, lr=.00001, momentum=0.0
# solo eps = 100, batch = 512, lr = 0.01, momentum=0