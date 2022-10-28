from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.datasets as dset

import EMCaps
from EMCaps import SpreadLoss
from EMCaps import EMCapsules
from EMtrain import test

import model_14


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--dataset', default ='dataset/deepfaketimit', help='path to root dataset')
parser.add_argument('--train_set', default ='train', help='train set')
parser.add_argument('--test_set', default ='test', help='test set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=8, help='batch size')
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')

# no use
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
# no use
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='checkpoints\deepfaketimit', help='folder to output model checkpoints')

# no use
parser.add_argument('--disable_random', action='store_true', default=False, help='disable randomness for routing matrix')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout percentage')
parser.add_argument('--manualSeed', type=int, default=200, help='manual seed')


parser.add_argument('--weight-decay', type=float, default=2e-7, metavar='WD',
                    help='weight decay (default: 0)')  # moein - according to openreview
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--test-intvl', type=int, default=1, metavar='N',
                    help='test intvl (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--em-iters', type=int, default=2, metavar='N',
                    help='iterations of EM Routing')
parser.add_argument('--snapshot-folder', type=str, default='./snapshots', metavar='SF',
                    help='where to store the snapshots')





def main():
    global args, best_prec1
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.manualSeed)
    if args.cuda:
        torch.cuda.manual_seed(args.manualSeed)

    device = torch.device("cuda" if args.cuda else "cpu")

    # datasets
    num_class = 2
    opt = parser.parse_args(args=[])
    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_test = dset.ImageFolder(root=os.path.join(opt.dataset, opt.test_set), transform=transform_fwd)
    assert dataset_test
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False,
                                                 num_workers=int(opt.workers))


    # model
    A, B, C, D = 64, 8, 16, 16
    # A, B, C, D = 32, 32, 32, 32
    model = EMCapsules(A=A, B=B, C=C, D=D, E=num_class, iters=args.em_iters).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.snapshot_folder, 'model_' + str(opt.niter) + '.pth')))

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)

    best_acc = test(test_loader, model, criterion, device)
    return best_acc


if __name__ == '__main__':
    main()
