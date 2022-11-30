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
from SRCaps import SRNet
from SRCaps import BasicBlock

import model_14


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--dataset', default ='dataset/deepfaketimit', help='path to root dataset')
parser.add_argument('--train_set', default ='train', help='train set')
parser.add_argument('--val_set', default ='validation', help='validation set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=2, help='batch size')
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')

# no use
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
# no use
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='checkpoints', help='folder to output model checkpoints')

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
parser.add_argument('--em-iters', type=int, default=5, metavar='N',
                    help='iterations of EM Routing')
parser.add_argument('--snapshot-folder', type=str, default='./snapshots_emcaps', metavar='SF',
                    help='where to store the snapshots')




def get_setting(args):
    opt = parser.parse_args(args=[])

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_train = dset.ImageFolder(root=os.path.join(opt.dataset, opt.train_set), transform=transform_fwd)
    assert dataset_train
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True,
                                                   num_workers=int(opt.workers))

    dataset_val = dset.ImageFolder(root=os.path.join(opt.dataset, opt.val_set), transform=transform_fwd)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize, shuffle=False,
                                                 num_workers=int(opt.workers))

    num_class = 2

    return num_class, dataloader_train, dataloader_val


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def exp_lr_decay(optimizer, global_step, init_lr=3e-3, decay_steps=20000,
                 decay_rate=0.96, lr_clip=3e-3, staircase=False):
    ''' decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)  '''

    if staircase:
        lr = (init_lr * decay_rate ** (global_step // decay_steps))
    else:
        lr = (init_lr * decay_rate ** (global_step / decay_steps))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(epoch, model, train_loader, device, optimizer):
    """
    Train the model for 1 epoch of the training set.
    An epoch corresponds to one full pass through the entire
    training set in successive mini-batches.
    This is used by train() and should not be called manually.
    """
    model.train()
    vgg_ext = model_14.VggExtractor()
    vgg_ext.to(device)

    losses = AverageMeter()
    accs = AverageMeter()

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        b = x.shape[0]
        out = model(x)
        loss = nn.NLLLoss().to(device)

        loss = loss(out, y)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct = (pred == y).float()
        acc = 100 * (correct.sum() / len(y))

        # store
        losses.update(loss.data.item(), x.size()[0])
        accs.update(acc.data.item(), x.size()[0])

        # compute gradients and update SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return accs.avg, losses.avg


def snapshot(model, folder, epoch):
    path = os.path.join(folder, 'model_{}.pth'.format(epoch))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print('saving model to {}'.format(path))
    torch.save(model.state_dict(), path)


def validate(model, test_loader, device):
    """
    Test the model on the held-out test data.
    This function should only be called at the very
    end once the model has finished training.
    """
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = nn.NLLLoss().to(device)

        loss = loss(out, y)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct = (pred == y).float()
        acc = 100 * (correct.sum() / len(y))

        # store
        losses.update(loss.data.item(), x.size()[0])
        accs.update(acc.data.item(), x.size()[0])

    return accs.avg, losses.avg


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.manualSeed)
    if args.cuda:
        torch.cuda.manual_seed(args.manualSeed)

    device = torch.device("cuda" if args.cuda else "cpu")

    text_writer = open(os.path.join(args.outf, 'train.csv'), 'w')

    # datasets
    num_class, train_loader, test_loader = get_setting(args)

    # model
    # A, B, C, D = 64, 8, 16, 16

    model = SRNet(block=BasicBlock, num_blocks=[3, 3, 3], planes=4, cfg_data={'size': 32, 'channels': 3, 'classes': 2},
                  num_caps=32, caps_size=8, depth=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.niter + 1):
        acc, loss_train = train(epoch, model, train_loader, device, optimizer)
        acc /= len(train_loader)

        acc_test, loss_test = validate(model, test_loader, device)
        text_writer.write('%d,%.4f,%.2f,%.4f,%.2f\n'
                          % (epoch, loss_train, acc , loss_test, acc_test ))

        text_writer.flush()

    snapshot(model, args.snapshot_folder, args.niter)
    text_writer.close()


if __name__ == '__main__':
    main()
