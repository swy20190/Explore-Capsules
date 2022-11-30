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

import model_14


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--dataset', default ='dataset/deepfaketimit', help='path to root dataset')
parser.add_argument('--train_set', default ='train', help='train set')
parser.add_argument('--val_set', default ='validation', help='validation set')
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
    """Computes the accuracy for top k"""
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


def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    train_len = len(train_loader)
    epoch_acc = 0
    end = time.time()
    vgg_ext = model_14.VggExtractor()
    vgg_ext.to(device)

    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        data = vgg_ext(data)
        output = model(data)
        r = (1. * batch_idx + (epoch - 1) * train_len) / (args.niter * train_len)
        loss = criterion(output, target, r)
        acc = accuracy(output, target)

        global_step = (batch_idx + 1) + (epoch - 1) * len(train_loader)
        exp_lr_decay(optimizer=optimizer, global_step=global_step)  # moein - change the learning rate exponentially

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        epoch_acc += acc[0].item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Loss: {:.6f}\tAccuracy: {:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.item(), acc[0].item(),
                batch_time=batch_time, data_time=data_time))
    return epoch_acc, loss.item()


def snapshot(model, folder, epoch):
    path = os.path.join(folder, 'model_{}.pth'.format(epoch))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print('saving model to {}'.format(path))
    torch.save(model.state_dict(), path)


def test(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    acc = 0
    test_len = len(test_loader)
    vgg_ext = model_14.VggExtractor()
    vgg_ext.to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = vgg_ext(data)
            output = model(data)
            test_loss += criterion(output, target, r=1).item()
            acc += accuracy(output, target)[0].item()

    test_loss /= test_len
    acc /= test_len
    print('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f} \n'.format(
        test_loss, acc))
    return acc, test_loss


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
    A, B, C, D = 16, 64, 16, 8
    model = EMCapsules(A=A, B=B, C=C, D=D, E=num_class, iters=args.em_iters).to(device)

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = test(test_loader, model, criterion, device)
    for epoch in range(1, args.niter + 1):
        acc, loss_train = train(train_loader, model, criterion, optimizer, epoch, device)
        acc /= len(train_loader)

        acc_test, loss_test = test(test_loader, model, criterion, device)
        text_writer.write('%d,%.4f,%.2f,%.4f,%.2f\n'
                          % (epoch, loss_train, acc , loss_test, acc_test ))

        text_writer.flush()

    snapshot(model, args.snapshot_folder, args.niter)
    text_writer.close()


if __name__ == '__main__':
    main()
