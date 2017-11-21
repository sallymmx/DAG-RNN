from __future__ import print_function
import argparse
import os.path as osp
import shutil
import time
import __init__
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import models, transforms 
from torch.autograd import Variable

from model import VGG16_MIL, Resnet50_MIL,Resnet50_RNN_DAG, Resnet101_MIL, Resnet152_MIL
from nuswide import NUS_WIDE_Dataset


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names.append("VGG16_MIL")
model_names.append("Resnet50_MIL")
model_names.append("Resnet50_RNN_DAG")
model_names.append("Resnet101_MIL")
model_names.append("Resnet152_MIL")

parser = argparse.ArgumentParser(description='PyTorch NUS-WIDE Training')
parser.add_argument('--trainval_dir', metavar='TRAINVAL_DIR', default='data',
        help='path which contains train.txt and val.txt)')
parser.add_argument('--root_dir', metavar='ROOT_DIR', 
        default='/home/zemeng.wmm/zemeng/visual-concepts/code/nus/new_imgs',
        help='prefix dir path which contains pics')
parser.add_argument('--checkpoint_dir', metavar='',default='checkpoint_v2',
        help='path to save checkpoint')
parser.add_argument('--gpu', metavar='GPU', default='0',
        help='which gpu device to use, if no-cuda is true, ignore this')
parser.add_argument('--arch', '-a', metavar='ARCH', default='Resnet50_MIL',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--resize-height', default=400, type=int,
        metavar='N', help='resize-height size (default: 400)')
parser.add_argument('--resize-width', default=400, type=int,
        metavar='N', help='resize-width size (default: 400')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',default=True,
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',default=True,
                    help='use pre-trained model')
parser.add_argument('--no-cuda', action='store_true', default=False, 
        help='disables CUDA training')
parser.add_argument('--numclass', default=81, type=int,
        metavar='N', help='numclass (default: 81)')		
parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
best_prec1 = 0

def main():
    global args, best_prec1,generator
    
    args = parser.parse_args()

    print('Called with args: ')
    print(args)
 

    # cuda config
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        ngpu = len(args.gpu.split(','))
        if ngpu > 1:
            args.multi_gpu = True
            gpu_list = []
            for i in xrange(ngpu):
                gpu_list.append(int(args.gpu.split(',')[i]))
        elif ngpu == 1:
            args.multi_gpu = False
            torch.cuda.set_device(int(args.gpu))
        else:
            raise Exception("wrong gpu input")
    
    # seed config     use  generator.seed() 
    generator = torch.manual_seed(args.seed)

    # model config
    assert args.arch in ['VGG16_MIL', 'Resnet50_MIL','Resnet101_MIL', 'Resnet50_RNN_DAG', 'Resnet152_MIL']
    
    """ VGG16_MIL pretrained, lr = 0.000015625 """
    if args.arch == 'VGG16_MIL':
        print('args.pretrained = [{}]'.format(args.pretrained))
        model = VGG16_MIL(args.pretrained, args.numclass)
        
        if args.cuda:
            if args.multi_gpu:
                model.features = torch.nn.DataParallel(model.features, device_ids=gpu_list)
            model.cuda()

    """ Resnet50_MIL pretrained, lr = 0.01 """
    if args.arch == 'Resnet50_MIL':
        print('args.pretrained = [{}]'.format(args.pretrained))
        model = Resnet50_MIL(args.pretrained, args.numclass)
        
        if args.cuda:
            if args.multi_gpu:
                model = torch.nn.DataParallel(model, device_ids=gpu_list).cuda()
            else:
                model.cuda()

    """ Resnet50_RNN_DAG pretrained, lr = 0.01 """
    if args.arch == 'Resnet50_RNN_DAG':
        print('args.pretrained = [{}]'.format(args.pretrained))
        model = Resnet50_RNN_DAG(args.pretrained, args.numclass)
        
        if args.cuda:
            if args.multi_gpu:
                model = torch.nn.DataParallel(model, device_ids=gpu_list).cuda()
            else:
                model.cuda()
				
	""" Resnet101_MIL pretrained, lr = 0.01 """
    if args.arch == 'Resnet101_MIL':
        print('args.pretrained = [{}]'.format(args.pretrained))
        model = Resnet101_MIL(args.pretrained, args.numclass)
        
        if args.cuda:
            if args.multi_gpu:
                model = torch.nn.DataParallel(model, device_ids=gpu_list).cuda()
            else:
                model.cuda()
    
	""" Resnet152_MIL pretrained, lr = 0.01 """
    if args.arch == 'Resnet152_MIL':
        print('args.pretrained = [{}]'.format(args.pretrained))
        model = Resnet152_MIL(args.pretrained, args.numclass)
        
        if args.cuda:
            if args.multi_gpu:
                model = torch.nn.DataParallel(model, device_ids=gpu_list).cuda()
            else:
                model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss()
    if args.cuda:
        criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum, weight_decay=args.weight_decay)

    #optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # optionally resume from a checkpoint
    '''
    if args.resume:
        if osp.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
    '''
    if args.resume:
        model_dict = model.state_dict()
        if osp.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # 1. filter out unnecessary keys
            checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            model_dict.update(checkpoint['state_dict']) 
            model.load_state_dict(model_dict)
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale([args.resize_width, args.resize_height]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Scale([args.resize_width, args.resize_height]),
            transforms.ToTensor(),
            normalize,
        ]),
    }
    
    train_dataset = NUS_WIDE_Dataset(args.trainval_dir, args.root_dir, train=True,
            transform=data_transforms['train'])

    val_dataset = NUS_WIDE_Dataset(args.trainval_dir, args.root_dir, train=False,
            transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
            shuffle=True, **kwargs) 
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
            shuffle=False, **kwargs)

    #if args.evaluate:
    #    validate(val_loader, model, criterion)
    #    return

    for epoch in range(args.start_epoch, args.epochs):
       
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('epoch [{}], best_prec1 is {:.3f}'.format(epoch, best_prec1))
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,epoch)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()

    f1  = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, sample  in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        input = sample['image']
        target = sample['targets'].type(torch.FloatTensor)
        if args.cuda:
            input = input.cuda() 
            target = target.cuda(async=True)
        
        input_var = Variable(input)
        target_var = Variable(target)
        batchsize,classes = target_var.size()
        target_var = target_var.view(batchsize,classes,1,1)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        loss = loss*20
        # measure accuracy and record loss
        prec3, rec3, f1_3 = accuracy(output.data, target, topk=(3, ))
        losses.update(loss.data[0], input.size(0))
        prec.update(prec3[0], input.size(0))
        rec.update(rec3[0], input.size(0))
        f1.update(f1_3[0], input.size(0)) 

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@3 {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Rec@3 {rec.val:.3f} ({rec.avg:.3f})\t'
                  'F1@3 {f1.val:.3f} ({f1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, prec=prec, rec=rec, f1=f1))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()
    f1  = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, sample in enumerate(val_loader):
        input = sample['image']
        target = sample['targets'].type(torch.FloatTensor)
        if args.cuda:
             input = input.cuda()
             target = target.cuda(async=True)
        input_var = Variable(input, volatile=True)
        target_var = Variable(target, volatile=True)
        #print(target_var.size())
        batchsize,classes = target_var.size()
        target_var = target_var.view(batchsize,classes,1,1)
        #print(target_var)
        # compute output
        output = model(input_var)
        #print(output)
        
        loss =20* criterion(output, target_var)
        
        #print(loss)

        # measure accuracy and record loss
        prec3, rec3, f1_3 = accuracy(output.data, target, topk=(3,))
        losses.update(loss.data[0], input.size(0))
        prec.update(prec3[0], input.size(0))
        rec.update(rec3[0], input.size(0))
        f1.update(f1_3[0],input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@3 {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Rec@3 {rec.val:.3f} ({rec.avg:.3f})\t'
                  'F1@3 {f1.val:.3f} ({f1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   prec=prec, rec=rec, f1=f1))

    print(' * Prec@3 {prec.avg:.3f} Rec@3 {rec.avg:.3f} F1@3 {f1.avg:.3f}'
          .format(prec=prec, rec=rec,f1=f1))

    return f1.avg


def save_checkpoint(state, is_best,epoch, filename='checkpoint.pth.tar'):
    filename='Epoch'+str(epoch)+'_'+filename
    torch.save(state, osp.join(args.checkpoint_dir, filename))
    if is_best:
        shutil.copyfile(osp.join(args.checkpoint_dir, filename),
                osp.join(args.checkpoint_dir, 'model_best.pth.tar'))


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    print('Epoch '+str(epoch)+' lr: '+str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    output = torch.squeeze(output)
    target = torch.squeeze(target)    
    batch_size = target.size(0)
     
    res = []
    for k in topk:
        
       valuesk, predk = output.topk(k, 1, True, True)
       lowestk = valuesk[:,-1].unsqueeze(1)
       if args.cuda:
         
         pred_newk = torch.ge(output,lowestk.expand_as(output)) 
         pred_newk = pred_newk.type(torch.cuda.FloatTensor)
         correct_k = target.mul(pred_newk.eq(target).type(torch.cuda.FloatTensor))
       else:   
         pred_newk = torch.ge(output,lowestk.expand_as(output)) 
         pred_newk = pred_newk.type(torch.FloatTensor)
         correct_k = target.mul(pred_newk.eq(target).type(torch.FloatTensor))
       #print(input)
       #print(correct_k)
       #print(correct_k.view(-1))
       correct_k = correct_k.view(-1).float().sum(0)
       #print(correct_k)
  
       pred_k = pred_newk.view(-1).float().sum(0)
       prec_k = correct_k/(pred_k+1e-16)
       label_k = target.view(-1).float().sum(0)
       rec_k = correct_k / (label_k+1e-16)
       f1_k = 2 * prec_k * rec_k /(prec_k+rec_k+1e-16)
       res.append(prec_k)
       res.append(rec_k) 
       res.append(f1_k)
    return res


if __name__ == '__main__':
    main()

