from __future__ import print_function
import argparse
import numpy 
import os.path as osp
import shutil
import time
import __init__
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import models, transforms 
from torch.autograd import Variable

from model import VGG16_MIL, Resnet50_MIL, Resnet101_MIL, Resnet152_MIL
from nuswide import NUS_WIDE_Dataset_Test


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names.append("VGG16_MIL")
model_names.append("Resnet50_MIL")
model_names.append("Resnet101_MIL")
model_names.append("Resnet152_MIL")
parser = argparse.ArgumentParser(description='PyTorch NUS-WIDE Testing')
parser.add_argument('--test_dir', metavar='TEST_DIR', default='data',
        help='path which contains test.txt)')
parser.add_argument('--root_dir', metavar='ROOT_DIR', 
        default='/home/zemeng.wmm/zemeng/visual-concepts/code/nus/new_imgs',
        help='prefix dir path which contains pics')
parser.add_argument('--gpu', metavar='GPU', default='0',
        help='which gpu device to use, if no-cuda is true, ignore this')
parser.add_argument('--arch', '-a', metavar='ARCH', default='Resnet101_MIL',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--resize-height', default=400, type=int,
        metavar='N', help='resize-height size (default: 400)')
parser.add_argument('--resize-width', default=400, type=int,
        metavar='N', help='resize-width size (default: 400')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',default=True,
                    help='use pre-trained model')
parser.add_argument('--no-cuda', action='store_true', default=False, 
        help='disables CUDA training')
parser.add_argument('--numclass', default=81, type=int,
        metavar='N', help='numclass (default: 81)')		
parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
parser.add_argument('--save_prob_file', type=str, default='', metavar='SAVE',
        help='the file to save the test results')
def main():
    global args,generator
    
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
    assert args.arch in ['VGG16_MIL', 'Resnet50_MIL', 'Resnet101_MIL', 'Resnet152_MIL']
    
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


    # optionally resume from a checkpoint
    if args.resume:
        if osp.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
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
        'test': transforms.Compose([
            transforms.Scale([args.resize_width, args.resize_height]),
            transforms.ToTensor(),
            normalize,
        ]),
    }
    
    test_dataset = NUS_WIDE_Dataset_Test(args.test_dir, args.root_dir,
            transform=data_transforms['test'])


    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
            shuffle=False, **kwargs) 
            
    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss()        
    # switch to evaluate mode
    model.eval()
    test(test_loader,model,criterion)



def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()
    f1  = AverageMeter()
    
    end = time.time()
    for i, sample in enumerate(test_loader):
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
        
        
        
        loss =20* criterion(output, target_var)
        output_ = torch.squeeze(output)
        output_save = output_.cpu()
        output_save = output_save.data.numpy()        
        #print(output)
        f2 = open(args.save_prob_file,'a')
        for i in range(output_save.shape[0]):
            for (j,precj) in enumerate( output_save[i,:]):
                 f2.write(str(numpy.round(precj,4))+' ' )
            f2.write('\n')
        # measure accuracy and record loss
        prec3, rec3, f1_3 = accuracy(output.data, target, topk=(3,))
        losses.update(loss.data[0], input.size(0))
        prec.update(prec3[0], input.size(0))
        rec.update(rec3[0], input.size(0))
        f1.update(f1_3[0],input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

    print(' * Prec@3 {prec.avg:.3f} Rec@3 {rec.avg:.3f} F1@3 {f1.avg:.3f} Loss {loss.avg:.3f} Time {batch_time.avg:.3f}'
          .format(prec=prec, rec=rec,f1=f1,loss=losses,batch_time=batch_time))
    
    return f1.avg



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



def accuracy(output, target, topk=(3,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    output = torch.squeeze(output)
    target = torch.squeeze(target)    
    batch_size = target.size(0)
     
    res = []       
    valuesk, predk = output.topk(maxk, 1, True, True)
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


