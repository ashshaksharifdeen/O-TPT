import os
import time
import random

import numpy as np

import shutil
from enum import Enum

import torch
import torchvision.transforms as transforms


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#pytorch implementation of CELoss

import torch
import torch.nn.functional as F

class CELoss(object):
    def __init__(self, n_bins=20, n_data=0, n_class=0):
        self.n_bins = n_bins
        self.n_data = n_data
        self.n_class = n_class

    def compute_bin_boundaries(self, probabilities=None):
        if probabilities is None:
            bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            bin_n = self.n_data // self.n_bins
            bin_boundaries = []

            probabilities_sort, _ = torch.sort(probabilities)

            for i in range(self.n_bins):
                bin_boundaries.append(probabilities_sort[i * bin_n])
            bin_boundaries.append(torch.tensor(1.0))

            self.bin_lowers = torch.stack(bin_boundaries[:-1])
            self.bin_uppers = torch.stack(bin_boundaries[1:])

    def get_probabilities(self, softmax_output,confidences,predictions, labels,args):
        """
        if logits:
            self.probabilities = F.softmax(output, dim=1)
        else:
            self.probabilities = softmax_output"""
        self.probabilities = softmax_output.to(device=args.gpu)
        predictions = torch.tensor(predictions).int().to(device=args.gpu)
        labels = torch.tensor(labels).int().to(device=args.gpu)
        confidences = torch.tensor(confidences).int().to(device=args.gpu)
        self.labels = labels
        self.confidences = confidences #torch.max(self.probabilities, dim=1).values
        self.predictions = predictions #torch.argmax(self.probabilities, dim=1)
        self.accuracies = self.predictions.eq(labels)

    def binary_matrices(self):
        idx = torch.arange(self.n_data)
        pred_matrix = torch.zeros((self.n_data, self.n_class))
        label_matrix = torch.zeros((self.n_data, self.n_class))

        pred_matrix[idx, self.predictions] = 1
        label_matrix[idx, self.labels] = 1

        self.acc_matrix = pred_matrix.eq(label_matrix)

    def compute_bins(self, index=None):
        self.bin_prop = torch.zeros(self.n_bins)
        self.bin_acc = torch.zeros(self.n_bins)
        self.bin_conf = torch.zeros(self.n_bins)
        self.bin_score = torch.zeros(self.n_bins)

        if index is None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:, index]
            accuracies = self.labels.eq(index).float()

        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            #in_bin = in_bin.nonzero(as_tuple=True)[0]
            self.bin_prop[i] = in_bin.float().mean()

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = accuracies[in_bin].float().mean()
                self.bin_conf[i] = confidences[in_bin].float().mean()
                self.bin_score[i] = (self.bin_conf[i] - self.bin_acc[i]).abs()


"""class CELoss(object):

    def compute_bin_boundaries(self, probabilities = np.array([])):

        #uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            #size of bins 
            bin_n = int(self.n_data/self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)  

            for i in range(0,self.n_bins):
                bin_boundaries = np.append(bin_boundaries,probabilities_sort[i*bin_n])
            bin_boundaries = np.append(bin_boundaries,1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]


    def get_probabilities(self, confidences, labels, logits):

        softmax = torch.nn.Softmax(dim=1)
        #If not probabilities apply softmax!
        
        if logits:
            self.probabilities = softmax(output)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=1) #no need i am passing list of maximum confidences
        self.predictions = np.argmax(self.probabilities, axis=1) #no need i am passing this
        self.accuracies = np.equal(self.predictions,labels)

    def binary_matrices(self):
        idx = np.arange(self.n_data)
        #make matrices of zeros
        pred_matrix = np.zeros([self.n_data,self.n_class])
        label_matrix = np.zeros([self.n_data,self.n_class])
        #self.acc_matrix = np.zeros([self.n_data,self.n_class])
        pred_matrix[idx,self.predictions] = 1
        label_matrix[idx,self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)


    def compute_bins(self, index = None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:,index]
            accuracies = (self.labels == index).astype("float")


        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])"""   

class SCELoss(CELoss):
    #list_max_confidence, list_prediction,list_label,number_of_class,
    def loss(self, softmax_output,list_max_confidence, list_prediction,list_label, number_of_class,n_bins, args,logits = False):
        sce = 0.0
        self.n_bins = n_bins
        self.n_data = len(list_max_confidence)
        self.n_class = number_of_class
        
        super().compute_bin_boundaries()
        super().get_probabilities(softmax_output,list_max_confidence, list_prediction, list_label,args)
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bins()
            sce += np.dot(self.bin_prop, self.bin_score)

        return sce/self.n_class





class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        """
        Definition: The top-1 accuracy is the percentage of times the model's 
        highest confidence prediction (i.e., the class with the highest predicted 
        probability) matches the true label.
        
        Definition: The top-5 accuracy is the percentage of times the true label is among 
        the model's top five highest confidence predictions.
        """
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        

def load_model_weight(load_path, model, device, args):
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint['state_dict']
        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        args.start_epoch = checkpoint['epoch']
        try:
            best_acc1 = checkpoint['best_acc1']
        except:
            best_acc1 = torch.tensor(0)
        if device != 'cpu':
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(device)
        try:
            model.load_state_dict(state_dict)
        except:
            # TODO: implement this method for the generator class
            model.prompt_generator.load_state_dict(state_dict, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_path, checkpoint['epoch']))
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("=> no checkpoint found at '{}'".format(load_path))


def validate(val_loader, model, criterion, args, output_mask=None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                if output_mask:
                    output = output[:, output_mask]
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        progress.display_summary()

    return top1.avg
