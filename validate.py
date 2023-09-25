import os
import torch
from common.utils.meter import AverageMeter, ProgressMeter
import argparse
from torch.utils.data import DataLoader
from icon.uda_backbone import ImageClassifier
from common.utils.metric import accuracy, ConfusionMatrix
import time
import torch.nn.functional as F


def validate_model(val_loader: DataLoader, source_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace, device, identifier="default"):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None

    labels = []
    clusters = []
    pseudo_labels = []
    features = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            
            # compute output
            # output, output_alt, _, _, _, _ = model(images)
            o = model(images)
            output = o["y"]
            output_cluster = o["y_cluster_u"]
            features_batch = o["bottleneck_feature"]
            loss = F.cross_entropy(output, target)
            _, pseudo_clusters = torch.max(F.softmax(output_cluster), dim=-1)
            _, pseudo_labels_batch = torch.max(F.softmax(output), dim=-1)

            labels.append(target.cpu())
            clusters.append(pseudo_clusters.cpu())
            pseudo_labels.append(pseudo_labels_batch.cpu())
            features.append(features_batch.cpu())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if confmat:
            _, acc, _ = confmat.compute()
            avg_return = acc.mean().item() * 100
            print(confmat.format(classes))
        else:
            avg_return = top1.avg

        labels = torch.cat(labels, dim=0)
        clusters = torch.cat(clusters, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
        features = torch.cat(features, dim=0)
    return avg_return