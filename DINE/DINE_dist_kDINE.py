import argparse
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import distutils
import distutils.util
import logging

import sys, os
sys.path.append("../util/")
from utils import resetRNGseed, init_logger, get_hostname, get_pid
sys.path.append("../pklib")
from pksolver import PK_solver

import time
timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    count = np.zeros(args.class_num)
    tr_txt = []
    te_txt = []
    for i in range(len(txt_src)):
        line = txt_src[i]
        reci = line.strip().split(' ')
        if count[int(reci[1])] < 3:
            count[int(reci[1])] += 1
            te_txt.append(line)
        else:
            tr_txt.append(line)

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["source_tr"] = ImageList(tr_txt, root="../data/{}/".format(args.dset), transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, root="../data/{}/".format(args.dset), transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["target"] = ImageList_idx(txt_tar, root="../data/{}/".format(args.dset), transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["target_te"] = ImageList(txt_tar, root="../data/{}/".format(args.dset), transform=image_test())
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, root="../data/{}/".format(args.dset), transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if netB is None:
                outputs = netC(netF(inputs))
            else:
                outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item() / np.log(all_label.size()[0])

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int),:]
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, mean_ent
    else:
        return accuracy*100, mean_ent

def train_source_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()
    netC = network.feat_classifier_simpl(class_num=args.class_num, feat_dim=netF.in_features).cuda()

    param_group = []
    learning_rate = args.lr_src
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netF(inputs_source))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)(outputs_source, labels_source)            
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, None, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.s, iter_num, max_iter, acc_s_te)
            if args.dset == 'visda-2017':
                acc_s_te, acc_list, _ = cal_acc(dset_loaders['source_te'], netF, None, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.s, iter_num, max_iter,
                                                                            acc_s_te) + '\n' + acc_list
            logging.info(log_str)

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netC.train()
                
    torch.save(best_netF, osp.join(args.output_dir_src,'{}_{}_source_F.pt'.format(args.s, args.net_src)))
    torch.save(best_netC, osp.join(args.output_dir_src, '{}_{}_source_C.pt'.format(args.s, args.net_src)))

    return netF, netC

def test_target_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()
    netC = network.feat_classifier_simpl(class_num = args.class_num, feat_dim=netF.in_features).cuda()
    
    args.modelpath = args.output_dir_src + '/{}_{}_source_F.pt'.format(args.s, args.net_src)
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/{}_{}_source_C.pt'.format(args.s, args.net_src)
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders['test'], netF, None, netC, False)
    log_str = '\nTask: {}->{}, Accuracy = {:.2f}%'.format(args.s, args.t, acc)
    if args.dset == 'visda-2017':
        acc_s_te, acc_list, _ = cal_acc(dset_loaders['test'], netF, None, netC, True)
        log_str = 'Task: {}, Accuracy = {:.2f}%'.format(args.s,  acc_s_te) + '\n' + acc_list

    logging.info(log_str)

def copy_target_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()   
    netC = network.feat_classifier_simpl(class_num=args.class_num, feat_dim=netF.in_features).cuda()
    
    args.modelpath = args.output_dir_src + '/{}_{}_source_F.pt'.format(args.s, args.net_src)
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/{}_{}_source_C.pt'.format(args.s, args.net_src)
    netC.load_state_dict(torch.load(args.modelpath))
    source_model = nn.Sequential(netF, netC).cuda()
    source_model.eval()

    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net, pretrain=True).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    ent_best = 1.0
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // 10
    iter_num = 0

    model = nn.Sequential(netF, netB, netC).cuda()
    model.eval()

    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders["target_te"])
        for i in range(len(dset_loaders["target_te"])):
            data = iter_test.next()
            inputs, labels = data[0], data[1]
            inputs = inputs.cuda()
            outputs = source_model(inputs)
            outputs = nn.Softmax(dim=1)(outputs)
            _, src_idx = torch.sort(outputs, 1, descending=True)
            if args.topk > 0:
                topk = np.min([args.topk, args.class_num])
                for i in range(outputs.size()[0]):
                    outputs[i, src_idx[i, topk:]] = (1.0 - outputs[i, src_idx[i, :topk]].sum())/ (outputs.size()[1] - topk)

            if start_test:
                all_output = outputs.float()
                all_label = labels
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels), 0)
        mem_P = all_output.detach()

        # get ground-truth label probabilities of target domain
        cls_probs = torch.eye(args.class_num)[all_label].sum(0)
        cls_probs = cls_probs / cls_probs.sum()

        pk_solver = PK_solver(all_label.shape[0], args.class_num, pk_prior_weight=args.pk_prior_weight)
        if args.pk_type == 'ub':
            pk_solver.create_C_ub(cls_probs.cpu().numpy(), args.pk_uconf)
        elif args.pk_type == 'br':
            pk_solver.create_C_br(cls_probs.cpu().numpy(), args.pk_uconf)
        else:
            raise NotImplementedError

        mem_label = obtain_label(mem_P.cpu(), all_label.cpu(), None, args, pk_solver)
        mem_label = torch.from_numpy(mem_label).cuda()
        mem_label = torch.eye(args.class_num)[mem_label].cuda()

    model.train()
    while iter_num < max_iter:
 
        if args.ema < 1.0 and iter_num > 0 and iter_num % interval_iter == 0:
            model.eval()
            start_test = True
            with torch.no_grad():
                iter_test = iter(dset_loaders["target_te"])
                for i in range(len(dset_loaders["target_te"])):
                    data = iter_test.next()
                    inputs = data[0]
                    inputs = inputs.cuda()
                    outputs = model(inputs)
                    feas = model[1](model[0](inputs))
                    outputs = nn.Softmax(dim=1)(outputs)
                    if start_test:
                        all_fea = feas.float().cpu()
                        all_output = outputs.float()
                        start_test = False
                    else:
                        all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                        all_output = torch.cat((all_output, outputs.float()), 0)
                mem_P = mem_P * args.ema + all_output.detach() * (1 - args.ema)
            model.train()

            mem_label = obtain_label(mem_P.cpu(), all_label.cpu(), all_fea, args, pk_solver)
            mem_label = torch.from_numpy(mem_label).cuda()
            mem_label = torch.eye(args.class_num)[mem_label].cuda()

        try:
            inputs_target, y, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, y, tar_idx = iter_target.next()

        if inputs_target.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, power=1.5)
        inputs_target = inputs_target.cuda()
        with torch.no_grad():
            outputs_target_by_source = mem_P[tar_idx, :]
            _, src_idx = torch.sort(outputs_target_by_source, 1, descending=True)
        outputs_target = model(inputs_target)
        outputs_target = torch.nn.Softmax(dim=1)(outputs_target)

        target = (outputs_target_by_source + mem_label[tar_idx, :]*0.9 + 1/mem_label.shape[-1]*0.1) / 2
        if iter_num < interval_iter and args.dset == "visda-2017":
            target = outputs_target_by_source

        classifier_loss = nn.KLDivLoss(reduction='batchmean')(outputs_target.log(), target)
        optimizer.zero_grad()

        entropy_loss = torch.mean(loss.Entropy(outputs_target))
        msoftmax = outputs_target.mean(dim=0)
        gentropy_loss = torch.sum(- msoftmax * torch.log(msoftmax + 1e-5))
        entropy_loss -= gentropy_loss
        classifier_loss += entropy_loss

        classifier_loss.backward()

        if args.mix > 0:
            alpha = 0.3
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(inputs_target.size()[0]).cuda()
            mixed_input = lam * inputs_target + (1 - lam) * inputs_target[index, :]
            mixed_output = (lam * outputs_target + (1 - lam) * outputs_target[index, :]).detach()

            update_batch_stats(model, False)
            outputs_target_m = model(mixed_input)
            update_batch_stats(model, True)
            outputs_target_m = torch.nn.Softmax(dim=1)(outputs_target_m)
            classifier_loss = args.mix*nn.KLDivLoss(reduction='batchmean')(outputs_target_m.log(), mixed_output)
            classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            acc_s_te, mean_ent = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = 'Task: {}->{}, Iter:{}/{}; Accuracy = {:.2f}%, Ent = {:.4f}'.format(args.s, args.t, iter_num, max_iter, acc_s_te, mean_ent)
            if args.dset == 'visda-2017':
                acc_s_te, acc_list, mean_ent = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}->{}, Iter:{}/{}; Accuracy = {:.2f}%, Ent = {:.4f}'.format(args.s, args.t, iter_num, max_iter,
                                                                            acc_s_te, mean_ent) + '\n' + acc_list
            logging.info(log_str)
            model.train()

    torch.save(netF.state_dict(), osp.join(args.output_dir, "{}_{}_{}_{}_target_F".format(args.timestamp, args.s, args.t, args.net) + ".pt"))
    torch.save(netB.state_dict(), osp.join(args.output_dir, "{}_{}_{}_{}_target_B".format(args.timestamp, args.s, args.t, args.net) + ".pt"))
    torch.save(netC.state_dict(), osp.join(args.output_dir, "{}_{}_{}_{}_target_C".format(args.timestamp, args.s, args.t, args.net) + ".pt"))


def obtain_label(mem_P, all_label, all_fea, args, pk_solver):
    predict = mem_P.argmax(-1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    avg_accuracy = (matrix.diagonal() / matrix.sum(axis=1)).mean()

    # update labels with prior knowledge
    probs = mem_P
    # first solve without smooth regularization
    pred_label_PK = pk_solver.solve_soft(probs)

    acc_PK = np.sum(pred_label_PK == all_label.float().numpy()) / float(all_label.size()[0])
    matrix_PK = confusion_matrix(all_label.float().numpy(), pred_label_PK)
    avg_acc_PK = (matrix_PK.diagonal() / matrix_PK.sum(axis=1)).mean()
    log_str = 'PK Accuracy = {:.2f}% -> {:.2f}%    Per_class_accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc_PK * 100, avg_accuracy * 100, avg_acc_PK * 100)
    logging.info(log_str)

    if args.pk_knn > 0 and all_fea is not None:
        # now solve with smooth regularization
        predict = predict.cpu().numpy()
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        idx_unconf = np.where(pred_label_PK != predict)[0]
        knn_sample_idx = idx_unconf
        idx_conf = np.where(pred_label_PK == predict)[0]

        if len(idx_unconf) > 0 and len(idx_conf) > 0:
            # get knn of each samples
            dd_knn = cdist(all_fea[idx_unconf], all_fea[idx_conf], args.distance)
            knn_idx = []
            K = args.pk_knn
            for i in range(dd_knn.shape[0]):
                ind = np.argpartition(dd_knn[i], K)[:K]
                knn_idx.append(idx_conf[ind])

            knn_idx = np.stack(knn_idx, axis=0)
            knn_regs = list(zip(knn_sample_idx, knn_idx))
            pred_label_PK = pk_solver.solve_soft_knn_cst(probs, knn_regs=knn_regs)


        acc_PK = np.sum(pred_label_PK == all_label.float().numpy()) / len(all_fea)
        matrix_PK = confusion_matrix(all_label.float().numpy(), pred_label_PK)
        avg_acc_PK = (matrix_PK.diagonal() / matrix_PK.sum(axis=1)).mean()
        if args.da == 'pda':
            avg_acc_PK = 0
        log_str = 'PK Accuracy = {:.2f}% -> {:.2f}%    Per_class_accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc_PK * 100, avg_accuracy * 100, avg_acc_PK * 100)
        logging.info(log_str)

    return pred_label_PK.astype('int')

def update_batch_stats(model, flag):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.update_batch_stats = flag

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DINE')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=str, default=None, help="source")
    parser.add_argument('--t', type=str, default=None, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['visda-2017', 'office31', 'image-clef', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--lr_src', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net_src', type=str, default='resnet50', help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--output_src', type=str, default='san')  

    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--topk', type=int, default=1)

    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--ema', type=float, default=0.6)
    parser.add_argument('--mix', type=float, default=1.0)

    parser.add_argument('--timestamp', default=timestamp, type=str, help='timestamp')
    parser.add_argument('--use_file_logger', default='True', type=lambda x: bool(distutils.util.strtobool(x)),
                        help='whether use file logger')
    parser.add_argument('--names', default=[], type=list, help='names of tasks')

    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])

    parser.add_argument('--pk_uconf', type=float, default=0.0)
    parser.add_argument('--pk_type', type=str, default="ub")
    parser.add_argument('--pk_allow', type=int, default=None)
    parser.add_argument('--pk_temp', type=float, default=1.0)
    parser.add_argument('--pk_prior_weight', type=float, default=10.)
    parser.add_argument('--pk_knn', type=int, default=1)
    parser.add_argument('--method', type=str, default="method")

    args = parser.parse_args()
    args.method = '_'.join(sys.argv[0].split('.py')[0].split('_')[2:]).lower()

    if args.dset == 'office-home':
        args.names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'visda-2017':
        args.names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office31':
        args.names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    resetRNGseed(args.seed)

    if not args.distill:
        dir = "{}_{}_{}_{}_source".format(args.timestamp, args.s, args.da, args.method)
    else:
        dir = "{}_{}_{}_{}".format(args.timestamp, args.s, args.da, args.method)
    if args.use_file_logger:
        init_logger(dir, True, '../logs/DINE/{}/'.format(args.method))
    logging.info("{}:{}".format(get_hostname(), get_pid()))

    folder = '../data/'
    args.s_dset_path = folder + args.dset + '/image_list/' + args.s + '.txt'
    args.t_dset_path = folder + args.dset + '/image_list/' + args.s + '.txt'
    args.test_dset_path = folder + args.dset + '/image_list/' + args.s + '.txt'

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [33, 32, 36, 15, 19, 2, 46, 49, 48, 53, 47, 54, 4, 18, 57, 23, 0, 45, 1, 38, 5, 13, 50, 11, 58]

    args.output_dir_src = "../checkpoints/DINE/{}/source/{}/".format(args.seed, args.da)

    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)
        
    if not args.distill: 
        logging.info(print_args(args))
        train_source_simp(args)

        for t in args.names:
            if t == args.s:
                continue
            args.t = t
            args.t_dset_path = folder + args.dset + '/image_list/' + args.t + '.txt'
            args.test_dset_path = folder + args.dset + '/image_list/' + args.t + '.txt'
                            
            test_target_simp(args)

    if args.distill:
        for t in args.names:
            if t == args.s:
                continue
            args.t = t
            args.output_dir = "../checkpoints/DINE/{}/target/{}/".format(args.seed, args.da)
            if not osp.exists(args.output_dir):
                os.system('mkdir -p ' + args.output_dir)
            if not osp.exists(args.output_dir):
                os.mkdir(args.output_dir)

            args.t_dset_path = folder + args.dset + '/image_list/' + args.t + '.txt'
            args.test_dset_path = folder + args.dset + '/image_list/' + args.t + '.txt'

            logging.info(print_args(args))

            copy_target_simp(args)