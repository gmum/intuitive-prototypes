from pipnet.pipnet import PIPNet, get_network
from util.log import Log
import torch.nn as nn
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
from util.func import init_weights_xavier
from pipnet.train import train_pipnet
from pipnet.test import eval_pipnet, get_thresholds, eval_ood
from util.eval_cub_csv import eval_prototypes_cub_parts_csv, get_topk_cub, get_proto_patches_cub
import torch
from util.vis_pipnet import visualize, visualize_topk
from util.visualize_prediction import vis_pred, vis_pred_experiments
import sys, os
import random
import numpy as np
from shutil import copy
import matplotlib.pyplot as plt
from copy import deepcopy


class DinoPIPNet(PIPNet):

    def forward(self, *args, **kwargs):
        _proto_features, pooled, _out = self.forward_full_return(*args, **kwargs)
        return pooled

    def forward_full_return(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


def load_pipnet_for_dino(args=None):
    args = args or get_args()
    args.dataset = 'CUB-200-2011'
    args.validation_size = 0.0
    args.net = 'resnet50'
    args.batch_size = 32
    args.batch_size_pretrain = 64
    args.epochs = 100
    args.lr_block = 0.0005
    args.lr_net = 0.0005
    args.log_dir = 'runs/pipnet_cub_resnet50_dino'
    args.num_features = 0
    args.image_size = 224
    args.state_dict_dir_net = ''
    args.freeze_epochs = 10
    args.dir_for_saving_images = 'Visualization_results'
    args.epochs_pretrain = 10
    args.seed = 1
    args.gpu_ids = ''
    args.num_workers = 8
    args.optimizer = 'Adam'
    assert args.batch_size > 1
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    # Log the run arguments
    save_args(args, log.metadata_dir)

    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids != '':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))

    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids) == 1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids) == 0:
            device = torch.device('cuda')
            print("CUDA device set without id specification", flush=True)
            device_ids.append(torch.cuda.current_device())
        else:
            print(
                "This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU.",
                flush=True)
            device_str = ''
            for d in device_ids:
                device_str += str(d)
                device_str += ","
            device = torch.device('cuda:' + str(device_ids[0]))
    else:
        device = torch.device('cpu')

    # Log which device was actually used
    print("Device used: ", device, "with id", device_ids, flush=True)

    # Obtain the dataset and dataloaders
    trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(
        args, device)
    if len(classes) <= 20:
        if args.validation_size == 0.:
            print("Classes: ", testloader.dataset.class_to_idx, flush=True)
        else:
            print("Classes: ", str(classes), flush=True)

    # Create a convolutional network based on arguments and add 1x1 conv layer
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(len(classes), args)

    # Create a PIP-Net
    net = DinoPIPNet(num_classes=len(classes),
                 num_prototypes=num_prototypes,
                 feature_net=feature_net,
                 args=args,
                 add_on_layers=add_on_layers,
                 pool_layer=pool_layer,
                 classification_layer=classification_layer
                 )
    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids=device_ids)

    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net,
                                                                                                               args)

    # Initialize or load model
    with torch.no_grad():
        if args.state_dict_dir_net != '':
            epoch = 0
            checkpoint = torch.load(args.state_dict_dir_net, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("Pretrained network loaded", flush=True)
            net.module._multiplier.requires_grad = False
            try:
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict'])
            except:
                pass
            if torch.mean(net.module._classification.weight).item() > 1.0 and torch.mean(
                    net.module._classification.weight).item() < 3.0 and torch.count_nonzero(
                    torch.relu(net.module._classification.weight - 1e-5)).float().item() > 0.8 * (
                    num_prototypes * len(
                    classes)):  # assume that the linear classification layer is not yet trained (e.g. when loading a pretrained backbone only)
                print("We assume that the classification layer is not yet trained. We re-initialize it...",
                      flush=True)
                torch.nn.init.normal_(net.module._classification.weight, mean=1.0, std=0.1)
                torch.nn.init.constant_(net.module._multiplier, val=2.)
                print("Classification layer initialized with mean",
                      torch.mean(net.module._classification.weight).item(), flush=True)
                if args.bias:
                    torch.nn.init.constant_(net.module._classification.bias, val=0.)
            else:
                if 'optimizer_classifier_state_dict' in checkpoint.keys():
                    optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])

        else:
            net.module._add_on.apply(init_weights_xavier)
            torch.nn.init.normal_(net.module._classification.weight, mean=1.0, std=0.1)
            if args.bias:
                torch.nn.init.constant_(net.module._classification.bias, val=0.)
            torch.nn.init.constant_(net.module._multiplier, val=2.)
            net.module._multiplier.requires_grad = False

            print("Classification layer initialized with mean",
                  torch.mean(net.module._classification.weight).item(), flush=True)

    for param in net.module._add_on.parameters():
        param.requires_grad = True
    for param in params_to_freeze:
        param.requires_grad = True
    for param in params_to_train:
        param.requires_grad = True
    for param in params_backbone:
        param.requires_grad = True

    net.module._classification = None
    net.module._multiplier = None

    return net.module, num_prototypes
