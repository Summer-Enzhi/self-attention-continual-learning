import math

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from backbone.inc_net import CosineIncrementalNet
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, tensor2numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score

# epochs = 160
# lrate = 0.1
# ft_epochs = 20
# ft_lrate = 0.005
# batch_size = 128
# lambda_c_base = 5
# lambda_f_base = 1
# nb_proxy = 10
# weight_decay = 5e-4
# num_workers = 4

'''
Distillation losses: POD-flat (lambda_f=1) + POD-spatial (lambda_c=5)
NME results are shown.
The reproduced results are not in line with the reported results.
Maybe I missed something...
+--------------------+--------------------+--------------------+--------------------+
|     Classifier     |       Steps        |    Reported (%)    |   Reproduced (%)   |
+--------------------+--------------------+--------------------+--------------------+
|    Cosine (k=1)    |         50         |       56.69        |       55.49        |
+--------------------+--------------------+--------------------+--------------------+
|    LSC-CE (k=10)   |         50         |       59.86        |       55.69        |
+--------------------+--------------------+--------------------+--------------------+
|   LSC-NCA (k=10)   |         50         |       61.40        |       56.50        |
+--------------------+--------------------+--------------------+--------------------+
|    LSC-CE (k=10)   |         25         |       -----        |       59.16        |
+--------------------+--------------------+--------------------+--------------------+
|   LSC-NCA (k=10)   |         25         |       62.71        |       59.79        |
+--------------------+--------------------+--------------------+--------------------+
|    LSC-CE (k=10)   |         10         |       -----        |       62.59        |
+--------------------+--------------------+--------------------+--------------------+
|   LSC-NCA (k=10)   |         10         |       64.03        |       62.81        |
+--------------------+--------------------+--------------------+--------------------+
|    LSC-CE (k=10)   |         5          |       -----        |       64.16        |
+--------------------+--------------------+--------------------+--------------------+
|   LSC-NCA (k=10)   |         5          |       64.48        |       64.37        |
+--------------------+--------------------+--------------------+--------------------+
'''

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--layer_names', nargs='+', type=str, default=None, help='layers to apply prompt, e.t. [layer1, layer2]')
    parser.add_argument('--lambda_c_base', type=float, default=None, help='lambda_c_base for podnet') # podnet
    parser.add_argument('--lambda_f_base', type=float, default=None, help='lambda_f_base for podnet') # podnet
    parser.add_argument('--nb_proxy', type=int, default=None, help='nb_proxy for podnet') # podnet
    parser.add_argument('--epochs_finetune', type=int, default=None, help='balance finetune epochs')
    parser.add_argument('--lrate_finetune', type=float, default=None, help='balance finetune learning rate')
    parser.add_argument('--milestones_finetune', nargs='+', type=int, default=None, help='for multi step learning rate decay scheduler')
    return parser


class PODNet(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._lambda_c_base = config.lambda_c_base
        self._lambda_f_base = config.lambda_f_base
        self._nb_proxy = config.nb_proxy
        self._epochs_finetune = config.epochs_finetune
        self._lrate_finetune = config.lrate_finetune
        self._layer_names = config.layer_names
        self.factor = 0
        self._old_network = None
        if self._incre_type != 'cil':
            raise ValueError('PODNet is a class incremental method!')

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = CosineIncrementalNet(self._logger, self._config.backbone, self._config.pretrained, 
                        self._config.pretrain_path, self._layer_names, self._nb_proxy)
        self._network.update_fc(self._total_classes, self._cur_task)

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            if checkpoint['memory_class_means'] is not None and self._memory_bank is not None:
                self._memory_bank.set_class_means(checkpoint['memory_class_means'])
            self._logger.info("Loaded checkpoint model's state_dict !")

        for name, param in self._network.fc.named_parameters():
            if 'fc1' in name:
                param.requires_grad = False
                self._logger.info('{} requires_grad=False'.format(name))
        self._logger.info('Freezing SplitCosineLinear.fc1 ...')
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()
        if self._old_network != None:
            self._old_network = self._old_network.cuda()

        if self._cur_task == 0:
            self.factor = 0
        else:
            self.factor = math.sqrt(self._total_classes / (self._total_classes - self._known_classes))
        self._logger.info('Adaptive factor: {}'.format(self.factor))

    def after_task(self):
        super().after_task()
        logger = self._network._logger
        self._network._logger = None
        self._old_network = self._network.copy().freeze()
        self._network._logger = logger

    def incremental_train(self):
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        if self._cur_task == 0:
            epochs = self._init_epochs
        else:
            epochs = self._epochs
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, self._cur_task==0)
        scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
        self._network = self._train_model(self._network, self._train_loader,self._val_loader, self._test_loader, optimizer, scheduler, task_id=self._cur_task, epochs=epochs)
        
        # if self._cur_task > 0:
        #     self._logger.info('Finetune the network (classifier part) with the balanced dataset!')
        #     finetune_train_dataset = self._memory_bank.get_unified_sample_dataset(self._train_dataset, self._network)
        #     finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=self._batch_size,
        #                                     shuffle=True, num_workers=self._num_workers)
        #     self._logger.info('The size of finetune dataset: {}'.format(len(finetune_train_dataset)))
        #     ft_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), momentum=0.9, lr=self._lrate_finetune, weight_decay=self._config.weight_decay)
        #     ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=ft_optimizer, T_max=self._epochs_finetune)
        #     self._network = self._train_model(self._network, finetune_train_loader, self._test_loader, ft_optimizer, ft_scheduler, task_id=self._cur_task, epochs=self._epochs_finetune)


    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        losses_lsc, losses_spatial, losses_flat = 0., 0., 0.
        correct, total = 0, 0
        model.train()
        num_classes = 0
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, feature_outputs = model(inputs)
            task_scores = torch.sigmoid(logits).reshape(logits.shape)
            
            loss_lsc = nca(logits, targets)
            losses_lsc += loss_lsc.item()
            if self._old_network == None:
                loss = loss_lsc
            else:
                old_logits, old_feature_outputs = self._old_network(inputs)
                # Less forgetting loss (similar to lucir)
                loss_flat = F.cosine_embedding_loss(feature_outputs['features'], old_feature_outputs['features'].detach(),
                                torch.ones(inputs.shape[0]).cuda()) * self.factor * self._lambda_c_base
                losses_flat += loss_flat.item()
                # spacial loss
                fmaps, old_fmaps = [], []
                for layer_name in self._layer_names:
                    fmaps.append(feature_outputs[layer_name])
                    old_fmaps.append(old_feature_outputs[layer_name].detach())
                loss_spacial = pod_spatial_loss(fmaps, old_fmaps) * self.factor * self._lambda_c_base
                losses_spatial += loss_spacial.item()
                
                loss = loss_lsc + loss_flat + loss_spacial

            predsAll = torch.concat([predsAll,task_scores.clone()],dim=0)
            targetsAll = torch.concat([targetsAll,targets.clone()],dim=0)
            
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            total += len(targets)
            
        if num_classes==0:
            num_classes = logits.shape[-1]
            
        if scheduler != None:
            scheduler.step()
        auc_list = []
        for j in range(num_classes):
            try:
                auc = roc_auc_score(targetsAll[:,j].detach().cpu().numpy(),predsAll[:,j].detach().cpu().numpy())
                auc_list.append(auc)
            except:
                pass
        avg = sum(auc_list)/len(auc_list)
        
        train_loss = ['Loss', losses/len(train_loader),]
        return model, avg, train_loss


def pod_spatial_loss(old_fmaps, fmaps, normalize=True):
    '''
    a, b: list of [bs, c, w, h]
    '''
    loss = torch.tensor(0.).to(fmaps[0].device)
    for i, (a, b) in enumerate(zip(old_fmaps, fmaps)):
        assert a.shape == b.shape, 'Shape error'

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        a_h = a.sum(dim=3).view(a.shape[0], -1)  # [bs, c*w]
        b_h = b.sum(dim=3).view(b.shape[0], -1)  # [bs, c*w]
        a_w = a.sum(dim=2).view(a.shape[0], -1)  # [bs, c*h]
        b_w = b.sum(dim=2).view(b.shape[0], -1)  # [bs, c*h]

        a = torch.cat([a_h, a_w], dim=-1)
        b = torch.cat([b_h, b_w], dim=-1)

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(fmaps)


def nca(
    similarities,
    targets,
    class_weights=None,
    focal_gamma=None,
    scale=1.0,
    margin=0.6,
    exclude_pos_denominator=True,
    hinge_proxynca=False,
    memory_flags=None,
):
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margin)

    if exclude_pos_denominator:  
        similarities = similarities - similarities.max(1)[0].view(-1, 1)  

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return F.cross_entropy(similarities, targets, weight=class_weights, reduction="mean")
