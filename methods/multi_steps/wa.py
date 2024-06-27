import numpy as np
import torch
from torch.nn import functional as F
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import tensor2numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
import torch.nn as nn
from argparse import ArgumentParser

EPSILON = 1e-8


# init_epoch=200
# init_lr=0.1
# init_milestones=[60,120,170]
# init_lr_decay=0.1
# init_weight_decay=0.0005


# epochs = 170
# lrate = 0.1
# milestones = [60, 100,140]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay=2e-4
# num_workers=8
# T=2

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--T', type=float, default=None, help='tempreture apply to the output logits befor softmax')
    return parser

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, gamma=4):
        super(MultiLabelFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        sigmoid_inputs = torch.sigmoid(inputs)
        pos_loss = -targets * (1 - sigmoid_inputs)**self.gamma * torch.log(sigmoid_inputs + 1e-10)*2
        neg_loss = -(1 - targets) * sigmoid_inputs**self.gamma * torch.log(1 - sigmoid_inputs + 1e-10)
        loss = pos_loss + neg_loss
        return loss.mean()
    
class WA(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._T = config.T
        self._old_network = None
        self._kd_lamda = None
        if self._incre_type != 'cil':
            raise ValueError('WA is a class incremental method!')

    def prepare_model(self, checkpoint=None):
        super().prepare_model(checkpoint)
        if self._old_network is not None:
            self._old_network.cuda()

    def after_task(self):
        super().after_task()
        self._old_network = self._network.copy().freeze()

    def incremental_train(self):
        self._kd_lamda = self._known_classes / self._total_classes
        super().incremental_train()
        if self._cur_task > 0:
            self._network.weight_align(self._total_classes-self._known_classes)

    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        correct, total = 0, 0
        model.train()
        num_classes = 0
        predsAll = torch.tensor([]).cuda()
        targetsAll = torch.tensor([]).cuda()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            logits, output_features = model(inputs)
            if num_classes==0:
                num_classes = logits.shape[-1]
            # loss = cross_entropy(logits/self._T, targets)

            task_scores = torch.sigmoid(logits).reshape(logits.shape)
            ce_loss = MultiLabelFocalLoss()(logits, targets.float())
                
            if self._old_network is None:
                loss = ce_loss
            else:
                kd_loss = self._KD_loss(logits, self._old_network(inputs)[0], self._T)
                loss = ce_loss + kd_loss  
                
                
                
            predsAll = torch.concat([predsAll,task_scores.clone()],dim=0)
            targetsAll = torch.concat([targetsAll,targets.clone()],dim=0)
            
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            total += len(targets)
        
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
    
    def _KD_loss(self, pred, soft, T):
        # pred = torch.log_softmax(pred / T, dim=1)
        # soft = torch.softmax(soft / T, dim=1)
        pred = torch.sigmoid(pred / T).reshape(pred.shape)
        soft = torch.sigmoid(soft / T).reshape(pred.shape)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]