import numpy as np
import torch
from argparse import ArgumentParser
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import target2onehot, tensor2numpy
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score


EPSILON = 1e-8

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
    
class iCaRL(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._old_network = None
        self._T = config.T
        if self._incre_type != 'cil':
            raise ValueError('iCaRL is a class incremental method!')
    
    def prepare_model(self, checkpoint=None):
        super().prepare_model(checkpoint)
        if self._old_network is not None:
            self._old_network.cuda()
    
    def after_task(self):
        super().after_task()
        logger = self._network._logger
        self._network._logger = None
        self._old_network = self._network.copy().freeze()
        self._network._logger = logger
        
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
        
        # random shuffle soft (teacher logits)
        # b, dim = soft.shape
        # max_idx = torch.argmax(soft, dim=1)
        # mask = torch.ones_like(soft, dtype=bool)
        # mask[torch.arange(mask.shape[0]) ,max_idx] = False
        # soft[mask] = soft[mask].view(b, dim-1)[:, torch.randperm(dim-1)].view(-1)

        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]