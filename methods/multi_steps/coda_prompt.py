import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Subset

from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import tensor2numpy, count_parameters

from backbone.vit_prompts import CodaPrompt as Prompt
from backbone.vit_zoo import ViTZoo
from sklearn.metrics import roc_auc_score
EPSILON = 1e-8

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
    
def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--prompt_pool', type=int, default=None, help='size of prompt pool')
    parser.add_argument('--prompt_length', type=int, default=None, help='length of prompt')
    parser.add_argument('--ortho_weight', type=float, default=None, help='ortho penalty loss weight')
    return parser

class CODA_Prompt(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._prompt_pool = config.prompt_pool
        self._prompt_length = config.prompt_length
        self._ortho_weight = config.ortho_weight

        # if self._incre_type != 'cil':
        #     raise ValueError('DualPrompt is a class incremental method!')
    
    def prepare_model(self, checkpoint=None):
        if self._network == None:
            prompt_module = Prompt(768, self._config.nb_tasks, self._prompt_pool, self._prompt_length, self._ortho_weight)
            self._network = ViTZoo(self._logger, prompt_module=prompt_module)
        set_random(1993)
        self._network.update_fc(self._total_classes)
        
        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            if checkpoint['memory_class_means'] is not None and self._memory_bank is not None:
                self._memory_bank.set_class_means(checkpoint['memory_class_means'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        if self._config.freeze_fe:
            self._network.freeze_FE()

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()

        for name, param in self._network.named_parameters():
            if param.requires_grad:
                self._logger.info('{} require grad!'.format(name))
                
    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes

        if self._cur_task > 0 and self._memory_bank != None:
            self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), 
                    source='train', mode='train', appendent=self._memory_bank.get_memory())
        else:
            self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='train')
        
        self._test_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='test', mode='test')
        # self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(self._test_dataset, batch_size=2, shuffle=False, num_workers=self._num_workers)
        # self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._sampler_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='test')
            
    def incremental_train(self):
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        if self._cur_task==0:
            self.avg_res = []
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, self._cur_task==0)
        scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
        if self._cur_task == 0:
            epochs = self._init_epochs
        else:
            epochs = self._epochs
        self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler, task_id=self._cur_task, epochs=epochs)
        
        print(f'self.res:{self.avg_res}')
        print(f'self.avg_res:{sum(self.avg_res)/len(self.avg_res)}')      
        
    def store_samples(self):
        if self._memory_bank != None:
            self._memory_bank.store_samples(self._sampler_dataset, self._network)
    
    def _train_model(self, model, train_loader, test_loader, optimizer, scheduler, task_id=None, epochs=100, note=''):
        task_begin = sum(self._increment_steps[:task_id])
        task_end = task_begin + self._increment_steps[task_id]
        if note != '':
            note += '_'
        for epoch in range(epochs):
            model, train_acc, train_losses = self._epoch_train(model, train_loader, optimizer, scheduler,
                                task_begin=task_begin, task_end=task_end, task_id=task_id)
            # update record_dict
            record_dict = {}

            info = ('Task {}, Epoch {}/{} => '.format(task_id, epoch+1, epochs) + 
                ('{} {:.3f}, '* int(len(train_losses)/2)).format(*train_losses))
            for i in range(int(len(train_losses)/2)):
                record_dict['Task{}_{}'.format(task_id, note)+train_losses[i*2]] = train_losses[i*2+1]
            
            if train_acc is not None:
                record_dict['Task{}_{}Train_Acc'.format(task_id, note)] = train_acc
                info = info + 'Train_accy {:.2f}, '.format(train_acc)          
            with torch.no_grad():
                test_acc, task_test_acc = self._epoch_test(model, test_loader, ret_task_acc=True,
                                                    task_begin=task_begin, task_end=task_end, task_id=task_id)
            
            if epoch==epochs-1:
                self.avg_res.append(test_acc)
            record_dict['Task{}_{}Test_Acc(inner task)'.format(task_id, note)] = task_test_acc
            info = info + 'Task{}_Test_accy {:.2f}, '.format(task_id, task_test_acc)
            if self._incre_type == 'cil': # only show epoch test acc in cil, because epoch test acc is worth nothing in til
                record_dict['Task{}_{}Test_Acc'.format(task_id, note)] = test_acc
                info = info + 'Test_accy {:.2f}'.format(test_acc)

            self._logger.info(info)
            self._logger.visual_log('train', record_dict, step=epoch)
        
        return model
    
    
    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        ce_losses, prompt_losses = 0., 0.
        correct, total = 0, 0
        model.train()
        
        num_classes = 0
        predsAll = torch.tensor([]).cuda()
        targetsAll = torch.tensor([]).cuda()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, prompt_loss = model(inputs, train=True)
            
            loss = prompt_loss.sum()
            prompt_losses += loss.sum()
            
            if num_classes==0:
                num_classes = logits.shape[-1]
                
            ce_loss = MultiLabelFocalLoss()(logits, targets)
            loss += ce_loss
            ce_losses += ce_loss.item()

            preds = torch.max(logits, dim=1)[1]
            
            task_scores = torch.sigmoid(logits).reshape(logits.shape)
            predsAll = torch.concat([predsAll,task_scores.clone()],dim=0)
            targetsAll = torch.concat([targetsAll,targets.clone()],dim=0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
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
            
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Loss_ce', ce_losses/len(train_loader), 'Loss_prompt', prompt_losses/len(train_loader)]
        return model, avg, train_loss

    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None):
        cnn_correct, cnn_task_correct, total, task_total = 0, 0, 0, 0
        cnn_pred_all, target_all = [], []
        cnn_max_scores_all = []
        model.eval()
        
        th = 0.5 if task_id==0 else 0.1
        num_classes = 0
        # Calculate accuracy for each class
        correct0 = [0]
        total0 = [0]
        
        predsAll = torch.tensor([]).cuda()
        targetsAll = torch.tensor([]).cuda()
        
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = model(inputs)
            cnn_max_scores, cnn_preds = torch.max(torch.softmax(logits, dim=-1), dim=-1)
            
            if num_classes==0:
                num_classes = task_end-task_begin
                correct0 = [0] * num_classes
                total0 = [0] * num_classes
                       
            if ret_pred_target:
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
                cnn_max_scores_all.append(tensor2numpy(cnn_max_scores))
            else:
                if ret_task_acc:
                    # task_data_idxs = torch.argwhere(torch.logical_and(targets>=task_begin, targets<task_end))
                    # cnn_task_correct += cnn_preds[task_data_idxs].eq(targets[task_data_idxs]).cpu().sum()
                    # task_total += len(task_data_idxs)
                        cnn_task_correct += cnn_preds.eq(targets).cpu().sum()/num_classes
                        task_total += len(targets)
                        
                        
                        for i in range(num_classes):
                            # 获取第i个类别的预测和目标标签
                            class_preds = cnn_preds.reshape(targets.shape)[:, i]
                            class_targets = targets[:, i]
                            
                            # 计算该类别的预测是否正确
                            correctSingle = torch.eq(class_preds.round(), class_targets.float()).sum().item()
                            
                            total0[i] += len(class_targets)
                            # 计算该类别的精度
                            correct0[i] += correctSingle
                            
                task_scores = torch.sigmoid(logits).reshape(logits.shape)
                predsAll = torch.concat([predsAll,task_scores.clone()],dim=0)
                targetsAll = torch.concat([targetsAll,targets.clone()],dim=0)
                    
                cnn_correct += cnn_preds.eq(targets).cpu().sum()
                    
                total += len(targets)
        
        if ret_pred_target:
            cnn_pred_all = np.concatenate(cnn_pred_all)
            cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
            target_all = np.concatenate(target_all)
            return cnn_pred_all, None, cnn_max_scores_all, None, target_all, None
        else:
            test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
            if ret_task_acc:
                # test_task_acc = np.around(tensor2numpy(cnn_task_correct)*100 / task_total, decimals=2)
                
                auc_list = []
                for j in range(num_classes):
                    try:
                        auc = roc_auc_score(targetsAll[:,j].cpu().numpy(),predsAll[:,j].cpu().numpy())
                        auc_list.append(auc)
                    except:
                        pass
                avg = sum(auc_list)/len(auc_list)
                # print(f'Average AUC Score: {avg}\n')
                
                
                return avg, avg
            else:
                return test_acc
            
