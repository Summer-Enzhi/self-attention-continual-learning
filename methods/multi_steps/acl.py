import copy
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

import random
import dill
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from backbone.adapter_cl_net import CNN_Adapter_Net_CIL_V2
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, tensor2numpy
import torch.nn as nn
from utils.logger import MyLogger
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
    parser.add_argument('--layer_names', nargs='+', type=str, default=None, help='layers to apply prompt, e.t. [layer1, layer2]')
    parser.add_argument('--epochs_finetune', type=int, default=None, help='balance finetune epochs')
    parser.add_argument('--lrate_finetune', type=float, default=None, help='balance finetune learning rate')
    parser.add_argument('--milestones_finetune', nargs='+', type=int, default=None, help='for multi step learning rate decay scheduler')
    return parser 


class ACL(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._layer_names = config.layer_names

        self._epochs_finetune = config.epochs_finetune
        self._lrate_finetune = config.lrate_finetune
        self._milestones_finetune = config.milestones_finetune

        self._is_training_adapters = False
        self.criterion = MultiLabelFocalLoss()
        # if self._incre_type != 'cil':
        #     raise ValueError('Continual_Adapter is a class incremental method!')

        self._class_means_list = []
        
  
    def prepare_task_data(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes
        print(f'self._known_classes:{self._known_classes}')
        print(f'self._total_classes:{self._total_classes}')
        self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='train')
        # self._val_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='test', mode='test')
        # print(f'_val_dataset:{len(self._val_dataset)}')
        # self._test_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='valid', mode='valid')
        
        # skin8
        self._val_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='valid', mode='valid', increment_steps = self._increment_steps)
        # print(f'_val_dataset:{len(self._val_dataset)}')
        self._test_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='test', mode='test', increment_steps = self._increment_steps)
        # self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Valid dataset size: {}'.format(len(self._val_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._val_loader = DataLoader(self._val_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        # self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._sampler_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='test')

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = CNN_Adapter_Net_CIL_V2(self._logger, self._config.backbone, self._config.pretrained,
                    pretrain_path=self._config.pretrain_path, layer_names=self._layer_names)
        set_random(1993)
        self._network.update_fc(self._total_classes)

        self._network.freeze_FE()
        self._network = self._network.cuda()

        self._logger.info('Initializing task-{} adapter in network...'.format(self._cur_task))
        with torch.no_grad():
            self._network.eval()
            self._network.train_adapter_mode()
            self._network(torch.rand(1, 3, self._config.img_size, self._config.img_size).cuda())
        self._network.freeze_adapters(mode='old')

        # if checkpoint is not None:
        #     self._network.load_state_dict(checkpoint['state_dict'])
        #     self._logger.info("Loaded checkpoint model's state_dict !")
        
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        all_params, all_trainable_params = 0, 0
        for layer_id in self._config.layer_names:
            adapter_id = layer_id.replace('.', '_')+'_adapters'
            if hasattr(self._network, adapter_id):
                adapter_module = getattr(self._network, adapter_id)
                
                layer_params = count_parameters(adapter_module)
                layer_trainable_params = count_parameters(adapter_module, True)
                self._logger.info('{} params: {} , trainable params: {}'.format(adapter_id,
                    layer_params, layer_trainable_params))
                
                all_params += layer_params
                all_trainable_params += layer_trainable_params
        self._logger.info('all adapters params: {} , trainable params: {}'.format(all_params, all_trainable_params))
        self._logger.info('seperate fc params: {} , trainable params: {}'.format(count_parameters(self._network.seperate_fc), count_parameters(self._network.seperate_fc, True)))
        self._logger.info('aux fc params: {} , trainable params: {}'.format(count_parameters(self._network.aux_fc), count_parameters(self._network.aux_fc, True)))
        
        # for name, param in self._network.named_parameters():
        #     if param.requires_grad:
        #         self._logger.info('{} require grad!'.format(name))

    def incremental_train(self):
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        
        self._is_training_adapters = True
        if self._cur_task==0:
            self.avg_res = []
            self.avg_res2 = []
            self.qk = [[],[]]
            self.openset_res = []
            self.openset_f1 = [] 
        self._network.train_adapter_mode()
        self._network.activate_new_adapters()
        paras = list(filter(lambda p: p.requires_grad, self._network.parameters()))
        
        optimizer = self._get_optimizer(paras, self._config, False)
        scheduler = self._get_scheduler(optimizer, self._config, False)
        
        
        self._network = self._train_model(self._network, self._train_loader, self._val_loader,self._test_loader, optimizer, scheduler,
            task_id=self._cur_task, epochs=self._epochs, note='stage1')
        self._is_training_adapters = False
        print(f'self.res:{self.avg_res}')
        print(f'self.avg_res:{sum(self.avg_res)/len(self.avg_res)}')         
        
        # if self._cur_task > 1:
        #     old_model = self._network
        #     old_model.eval()
        #     old_model.test_mode()
        #     self._is_training_adapters = True #X
        #     num = len(old_model.task_sizes)   #X
        #     new_task_size = old_model.task_sizes[-1]  # X
        #     old_model.aux_fc = CrossAttentionModule(self._network._feature_dim, num ,new_task_size+1).cuda()
            
        #     optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, old_model.parameters()), lr=0.0001,weight_decay=0.00)
        #     scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer=optimizer2, milestones=[50,80], gamma=0.5)
            
        #     _,test_auc = self._train_model(old_model, self._train_loader, self._val_loader, self._test_loader, optimizer2, scheduler2,
        #         task_id=self._cur_task, epochs=self._epochs, note=f'stage2',return_auc = True, stage = 2)   
        #     self._is_training_adapters = False
        #     print(f'self.res2:{self.avg_res2}')
        #     print(f'self.avg_res2:{sum(self.avg_res2)/len(self.avg_res2)}')         
        #     self._logger.info('aux fc params: {} , trainable params: {}'.format(count_parameters(self._network.aux_fc), count_parameters(self._network.aux_fc, True)))
            
            ## Stage 2  Fusion Part  End ##
            
        ## Openset test
        logger = self._network._logger
        self._network._logger = None
        # if self._cur_task > 1:
        #     torch.save(old_model, f'{self._logger.log_file_name}/model_task_{self._cur_task}.pth')
        # else:
        #     torch.save(self._network, f'{self._logger.log_file_name}/model_task_{self._cur_task}.pth')
        torch.save(self._network, f'{self._logger.log_file_name}/model_task_{self._cur_task}.pth')
            
        self._network._logger = logger
        self.test_all_tasks(self._cur_task+1)
    
    def test_all_tasks(self,num_tasks):
        self.openset_res.append([])
        self.openset_f1.append([])
        for task in range(num_tasks):
            model = torch.load(f'{self._logger.log_file_name}/model_task_{task}.pth').cuda()
            
            with torch.no_grad():
                # _known_classes控制 测试哪个label
                # openset = _total_classes 控制 sofar 数量
                sofar = sum(self._increment_steps[:task+1])
                openset_test_dataset = self.data_manager.get_dataset(indices=np.arange(sofar-1, sofar), source='test', mode='test', increment_steps = self._increment_steps, openset = sum(self._increment_steps[:num_tasks]))
                openset_test_loader = DataLoader(openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
                # stage = 1 if task < 2 else 2
                stage = 1
                test_acc, f1 = self._epoch_test(model, openset_test_loader, ret_task_acc=True,stage = stage)
            self.openset_res[-1].append(test_acc)
            self.openset_f1[-1].append(f1)
        del model
        
        print(f'self.openset_auc:')
        for index,i in enumerate(self.openset_res):
            print(f'Task {index}: {i}')
        
        avg = [sum(i)/len(i) for i in self.openset_res]
        print(f'Avg_auc:{avg}')
        
        print(f'self.openset_f1:')
        for index,i in enumerate(self.openset_f1):
            print(f'Task {index}: {i}')
        
        avgf1 = [sum(i)/len(i) for i in self.openset_f1]
        print(f'Avg_f1:{avgf1}')
        
    def _train_model(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, task_id=None, epochs=100, note='', return_auc=False,  stage = 1):
        task_begin = sum(self._increment_steps[:task_id])
        task_end = task_begin + self._increment_steps[task_id]
        if note != '':
            note += '_'
        best_val_acc = 0
        early_stopping_counter = 0
        self._early_stopping_patience = 10
        best_model_state_dict = None
        best_epoch = 0
        for epoch in range(epochs):
            model, train_acc, train_losses = self._epoch_train(model, train_loader, optimizer, scheduler,
                                                                task_begin=task_begin, task_end=task_end, task_id=task_id, epoch=epoch, stage = stage)
            # Update record_dict
            record_dict = {}

            info = ('Task {}, Epoch {}/{} => '.format(task_id, epoch + 1, epochs) +
                    ('{} {:.3f}, ' * int(len(train_losses) / 2)).format(*train_losses))
            for i in range(int(len(train_losses) / 2)):
                record_dict['Task{}_{}'.format(task_id, note) + train_losses[i * 2]] = train_losses[i * 2 + 1]

            # if train_acc is not None:
            #     record_dict['Task{}_{}Train_Acc'.format(task_id, note)] = train_acc
            #     info = info + 'Train_accy {:.2f}, '.format(train_acc)

            # Validation phase
            with torch.no_grad():
                val_acc, val_test_acc = self._epoch_test(model, val_loader, ret_task_acc=True,
                                                    task_begin=task_begin, task_end=task_end, task_id=task_id,
                                                    epoch=epoch, stage = stage)

            # Update best model and early stopping counter
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_ = model.module if isinstance(model, nn.DataParallel) else model
                best_model_state_dict = model_.state_dict()
                best_epoch = epoch
                torch.save(best_model_state_dict, f'{self._logger.log_file_name}/best_checkpoint.pkl')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1


            # Early stopping check
            # if early_stopping_counter >= self._early_stopping_patience:
            #     self._logger.info(f"Early stopping at epoch {epoch}.")
            #     break

            # Test phase
            with torch.no_grad():
                test_acc, task_test_acc = self._epoch_test(model, test_loader, ret_task_acc=True,
                                                        task_begin=task_begin, task_end=task_end, task_id=task_id,
                                                        epoch=epoch, stage = stage)
            if epoch != epochs-1:
                info += 'val_acc {:.3f}, '.format(val_acc)
                info = info + 'Test_accy {:.3f}, '.format(test_acc)
                record_dict['Task{}_{}Test_Acc(inner task)'.format(task_id, note)] = test_acc
                
                if self._incre_type == 'cil':
                    record_dict['Task{}_{}Test_Acc'.format(task_id, note)] = test_acc
                    info = info + 'Test_accy {:.3f}'.format(test_acc)

                self._logger.info(info)
                self._logger.visual_log('train', record_dict, step=epoch)

        # Load best model state_dict
        if best_model_state_dict is not None:
            model.load_state_dict(torch.load(f'{self._logger.log_file_name}/best_checkpoint.pkl'))
            
            with torch.no_grad():
                if stage == 2:
                    qk = [] #qk
                    val_acc, val_test_acc = self._epoch_test(model, val_loader, ret_task_acc=True,
                                                        task_begin=task_begin, task_end=task_end, task_id=task_id,
                                                        epoch=best_epoch, stage = stage,need_qk = False) # True
                else:
                    val_acc, val_test_acc = self._epoch_test(model, val_loader, ret_task_acc=True,
                                                        task_begin=task_begin, task_end=task_end, task_id=task_id,
                                                        epoch=best_epoch, stage = stage)
                    
                test_acc, task_test_acc = self._epoch_test(model, test_loader, ret_task_acc=True,
                                                        task_begin=task_begin, task_end=task_end, task_id=task_id,
                                                        epoch=best_epoch, stage = stage)
            info += 'val_acc {:.3f}, '.format(val_acc)
            info += 'Test_accy {:.3f}, '.format(test_acc)
            record_dict['Task{}_{}Test_Acc(inner task)'.format(task_id, note)] = test_acc

            
            self._logger.info(info)
            self._logger.visual_log('train', record_dict, step=epoch)
            if stage==1:
                self.avg_res.append(test_acc)
            else:
                self.avg_res2.append(test_acc)
                self.qk.append(qk)
                
            if task_id==0 or task_id==1:
                self.avg_res2.append(test_acc)
                
        if return_auc:
            return model, task_test_acc
        return model


    
    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None, epoch = 0, stage = 1):
        losses = 0.
        if self._is_training_adapters:
            model.new_adapters_train()
        else:
            model.eval()
            
        num_classes = 0
        predsAll = torch.tensor([]).cuda()
        targetsAll = torch.tensor([]).cuda()
        for _, inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            # model forward has two mode which shoule be noticed before forward!
            if stage==1:
                logits, output_features = model(inputs)
            else:
                model.new_adapters_train()
                model.aux_fc.train()
                _, output_features = model(inputs)
                logits = model.aux_fc(output_features['features'])
            if num_classes==0:
                num_classes = logits[:,:-1].shape[-1]

            task_scores = torch.sigmoid(logits[:,:-1]).reshape(logits[:,:-1].shape)
            loss = MultiLabelFocalLoss()(logits[:,:-1], targets.float())
                
            predsAll = torch.concat([predsAll,task_scores.clone()],dim=0)
            targetsAll = torch.concat([targetsAll,targets.clone()],dim=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
        
        if scheduler != None:
            scheduler.step()
        

        train_loss = ['Loss', losses/len(train_loader)]
        auc_list = []
        for j in range(num_classes):
            try:
                auc = roc_auc_score(targetsAll[:,j].detach().cpu().numpy(),predsAll[:,j].detach().cpu().numpy())
                auc_list.append(auc)
            except:
                pass
        avg = sum(auc_list)/len(auc_list)
        # print(f'Average AUC Score:  {avg}\n')
        
        return model, avg, train_loss
    
    # 2024-02-06 01:43:45,796 => Task 5, Epoch 1/1 => Loss 0.192, ce_loss 0.000, Train_accy 59.03, Task5_Test_accy 50.60, 
   
    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None, stage = 1,need_qk = False,epoch=0):
        with torch.no_grad():
            cnn_correct, cnn_task_correct, total, task_total = 0, 0, 0, 0
            task_id_correct = 0
            cnn_pred_all, target_all = [], []
            cnn_max_scores_all = []
            features_all = []
            
            model.eval()
            if stage==2:
                model.aux_fc.eval()
            
            th = 0.5
            num_classes = 0
            # Calculate accuracy for each class
            correct0 = [0]
            total0 = [0]
            
            predsAll = torch.tensor([]).cuda()
            targetsAll = torch.tensor([]).cuda()
            qk_all = []
            for _, inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()

                # model forward has two mode which shoule be noticed before forward!
                if stage==1:
                    logits, _ = model(inputs)
                else:
                    _, output_features = model(inputs)
                    if need_qk:
                        logits,qk = model.aux_fc(output_features['features'],need_qk = need_qk)
                        qk_all.append(qk)
                    else:
                        logits = model.aux_fc(output_features['features'])
                         
                if num_classes==0:
                    num_classes = logits[:,:-1].shape[-1]
                    correct0 = [0] * num_classes
                    total0 = [0] * num_classes
                    
                task_scores = torch.sigmoid(logits[:,:-1]).reshape(logits[:,:-1].shape)
                cnn_preds = torch.where(task_scores>th,1,0)

                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
                if ret_pred_target: # only apply when self._is_training_adapters is False
                    cnn_max_scores_all.append(tensor2numpy(''))
                    # features_all.append(tensor2numpy(feature_outputs['features']))
                else:
                    if ret_task_acc:
                        # task_data_idxs = torch.argwhere(torch.logical_and(targets>=task_begin, targets<task_end))
                        # cnn_task_correct += cnn_preds[task_data_idxs].eq(targets[task_data_idxs]).cpu().sum()
                        cnn_task_correct += cnn_preds.eq(targets).cpu().sum()/num_classes
                        task_total += len(targets)
                        
                        for i in range(num_classes):
                            # 获取第i个类别的预测和目标标签
                            class_preds = cnn_preds[:, i]
                            class_targets = targets[:, i]
                            
                            # 计算该类别的预测是否正确
                            correctSingle = torch.eq(class_preds.round(), class_targets.float()).sum().item()
                            
                            total0[i] += len(class_targets)
                            # 计算该类别的精度
                            correct0[i] += correctSingle
                            
                    predsAll = torch.concat([predsAll,task_scores.clone()],dim=0)
                    targetsAll = torch.concat([targetsAll,targets.clone()],dim=0)
                        
                    cnn_correct += cnn_preds.eq(targets).cpu().sum()
                
                # for print out task id predict acc
                total += len(targets)
            class_accuracies = [(c / t) if t != 0 else 0 for c, t in zip(correct0, total0)]
            

            if ret_pred_target:
                cnn_pred_all = np.concatenate(cnn_pred_all)
                target_all = np.concatenate(target_all)
                cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
                features_all = np.concatenate(features_all)
                return cnn_pred_all, None, cnn_max_scores_all, None, target_all, features_all
            else:
                if ret_task_acc:
                    auc_list = []
                    for j in range(num_classes):
                        try:
                            auc = roc_auc_score(targetsAll[:,j].cpu().numpy(),predsAll[:,j].cpu().numpy())
                            auc_list.append(auc)
                        except:
                            pass
                    avg = sum(auc_list)/len(auc_list)
                    # print(f'Average AUC Score: {avg}\n')
                if need_qk:
                    qk_all = list(np.array(qk_all).mean(axis=0))
                    return avg, avg, qk_all
                cnn_pred_all = np.concatenate(cnn_pred_all)
                target_all = np.concatenate(target_all)
                f1 = f1_score(target_all, cnn_pred_all)
                return avg, f1
    
    def min_others_test(self, logits, targets, task_id):
         ### predict task id based on the unknown scores ###
        unknown_scores = []
        known_scores = []
        known_class_num, total_class_num = 0, 0
        task_id_targets = torch.zeros(targets.shape[0], dtype=int).cuda()
        for id, cur_class_num in enumerate(self._increment_steps[:task_id+1]):
            total_class_num += cur_class_num + 1
            task_logits = logits[:,known_class_num:total_class_num]
            task_scores = torch.softmax(task_logits, dim=1)

            unknown_scores.append(task_scores[:, -1])
            
            known_task_scores = torch.zeros((targets.shape[0], max(self._increment_steps))).cuda()
            known_task_scores[:, :(task_scores.shape[1]-1)] = task_scores[:, :-1]
            known_scores.append(known_task_scores)

            # generate task_id_targets
            task_data_idxs = torch.argwhere(torch.logical_and(targets>=known_class_num-id, targets<total_class_num-id-1)).squeeze(-1)
            if len(task_data_idxs) > 0:
                task_id_targets[task_data_idxs] = id

            known_class_num = total_class_num

        known_scores = torch.stack(known_scores, dim=0) # task num, b, max(task_sizes)
        unknown_scores = torch.stack(unknown_scores, dim=-1) # b, task num
    
        min_scores, task_id_predict = torch.min(unknown_scores, dim=-1)
        cnn_max_scores = 1 - min_scores
        ###
        
        ### predict class based on task id and known scores ###
        cnn_preds = torch.zeros(targets.shape[0], dtype=int).cuda()
        known_class_num, total_class_num = 0, 0
        for id, cur_class_num in enumerate(self._increment_steps[:task_id+1]):
            total_class_num += cur_class_num # do not have others category !
            task_logits_idxs = torch.argwhere(task_id_predict==id).squeeze(-1)
            if len(task_logits_idxs) > 0:
                cnn_preds[task_logits_idxs] = torch.argmax(known_scores[id, task_logits_idxs], dim=1) + known_class_num
                
            known_class_num = total_class_num
        
        task_id_correct = task_id_predict.eq(task_id_targets).cpu().sum()

        return cnn_preds, cnn_max_scores, task_id_correct
    

# class CrossAttentionModule(nn.Module):
#     def __init__(self, input_dim=512, num_adapters=2, out_dim=2, num_heads=2, dropout=0.1):
#         super(CrossAttentionModule, self).__init__()
#         self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout,batch_first=True)
#         self.linear1 = nn.Linear(input_dim, input_dim)
#         self.linear2 = nn.Linear(input_dim, out_dim)
#         self.norm1 = nn.LayerNorm(input_dim)
#         self.norm2 = nn.LayerNorm(out_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, adapters, epochs=0, need_qk=False):
        
#         x = torch.stack(adapters[1:]).transpose(0, 1) # B x N x input_dim 
#         # Extract q, k, v
#         q = x[:, -1:, :]  # Query: Last token
#         k_v = x[:, :-1, :]  # Keys and Values: All tokens except the last one

#         # Multihead cross attention
#         attn_output, attn_weights = self.multihead_attn(q, k_v, k_v)

#         # Residual connection and layer normalization
#         x = self.norm1(q + attn_output)

#         # Feed forward layer
#         x = x.squeeze(1)  # Remove the added dimension from query
#         x_ff = self.linear2(torch.relu(self.linear1(x)))
#         x_ff = self.dropout(x_ff)

#         if need_qk:
#             return x_ff, attn_weights
#         else:
            # return x_ff



class CrossAttentionModule(nn.Module):
    def __init__(self, input_dim=512, num_adapters=2, out_dim = 2, num_heads = 2, dropout=0.5):
        super(CrossAttentionModule, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(input_dim * (num_adapters-1), input_dim)
        self.linear2 = nn.Linear(input_dim, out_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        # self.norm2 = nn.LayerNorm(out_dim)
        # self.norm1 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adapters, epochs = 0, need_qk = False):
        # Multihead self attention
        x = torch.stack(adapters[1:]).transpose(0, 1) # B x N x input_dim 
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        attn_output = self.dropout(attn_output)
        # Residual connection and layer normalization
        # x = x + attn_output
        # x = self.norm1(x + attn_output)
        x = self.norm1(attn_output)
        # x = attn_output
        
        # Feed forward layer
        x = x.reshape(x.shape[0],-1)
        x_ff = self.linear2(torch.relu(self.linear1(x)))
        x_ff = self.dropout(x_ff)
        if need_qk:
            return x_ff, attn_weights
        else:
            return x_ff


    