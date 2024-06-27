import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from backbone.adapter_cl_net import CNN_Adapter_Net_CIL_V2
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, tensor2numpy
import torch.nn as nn

EPSILON = 1e-8

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
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes
        print(f'self._known_classes:{self._known_classes}')
        print(f'self._total_classes:{self._total_classes}')
        self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='train')
        self._test_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='test', mode='test')
        # self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        # self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._sampler_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='test')

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = CNN_Adapter_Net_CIL_V2(self._logger, self._config.backbone, self._config.pretrained,
                    pretrain_path=self._config.pretrain_path, layer_names=self._layer_names)
        self._network.update_fc(self._total_classes)

        self._network.freeze_FE()
        self._network = self._network.cuda()

        self._logger.info('Initializing task-{} adapter in network...'.format(self._cur_task))
        with torch.no_grad():
            self._network.eval()
            self._network.train_adapter_mode()
            # if self._cur_task<2:
            #     self._network.train_adapter_mode()
            # else:
            #     self._network.test_mode()
            self._network(torch.rand(1, 3, self._config.img_size, self._config.img_size).cuda())
        self._network.freeze_adapters(mode='old')

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        
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
        
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                self._logger.info('{} require grad!'.format(name))

    def incremental_train(self):
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        
        # train adapters
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, False)
        scheduler = self._get_scheduler(optimizer, self._config, False)
        self._is_training_adapters = True
        self._network.train_adapter_mode()
        # if self._cur_task<2:
        #     self._network.train_adapter_mode()
        # else:
        #     self._network.test_mode()
        if self._cur_task<2:
            self._network = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler,
                task_id=self._cur_task, epochs=self._epochs, note='stage1')
        else:
            max_adapter = 0
            max_adapter_auc = 0
            max_adapter_state = None
            for select in range(self._cur_task):
                self._network.select_model_init(select)
                self._network,test_auc = self._train_model(self._network, self._train_loader, self._test_loader, optimizer, scheduler,
                    task_id=self._cur_task, epochs=self._epochs, note=f'select{select}',return_auc = True)
                if test_auc > max_adapter_auc:
                    max_adapter_auc = test_auc
                    max_adapter_state = self._network.state_dict()
                    max_adapter = select
                    
            print(f'最优adapter：{max_adapter}，最优AUC：{max_adapter_auc}')
            self._network.load_state_dict(max_adapter_state)
            
        self._network.seperate_fc[self._cur_task].load_state_dict(self._network.aux_fc.state_dict())
        self._is_training_adapters = False
        

    
    def store_samples(self):
        if self._memory_bank is not None:
            self._network.store_sample_mode()
            self._memory_bank.store_samples(self._sampler_dataset, self._network)
        # prepare for eval()
        self._network.test_mode()
        
    def _train_model(self, model, train_loader, test_loader, optimizer, scheduler, task_id=None, epochs=100, note='',return_auc = False):
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
            
            record_dict['Task{}_{}Test_Acc(inner task)'.format(task_id, note)] = task_test_acc
            info = info + 'Task{}_Test_accy {:.2f}, '.format(task_id, task_test_acc)
            if self._incre_type == 'cil': # only show epoch test acc in cil, because epoch test acc is worth nothing in til
                record_dict['Task{}_{}Test_Acc'.format(task_id, note)] = test_acc
                info = info + 'Test_accy {:.2f}'.format(test_acc)

            self._logger.info(info)
            self._logger.visual_log('train', record_dict, step=epoch)
        if return_auc:
            return model,task_test_acc
        return model
    
    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        correct, total = 0, 0
        ce_losses = 0.
        losses = 0.
        th = 0.5 if task_id==0 else 0.1
        if self._is_training_adapters:
            model.new_adapters_train()
        else:
            model.eval()
            
        num_classes = 0
        correct0 = []
        total0 = []
        predsAll = torch.tensor([]).cuda()
        targetsAll = torch.tensor([]).cuda()
        for _, inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            # model forward has two mode which shoule be noticed before forward!
            logits, output_features = model(inputs)
            if num_classes==0:
                num_classes = logits[:,:-1].shape[-1]
                correct0 = [0] * num_classes
                total0 = [0] * num_classes
                
            if self._is_training_adapters:
                # if task_id==0:
                #     preds = torch.argmax(logits[:,:-1], dim=1,keepdim=True)
                #     weights = torch.ones(num_classes)  # Assuming num_classes is the total number of classes
                #     weights[1] = 80.0  # Set a higher weight for class 1

                #     # Define the CrossEntropyLoss with weights
                #     criterion = nn.CrossEntropyLoss(weight=weights.cuda())
                #     loss = criterion(logits[:,:-1], targets.reshape(-1))
                # else:
                task_scores = torch.sigmoid(logits[:,:-1]).reshape(logits[:,:-1].shape)
                loss = self.criterion(logits[:,:-1], targets.float())
                preds = torch.where(task_scores>th,1,0)
                correct += preds.eq(targets).cpu().sum()/num_classes
                predsAll = torch.concat([predsAll,task_scores.clone()],dim=0)
                targetsAll = torch.concat([targetsAll,targets.clone()],dim=0)
                # if task_id==0:
                # # Calculate accuracy for each class
                #     for pred, target in zip(preds.view(-1), targets.view(-1)):
                #         total0[target] += 1
                #         if pred == target:
                #             correct0[target] += 1
                # else:
                for i in range(num_classes):
                    # 获取第i个类别的预测和目标标签
                    class_preds = preds[:, i]
                    class_targets = targets[:, i]
                    
                    # 计算该类别的预测是否正确
                    correct1 = torch.eq(class_preds.round(), class_targets.float()).sum().item()
                    
                    total0[i] += len(class_targets)
                    # 计算该类别的精度
                    correct0[i] += correct1
                
            else:
                # stage 2
                loss = torch.tensor(0.0)
                known_class_num, total_class_num = 0, 0
                for id, cur_class_num in enumerate(self._increment_steps[:task_id+1]):
                    total_class_num += cur_class_num + 1
                    task_logits = logits[:,known_class_num:total_class_num]
                    
                    task_targets = (torch.ones(targets.shape[0], dtype=int) * cur_class_num).cuda() # class label: [0, cur_class_num]
                    task_data_idxs = torch.argwhere(torch.logical_and(targets>=known_class_num-id, targets<total_class_num-id-1)).squeeze(-1)
                    if len(task_data_idxs) > 0:
                        task_targets[task_data_idxs] = targets[task_data_idxs] - known_class_num + id
                        loss = loss + self.criterion(task_logits, task_targets)

                    if id == task_id:
                        preds = torch.argmax(logits[:,known_class_num:total_class_num], dim=1)

                    known_class_num = total_class_num
                
                ce_losses += loss.item()

                aux_targets = torch.where(targets-task_begin+1>0, targets-task_begin, task_end - task_begin)
                correct += preds.eq(aux_targets).cpu().sum()
    
            class_accuracies = [(c / t) if t != 0 else 0 for c, t in zip(correct0, total0)]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
        
        # train acc shows the performance of the model on current task
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'ce_loss', ce_losses/len(train_loader)]
        # for i in range(len(class_accuracies)):
            # print(f'class {i} accuracy:{class_accuracies[i]}')
        # print(f'\nTraining Accuracy Report:{"="*10}')
        auc_list = []
        for j in range(num_classes):
            try:
                auc = roc_auc_score(targetsAll[:,j].detach().cpu().numpy(),predsAll[:,j].detach().cpu().numpy())
                # print(f'Report calss {j}:  {auc}')
                auc_list.append(auc)
            except:
                pass
        avg = sum(auc_list)/len(auc_list)
        # print(f'Average AUC Score:  {avg}\n')
        
        return model, avg, train_loss
    
    # 2024-02-06 01:43:45,796 => Task 5, Epoch 1/1 => Loss 0.192, ce_loss 0.000, Train_accy 59.03, Task5_Test_accy 50.60, 
   
    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None):
        with torch.no_grad():
            cnn_correct, cnn_task_correct, total, task_total = 0, 0, 0, 0
            task_id_correct = 0
            cnn_pred_all, target_all = [], []
            cnn_max_scores_all = []
            features_all = []
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

                # model forward has two mode which shoule be noticed before forward!
                logits, feature_outputs = model(inputs)
                
                if num_classes==0:
                    num_classes = logits[:,:-1].shape[-1]
                    correct0 = [0] * num_classes
                    total0 = [0] * num_classes
                    
                if self._is_training_adapters:
                    # if task_id==0:
                    #     cnn_preds = torch.argmax(logits[:,:-1], dim=1,keepdim=True)
                    # else:
                    task_scores = torch.sigmoid(logits[:,:-1]).reshape(logits[:,:-1].shape)
                    cnn_preds = torch.where(task_scores>th,1,0)
                else:
                    # task_test_acc 有意义(反映当前task的预测准确率), test_acc 有意义(反映模型最终的预测结果)
                    ### predict task id based on the unknown scores ###
                    cnn_preds, cnn_max_scores, task_id_pred_correct= self.min_others_test(logits=logits, targets=targets, task_id=task_id)
                    task_id_correct += task_id_pred_correct

                if ret_pred_target: # only apply when self._is_training_adapters is False
                    cnn_pred_all.append(tensor2numpy(cnn_preds))
                    target_all.append(tensor2numpy(targets))
                    cnn_max_scores_all.append(tensor2numpy(cnn_max_scores))
                    features_all.append(tensor2numpy(feature_outputs['features']))
                else:
                    if ret_task_acc:
                        # task_data_idxs = torch.argwhere(torch.logical_and(targets>=task_begin, targets<task_end))
                        # cnn_task_correct += cnn_preds[task_data_idxs].eq(targets[task_data_idxs]).cpu().sum()
                        cnn_task_correct += cnn_preds.eq(targets).cpu().sum()/num_classes
                        task_total += len(targets)
                        
                        
                        # if task_id==0:
                        #     for pred, target in zip(cnn_preds.view(-1), targets.view(-1)):
                        #         total0[target] += 1
                        #         if pred == target:
                        #             correct0[target] += 1
                                    
                        # else:
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
            
            if not self._is_training_adapters:
                self._logger.info('Test task id predict acc (CNN) : {:.2f}'.format(np.around(task_id_correct*100 / total, decimals=2)))

            if ret_pred_target:
                cnn_pred_all = np.concatenate(cnn_pred_all)
                target_all = np.concatenate(target_all)
                cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
                features_all = np.concatenate(features_all)
                return cnn_pred_all, None, cnn_max_scores_all, None, target_all, features_all
            else:
                test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
                if ret_task_acc:
                    test_task_acc = np.around(tensor2numpy(cnn_task_correct)*100 / task_total, decimals=2)
                    # for i in range(len(class_accuracies)):
                    #     print(f'class {i} accuracy:{class_accuracies[i]}')
                    # print(f'\nTest Accuracy Report:{"="*10}')
                    auc_list = []
                    for j in range(num_classes):
                        try:
                            auc = roc_auc_score(targetsAll[:,j].cpu().numpy(),predsAll[:,j].cpu().numpy())
                            # print(f'Report calss {j}: {auc}')
                            auc_list.append(auc)
                        except:
                            pass
                    avg = sum(auc_list)/len(auc_list)
                    # print(f'Average AUC Score: {avg}\n')
                    return avg, avg
                else:
                    return avg
    
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