import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from backbone.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.toolkit import accuracy, mean_class_recall, count_parameters, cal_ece, tensor2numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
import torch.nn as nn

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
'''
新方法命名规则: 
python文件(方法名小写) 
类名(方法名中词语字母大写)
'''

# base is finetune with or without memory_bank
class Finetune_normal(BaseLearner):

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
            self._network = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path)
        self._network.update_fc(self._total_classes)
        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        if self._config.freeze_fe:
            self._network.freeze_FE()
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()
        
    def incremental_train(self):
        if self._cur_task==0:
            self.avg_res = []
            self.openset_res = []
        if self._gpu_num > 1:
            self._network = nn.DataParallel(self._network, list(range(self._gpu_num)))
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config)
        scheduler = self._get_scheduler(optimizer, self._config)
        self._network = self._train_model(model=self._network, train_loader=self._train_loader, test_loader=self._test_loader, 
                optimizer=optimizer, scheduler=scheduler, epochs=self._epochs, valid_loader=self._valid_loader)
        if self._gpu_num > 1:
            self._network = self._network.module
        
        self.test_all_tasks(self._network,self._cur_task+1)
        
    def test_all_tasks(self,model,num_tasks):
        self.openset_res.append([])
        for task in range(num_tasks):
            with torch.no_grad():
                # _known_classes控制 测试哪个label
                # openset = _total_classes 控制 sofar 数量
                sofar = sum(self._increment_steps[:task+1])
                openset_test_dataset = self.data_manager.get_dataset(indices=np.arange(sofar-1, sofar), source='test', mode='test', increment_steps = self._increment_steps, openset = sum(self._increment_steps[:num_tasks]))
                openset_test_loader = DataLoader(openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
                test_acc, task_test_acc = self._epoch_test(model, openset_test_loader, ret_task_acc=True)
            self.openset_res[-1].append(test_acc)
        
        print(f'self.openset_res:')
        for index,i in enumerate(self.openset_res):
            print(f'Task {index}: {i}')
        
        avg = [sum(i)/len(i) for i in self.openset_res]
        avg_avg = sum(avg)/len(avg)
        print(f'Avg:{avg}')
        print(f'avg_avg:{avg_avg}')
        
    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            logits, feature_outputs = model(inputs)
            loss = MultiLabelFocalLoss()(logits, targets)
            preds = torch.max(logits, dim=1)[1]
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader)]
        return model, train_acc, train_loss

    # def _epoch_test(self, model, test_loader, ret_pred_target=False):
    #     cnn_correct, total = 0, 0
    #     cnn_pred_all, target_all, features_all = [], [], []
    #     cnn_max_scores_all = []
    #     model.eval()
    #     for _, inputs, targets in test_loader:
    #         inputs, targets = inputs.cuda(), targets.cuda()
    #         outputs, feature_outputs = model(inputs)
    #         cnn_max_scores, cnn_preds = torch.max(torch.softmax(outputs, dim=-1), dim=-1)
            
    #         if ret_pred_target:
    #             cnn_pred_all.append(tensor2numpy(cnn_preds))
    #             target_all.append(tensor2numpy(targets))
    #             features_all.append(tensor2numpy(feature_outputs['features']))
    #             cnn_max_scores_all.append(tensor2numpy(cnn_max_scores))
    #         else:
    #             cnn_correct += cnn_preds.eq(targets).cpu().sum()
    #             total += len(targets)
        
    #     if ret_pred_target:
    #         cnn_pred_all = np.concatenate(cnn_pred_all)
    #         target_all = np.concatenate(target_all)
    #         features_all = np.concatenate(features_all)
    #         cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
    #         return cnn_pred_all, cnn_max_scores_all, target_all, features_all
    #     else:
    #         test_acc = np.around(tensor2numpy(cnn_correct)*100 / total, decimals=2)
    #         return test_acc
    
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

                if ret_pred_target: # only apply when self._is_training_adapters is False
                    cnn_pred_all.append(tensor2numpy(cnn_preds))
                    target_all.append(tensor2numpy(targets))
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
                return avg, avg
            
            
    def eval_task(self):
        self._logger.info(50*"-")
        self._logger.info("log {} of the task".format(self._eval_metric))
        self._logger.info(50*"-")
        cnn_pred, cnn_pred_score, y_true, features = self._epoch_test(self._network, self._test_loader, True)

        if self._eval_metric == 'acc':
            cnn_total, cnn_task = accuracy(cnn_pred.T, y_true, self._total_classes, self._increment_steps)
        elif self._eval_metric == 'mcr':
            cnn_total, cnn_task = mean_class_recall(cnn_pred.T, y_true, self._total_classes, self._increment_steps)
        else:
            raise ValueError('Unknown evaluate metric: {}'.format(self._eval_metric))
        self._logger.info("Final Test Acc: {:.2f}".format(cnn_total))
        self._logger.info("The Expected Calibration Error (ECE) of the model: {:.4f}".format(cal_ece(cnn_pred, cnn_pred_score, y_true)))
        self._logger.info(' ')
        
        self._known_classes = self._total_classes
    