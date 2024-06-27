import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torch import optim
from argparse import ArgumentParser

from backbone.dynamic_er_net import DERNet
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, tensor2numpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
import random

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
    parser.add_argument('--T', type=float, default=None, help='tempreture apply to the output logits befor softmax')
    parser.add_argument('--epochs_finetune', type=int, default=None, help='balance finetune epochs')
    parser.add_argument('--lrate_finetune', type=float, default=None, help='balance finetune learning rate')
    parser.add_argument('--milestones_finetune', nargs='+', type=int, default=None, help='for multi step learning rate decay scheduler')
    return parser

class Dynamic_ER(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._T = self._config.T
        self._is_finetuning = False

        self._epochs_finetune = config.epochs_finetune
        self._lrate_finetune = config.lrate_finetune
        self._milestones_finetune = config.milestones_finetune
        # if self._incre_type != 'cil':
        #     raise ValueError('Dynamic_ER is a class incremental method!')

    def prepare_model(self, checkpoint=None):
        if self._network == None:
            self._network = DERNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path)
        self._network.update_fc(self._total_classes)

        set_random(1993)
        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            if checkpoint['memory_class_means'] is not None and self._memory_bank is not None:
                self._memory_bank.set_class_means(checkpoint['memory_class_means'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        if self._cur_task>0:
            for i in range(self._cur_task):
                for p in self._network.feature_extractor[i].parameters():
                    p.requires_grad = False
                self._network.feature_extractor[i].eval()
                self._logger.info('Freezing task extractor {} !'.format(i))
            
        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        self._network = self._network.cuda()
        
        
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
        
    def incremental_train(self):
        if self._gpu_num > 1:
            self._network = nn.DataParallel(self._network, list(range(self._gpu_num)))
        if self._cur_task==0:
            self.avg_res = []
            self.openset_res = []
            self.openset_f1 = [] 
        self._logger.info('-'*10 + ' Learning on task {}: {}-{} '.format(self._cur_task, self._known_classes, self._total_classes-1) + '-'*10)
        optimizer = self._get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._config, self._cur_task==0)
        scheduler = self._get_scheduler(optimizer, self._config, self._cur_task==0)
        if self._cur_task == 0:
            epochs = self._init_epochs
        else:
            epochs = self._epochs
        self._is_finetuning = False
        self._network = self._train_model(self._network, self._train_loader,self._val_loader, self._test_loader, optimizer, scheduler,
            task_id=self._cur_task, epochs=epochs, note='stage1', stage = 1)


        print(f'self.res:{self.avg_res}')
        print(f'self.avg_res:{sum(self.avg_res)/len(self.avg_res)}')     
        # if self._cur_task > 0:
        #     self._logger.info('Finetune the network (classifier part) with the balanced dataset!')
        #     finetune_train_dataset = self._memory_bank.get_unified_sample_dataset(self._train_dataset, self._network)
        #     finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=self._batch_size,
        #                                     shuffle=True, num_workers=self._num_workers)
        #     self._network.reset_fc_parameters()
        #     ft_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.fc.parameters()), momentum=0.9, lr=self._lrate_finetune)
        #     ft_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=ft_optimizer, milestones=self._milestones_finetune, gamma=0.1)
        #     self._is_finetuning = True
        #     self._network = self._train_model(self._network, finetune_train_loader, self._test_loader, ft_optimizer, ft_scheduler,
        #         task_id=self._cur_task, epochs=self._epochs_finetune, note='stage2')

        if self._gpu_num > 1:
            self._network = self._network.module
            
        logger = self._network._logger
        self._network._logger = None
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
                test_acc, f1 = self._epoch_test(model, openset_test_loader, ret_task_acc=True)
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
                                                                task_begin=task_begin, task_end=task_end, task_id=task_id, stage = stage)
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
                                                    stage = stage)

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
                                                         stage = stage)
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
                                                         stage = stage,need_qk = False) # True
                else:
                    val_acc, val_test_acc = self._epoch_test(model, val_loader, ret_task_acc=True,
                                                        task_begin=task_begin, task_end=task_end, task_id=task_id,
                                                         stage = stage)
                    
                test_acc, task_test_acc = self._epoch_test(model, test_loader, ret_task_acc=True,
                                                        task_begin=task_begin, task_end=task_end, task_id=task_id,
                                                        stage = stage)
            info += 'val_acc {:.3f}, '.format(val_acc)
            info += 'Test_accy {:.3f}, '.format(test_acc)
            record_dict['Task{}_{}Test_Acc(inner task)'.format(task_id, note)] = test_acc

            
            self._logger.info(info)
            self._logger.visual_log('train', record_dict, step=epoch)
            if stage==1:
                self.avg_res.append(test_acc)
        return model
    
    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None, stage = 1):
        losses = 0.
        correct, total = 0, 0
        losses_clf = 0.
        losses_aux = 0.
        num_classes = 0
        predsAll = torch.tensor([]).cuda()
        targetsAll = torch.tensor([]).cuda()
        
        if isinstance(model, nn.DataParallel):
            model.module.feature_extractor[-1].train()
        else:
            model.feature_extractor[-1].train()

        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            logits, output_features = model(inputs)
            if num_classes==0:
                num_classes = logits.shape[-1]
            # loss = cross_entropy(logits/self._T, targets)

            task_scores = torch.sigmoid(logits/self._T).reshape(logits.shape)
            loss = MultiLabelFocalLoss()(logits/self._T, targets.float())
                
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
    
    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None,stage = 1):
        model.eval()
        
        num_classes = 0
        # Calculate accuracy for each class
        th = 0.5
        predsAll = torch.tensor([]).cuda()
        targetsAll = torch.tensor([]).cuda()
        cnn_pred_all, target_all = [], []
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, output_features = model(inputs)
            
            if num_classes==0:
                num_classes = 1
                correct0 = [0] * num_classes
                total0 = [0] * num_classes
                       
            if ret_pred_target:
                target_all.append(tensor2numpy(targets))
            else:
    
                task_scores = torch.sigmoid(logits).reshape(logits.shape)
                predsAll = torch.concat([predsAll,task_scores.clone()],dim=0)
                targetsAll = torch.concat([targetsAll,targets.clone()],dim=0)
                
                cnn_preds = torch.where(task_scores>th,1,0)
                cnn_pred_all.append(tensor2numpy(cnn_preds))
                target_all.append(tensor2numpy(targets))
        
        if ret_pred_target:
            cnn_max_scores_all = np.concatenate(cnn_max_scores_all)
            return cnn_pred_all, None, cnn_max_scores_all, None, target_all, None
        else:
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
                
                cnn_pred_all = np.concatenate(cnn_pred_all)
                target_all = np.concatenate(target_all)
                f1 = f1_score(target_all, cnn_pred_all)
                return avg, f1
            else:
                return avg