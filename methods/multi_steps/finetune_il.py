from os.path import join

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score

from backbone.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.replayBank import ReplayBank
from utils.toolkit import accuracy, count_parameters, tensor2numpy, cal_bwf, mean_class_recall, cal_class_avg_acc, cal_avg_forgetting, cal_openset_test_metrics
from sklearn.metrics import f1_score

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
class Finetune_IL(BaseLearner):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        # 评价指标变化曲线
        self.cnn_task_metric_curve = []
        self.nme_task_metric_curve = []
        self.cnn_metric_curve = []
        self.nme_metric_curve = []
        # for openset test
        self.cnn_auc_curve = []
        self.nme_auc_curve = []
        self.cnn_fpr95_curve = []
        self.nme_fpr95_curve = []
        self.cnn_AP_curve = []
        self.nme_AP_curve = []

        self._incre_type = config.incre_type
        self._apply_nme = config.apply_nme
        self._memory_size = config.memory_size
        self._fixed_memory = config.fixed_memory
        self._sampling_method = config.sampling_method
        if self._fixed_memory:
            self._memory_per_class = config.memory_per_class
        self._memory_bank = None
        if self._fixed_memory != None: # memory replay only support cil or gem famalies
            self._memory_bank = ReplayBank(self._config, logger)
            self._logger.info('Memory bank created!')
        self._is_openset_test = config.openset_test
        
        self._replay_batch_size = config.batch_size if config.replay_batch_size is None else config.replay_batch_size

        self._init_epochs = config.epochs if config.init_epochs is None else config.init_epochs
        self._init_lrate = config.lrate if config.init_lrate is None else config.init_lrate
        self._init_scheduler = config.scheduler if config.init_scheduler is None else config.init_scheduler
        self._init_milestones = config.milestones if config.init_milestones is None else config.init_milestones
        self._init_lrate_decay = config.lrate_decay if config.init_lrate_decay is None else config.init_lrate_decay
        self._init_weight_decay = config.weight_decay if config.init_weight_decay is None else config.init_weight_decay
        self._init_opt_mom = config.opt_mom if config.init_opt_mom is None else config._init_opt_mom
        self._init_nesterov = config.nesterov if config.init_nesterov is None else config._init_nesterov
                
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
            self._network = IncrementalNet(self._logger, self._config.backbone, self._config.pretrained, self._config.pretrain_path, MLP_projector=self._config.MLP_projector)
        
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
            loss = MultiLabelFocalLoss()(logits, targets.float())
                
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
    
    def _epoch_test(self, model, test_loader, ret_task_acc=False, ret_pred_target=False, task_begin=None, task_end=None, task_id=None, stage = 1,need_qk = False,select_id = None):
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
                    logits, _ = model(inputs,select_id = select_id)
                else:
                    _, output_features = model(inputs)
                    if need_qk:
                        logits,qk = model.aux_fc(output_features['features'],need_qk = need_qk)
                        qk_all.append(qk)
                    else:
                        logits = model.aux_fc(output_features['features'])
                         
                if num_classes==0:
                    num_classes = logits.shape[-1]
                    correct0 = [0] * num_classes
                    total0 = [0] * num_classes
                    
                task_scores = torch.sigmoid(logits).reshape(logits.shape)
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

    def incremental_train(self):
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
        self._network = self._train_model(self._network, self._train_loader, self._val_loader, self._test_loader, optimizer, scheduler, task_id=self._cur_task, epochs=epochs)
        self.test_all_tasks(self._network,self._cur_task+1)
        
    def test_all_tasks(self,model,num_tasks):
        self.openset_res.append([])
        self.openset_f1.append([])
        for task in range(num_tasks):            
            with torch.no_grad():
                # _known_classes控制 测试哪个label
                # openset = _total_classes 控制 sofar 数量
                sofar = sum(self._increment_steps[:task+1])
                openset_test_dataset = self.data_manager.get_dataset(indices=np.arange(sofar-1, sofar), source='test', mode='test', increment_steps = self._increment_steps, openset = sum(self._increment_steps[:num_tasks]))
                openset_test_loader = DataLoader(openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
                test_acc, f1 = self._epoch_test(model, openset_test_loader, ret_task_acc=True, select_id = task)
            self.openset_res[-1].append(round(test_acc, 4))
            self.openset_f1[-1].append(round(f1, 4))
        
        print(f'self.openset_res:')
        for index,i in enumerate(self.openset_res):
            print(f'Task {index}: {i}')
        
        avg = [sum(i)/len(i) for i in self.openset_res]
        print(f'Avg_auc:{avg}')
        
        print(f'self.openset_f1:')
        for index,i in enumerate(self.openset_f1):
            print(f'Task {index}: {i}')
        
        avgf1 = [sum(i)/len(i) for i in self.openset_f1]
        print(f'Avg_f1:{avgf1}')
        
    def store_samples(self):
        if self._memory_bank != None:
            self._memory_bank.store_samples(self._sampler_dataset, self._network)
    
    def _train_model(self, model, train_loader,val_loader, test_loader, optimizer, scheduler, task_id=None, epochs=100, note=''):
        task_begin = sum(self._increment_steps[:task_id])
        task_end = task_begin + self._increment_steps[task_id]
        best_val_acc = 0
        early_stopping_counter = 0
        self._early_stopping_patience = 10
        best_model_state_dict = None
        
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
                val_acc, val_test_acc = self._epoch_test(model, val_loader, ret_task_acc=True,
                                                    task_begin=task_begin, task_end=task_end, task_id=task_id)
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
                         
            with torch.no_grad():
                test_acc, task_test_acc = self._epoch_test(model, test_loader, ret_task_acc=True,
                                                    task_begin=task_begin, task_end=task_end, task_id=task_id)
            
            
            if epoch != epochs-1:
                record_dict['Task{}_{}Test_Acc(inner task)'.format(task_id, note)] = test_acc
                info = info + 'Task{}_Test_accy {:.2f}, '.format(task_id, test_acc)
                if self._incre_type == 'cil': # only show epoch test acc in cil, because epoch test acc is worth nothing in til
                    record_dict['Task{}_{}Test_Acc'.format(task_id, note)] = test_acc
                    info = info + 'Test_accy {:.2f}'.format(test_acc)

                self._logger.info(info)
                self._logger.visual_log('train', record_dict, step=epoch)
                
        # Load best model state_dict
        if best_model_state_dict is not None:
            model.load_state_dict(torch.load(f'{self._logger.log_file_name}/best_checkpoint.pkl'))
        with torch.no_grad():
            test_acc, task_test_acc = self._epoch_test(model, test_loader, ret_task_acc=True,
                                                task_begin=task_begin, task_end=task_end, task_id=task_id)
            
            record_dict['Task{}_{}Test_Acc(inner task)'.format(task_id, note)] = test_acc
            info = info + 'Task{}_Test_accy {:.2f}, '.format(task_id, test_acc)
            if self._incre_type == 'cil': # only show epoch test acc in cil, because epoch test acc is worth nothing in til
                record_dict['Task{}_{}Test_Acc'.format(task_id, note)] = test_acc
                info = info + 'Test_accy {:.2f}'.format(test_acc)

            self._logger.info(info)
            self._logger.visual_log('train', record_dict, step=epoch)
            
        return model

    def after_task(self):
        self._known_classes = self._total_classes
        if self._save_models:
            self.save_checkpoint('seed{}_task{}_checkpoint.pkl'.format(self._seed, self._cur_task),
                self._network.cpu(), self._cur_task)

    def save_checkpoint(self, filename, model, task_id):
        save_path = join(self._logdir, filename)
        if self._memory_bank is None:
            memory_class_means = None
        else:
            memory_class_means = self._memory_bank.get_class_means()
        if isinstance(model, nn.DataParallel):
            model = model.module
        save_dict = {'state_dict': model.state_dict(), 'config':self._config.get_parameters_dict(),
                'task_id': task_id, 'memory_class_means':memory_class_means}
        torch.save(save_dict, save_path)
        self._logger.info('checkpoint saved at: {}'.format(save_path))
    
    def save_predict_records(self, cnn_pred, cnn_pred_scores, nme_pred, nme_pred_scores, targets, features):
        record_dict = {}        
        record_dict['cnn_pred'] = cnn_pred
        record_dict['cnn_pred_scores'] = cnn_pred_scores
        record_dict['nme_pred'] = nme_pred
        record_dict['nme_pred_scores'] = nme_pred_scores
        record_dict['targets'] = targets
        record_dict['features'] = features

        filename = 'pred_record_seed{}_task{}.npy'.format(self._seed, self._cur_task)
        np.save(join(self._logdir, filename), record_dict)
    
    def get_cil_pred_target(self, model, test_loader):
        return self._epoch_test(model, test_loader, ret_pred_target=True, task_begin=0, 
                    task_end=self._total_classes, task_id=self._cur_task)
    
    def get_til_pred_target(self, model, test_dataset):
        known_classes = 0
        total_classes = 0
        cnn_pred_result, nme_pred_result, y_true_result, cnn_predict_score, nme_predict_score, features_result = [], [], [], [], [], []
        for task_id in range(self._cur_task + 1):
            cur_classes = self._increment_steps[task_id]
            total_classes += cur_classes
            
            task_dataset = Subset(test_dataset, 
                np.argwhere(np.logical_and(test_dataset.targets >= known_classes, test_dataset.targets < total_classes)).squeeze())
            task_loader = DataLoader(task_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            
            cnn_pred, nme_pred, cnn_pred_score, nme_pred_score, y_true, features = self._epoch_test(model, task_loader,
                        ret_pred_target=True, task_begin=known_classes, task_end=total_classes, task_id=task_id)
            cnn_pred_result.append(cnn_pred)
            y_true_result.append(y_true)
            cnn_predict_score.append(cnn_pred_score)
            features_result.append(features)

            known_classes = total_classes

        cnn_pred_result = np.concatenate(cnn_pred_result)
        y_true_result = np.concatenate(y_true_result)
        cnn_predict_score = np.concatenate(cnn_predict_score)
        features_result = np.concatenate(features_result)

        return cnn_pred_result, nme_pred_result, cnn_predict_score, nme_predict_score, y_true_result, features_result
    
    def release(self):
        super().release()
        if self._memory_bank is not None:
            self._memory_bank = None

    def _get_optimizer(self, params, config, is_init:bool):
        optimizer = None
        if is_init:
            if config.opt_type == 'sgd':
                optimizer = optim.SGD(params, lr=self._init_lrate,
                                      momentum=0 if self._init_opt_mom is None else self._init_opt_mom,
                                      weight_decay=0 if self._init_weight_decay is None else self._init_weight_decay,
                                      nesterov=False if self._init_nesterov is None else self._init_nesterov)
                self._logger.info('Applying sgd: lr={}, momenton={}, weight_decay={}'.format(self._init_lrate, self._init_opt_mom, self._init_weight_decay))
            elif config.opt_type == 'adam':
                optimizer = optim.Adam(params, lr=self._init_lrate,
                                       weight_decay=0 if self._init_weight_decay is None else self._init_weight_decay)
                self._logger.info('Applying adam: lr={}, weight_decay={}'.format(self._init_lrate, self._init_weight_decay))
            elif config.opt_type == 'adamw':
                optimizer = optim.AdamW(params, lr=self._init_lrate,
                                        weight_decay=0 if self._init_weight_decay is None else self._init_weight_decay,)
                self._logger.info('Applying adamw: lr={}, weight_decay={}'.format(self._init_lrate, self._init_weight_decay))
            else:
                raise ValueError('No optimazer: {}'.format(config.opt_type))
        else:
            if config.opt_type == 'sgd':
                optimizer = optim.SGD(params, lr=config.lrate,
                                      momentum=0 if config.opt_mom is None else config.opt_mom,
                                      weight_decay=0 if config.weight_decay is None else config.weight_decay,
                                      nesterov=False if config.nesterov is None else config.nesterov)
                self._logger.info('Applying sgd: lr={}, momenton={}, weight_decay={}'.format(config.lrate, config.opt_mom, config.weight_decay))
            elif config.opt_type == 'adam':
                optimizer = optim.Adam(params, lr=config.lrate,
                                       weight_decay=0 if config.weight_decay is None else config.weight_decay)
                self._logger.info('Applying adam: lr={}, weight_decay={}'.format(config.lrate, config.weight_decay))
            else: 
                raise ValueError('No optimazer: {}'.format(config.opt_type))
        return optimizer
    
    def _get_scheduler(self, optimizer, config, is_init:bool):
        scheduler = None
        if is_init:
            if config.scheduler == 'multi_step':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self._init_milestones, gamma=self._init_lrate_decay)
                self._logger.info('Applying multi_step scheduler: lr_decay={}, milestone={}'.format(self._init_lrate_decay, self._init_milestones))
            elif config.scheduler == 'cos':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self._init_epochs)
                self._logger.info('Applying cos scheduler')
            elif config.scheduler == None:
                scheduler = None
            else: 
                raise ValueError('Unknown scheduler: {}'.format(config.scheduler))
        else:
            if config.scheduler == 'multi_step':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config.milestones, gamma=config.lrate_decay)
                self._logger.info('Applying multi_step scheduler: lr_decay={}, milestone={}'.format(config.lrate_decay, config.milestones))
            elif config.scheduler == 'cos':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs)
                self._logger.info('Applying cos scheduler: T_max={}'.format(config.epochs))
            # elif config.scheduler == 'coslrs':
            #     scheduler = optim.CosineLRScheduler(optimizer, t_initial=self._init_epochs, decay_rate=0.1, lr_min=1e-5, warmup_t=5, warmup_lr_init=1e-6)
            elif config.scheduler == None:
                scheduler = None
            else: 
                raise ValueError('No scheduler: {}'.format(config.scheduler))
        return scheduler