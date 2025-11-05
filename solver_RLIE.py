import gc
import torch
import torch.nn as nn
from scipy import stats
import numpy as np
import tqdm
from PLMPI import model
import data_loader
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def plcc_loss(y_pred, y):
    y = y.detach().float()
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

class Solver(object):
    """Solver for training and testing hyperIQA - 这是一个用于训练和测试hyperIQA模型的求解器类"""

    def __init__(self, config, path, train_idx, test_idx, num_worker=8):
        # 初始化逻辑不变，省略重复代码...
        print(f"lr 的大小: {config.lr}")
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.train_patch_num = config.train_patch_num 
        self.model = model().to(device)
        self.model.train(True)
        self.loss = nn.MSELoss().cuda()
        self.lr = config.lr
        self.gamma = config.gamma
        self.weight_decay = config.weight_decay
        self.num_workers=num_worker
        self.solver = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.solver, gamma=self.gamma)
        train_loader = my_loader.DataLoader(config.dataset, path, train_idx, config.patch_size,
                                              config.train_patch_num, batch_size=config.batch_size, istrain=True, num_workers=4, pin_memory=True)
        self.train_data = train_loader.get_data()
        test_loader = my_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, batch_size=config.batch_size,
                                             istrain=False, num_workers=4, pin_memory=True)
        self.test_data = test_loader.get_data()

    def train(self):
        best_srcc = 0.0
        best_plcc = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTrain_PLCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_tensor = torch.tensor([], device=device)
            gt_tensor = torch.tensor([], device=device)
            train_loop = tqdm.tqdm(self.train_data, desc=f"Epoch {self.epochs+1}/{self.epochs} (Train)", leave=False)
            for img, label in train_loop:
                img = img.cuda()
                label = label.cuda()

                self.solver.zero_grad()
                pred = self.model(img)

                pred_tensor = torch.cat([pred_tensor, pred.squeeze()])
                gt_tensor = torch.cat([gt_tensor, label])

                loss = self.loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()
            
            pred_scores = pred_tensor.detach().cpu().numpy() 
            gt_scores = gt_tensor.detach().cpu().numpy() 
            
            pred_scores = np.mean(np.reshape(pred_scores, (-1, self.train_patch_num)), axis=1)
            gt_scores = np.mean(np.reshape(gt_scores, (-1, self.train_patch_num)), axis=1)
            
            group_size = 10
            num_samples = len(pred_scores)
            num_groups = num_samples // group_size
            if num_samples % group_size != 0:
                pred_scores = pred_scores[:num_groups * group_size]
                gt_scores = gt_scores[:num_groups * group_size]
            
            train_srcc_list = []
            train_plcc_list = []
            for i in range(num_groups):
                start_idx = i * group_size
                end_idx = start_idx + group_size
                group_pred = pred_scores[start_idx:end_idx]
                group_gt = gt_scores[start_idx:end_idx]
                srcc, _ = stats.spearmanr(group_pred, group_gt)
                plcc, _ = stats.pearsonr(group_pred, group_gt)
                train_srcc_list.append(srcc)
                train_plcc_list.append(plcc)
            
            train_srcc = np.mean(train_srcc_list) if train_srcc_list else 0.0
            train_plcc = np.mean(train_plcc_list) if train_plcc_list else 0.0

            test_srcc, test_plcc = self.test(self.test_data)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc

            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, train_plcc, test_srcc, test_plcc))
        
        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))
        return best_srcc, best_plcc

    def test(self, data):
        self.model_loda.eval()
        pred_gpu = torch.tensor([], device=device)
        gt_gpu = torch.tensor([], device=device)
        with torch.no_grad():
            val_loop = tqdm.tqdm(data, desc="Val", leave=False)
            for img, label in val_loop: 
                img = img.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                pred = self.model_loda(img)
                pred_gpu = torch.cat([pred_gpu, pred.squeeze()])
                gt_gpu = torch.cat([gt_gpu, label.squeeze()])

        pred_scores = pred_gpu.cpu().numpy()
        gt_scores = gt_gpu.cpu().numpy()

        pred_scores = np.mean(np.reshape(pred_scores, (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(gt_scores, (-1, self.test_patch_num)), axis=1)
        group_size = 10
        num_samples = len(pred_scores)
        num_groups = num_samples // group_size
        if num_samples % group_size != 0:
            pred_scores = pred_scores[:num_groups * group_size]
            gt_scores = gt_scores[:num_groups * group_size]
        
        srcc_list = []
        plcc_list = []
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            group_pred = pred_scores[start_idx:end_idx]
            group_gt = gt_scores[start_idx:end_idx]
            srcc, _ = stats.spearmanr(group_pred, group_gt)
            plcc, _ = stats.pearsonr(group_pred, group_gt)
            srcc_list.append(srcc)
            plcc_list.append(plcc)
        
        avg_srcc = np.mean(srcc_list) if srcc_list else 0.0
        avg_plcc = np.mean(plcc_list) if plcc_list else 0.0
        return avg_srcc, avg_plcc