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
    """Solver for training and testing"""

    def __init__(self, config, path, train_idx, test_idx, num_worker=8):
        """
        初始化Solver对象

        参数:
        config - 配置对象，包含训练所需的各种参数
        path - 数据集所在的路径
        train_idx - 训练数据的索引
        test_idx - 测试数据的索引
        """
        print(f"lr 的大小: {config.lr}")
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.model = model().to(device)
        self.model.train(True)
        self.loss = nn.MSELoss().cuda()
        self.lr = config.lr
        self.gamma = config.gamma
        self.weight_decay = config.weight_decay
        self.num_workers=num_worker
        # Adam
        self.solver = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        # lr degratioin
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.solver, gamma=self.gamma)
        # train dataloader
        train_loader = my_loader.DataLoader(config.dataset, path, train_idx, config.patch_size,
                                              config.train_patch_num, batch_size=config.batch_size, istrain=True, num_workers=8, pin_memory=True)
        self.train_data = train_loader.get_data()
        # test dataloader
        test_loader = my_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, batch_size=config.batch_size,
                                             istrain=False, num_workers=8, pin_memory=True)
        self.test_data = test_loader.get_data()

    def train(self):
        best_srcc = 0.0
        best_plcc = 0.0
        
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        
        for t in range(self.epochs):
            epoch_loss = []
            # empty tensor
            pred_tensor = torch.tensor([], device=device)
            gt_tensor = torch.tensor([], device=device)

            pred_scores = []
            gt_scores = []
            train_loop = tqdm.tqdm(self.train_data, desc=f"Epoch {self.epochs+1}/{self.epochs} (Train)", leave=False)
            for img, label in train_loop:
                img = img.cuda()
                label = label.cuda()

                self.solver.zero_grad()

                pred = self.model(img)

                # in GPU tensor
                pred_tensor = torch.cat([pred_tensor, pred.squeeze()])
                gt_tensor = torch.cat([gt_tensor, label])

                loss = self.loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()
                
            # GPU->cpu tensor->list
            pred_scores = pred_tensor.cpu().tolist()
            gt_scores = gt_tensor.cpu().tolist()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)

            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc

            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))
            
            # 每个epoch结束后更新学习率
            # self.scheduler.step()

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
                
                # GPU
                pred_gpu = torch.cat([pred_gpu, pred.squeeze()])
                gt_gpu = torch.cat([gt_gpu, label.squeeze()])

        # GPU->CPU
        pred_scores = pred_gpu.cpu().numpy()
        gt_scores = gt_gpu.cpu().numpy()

        pred_scores = np.mean(np.reshape(pred_scores, (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(gt_scores, (-1, self.test_patch_num)), axis=1)
        
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        
        return test_srcc, test_plcc