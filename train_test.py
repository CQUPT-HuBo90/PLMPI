import argparse
import random
import numpy as np
from solver import Solver

def main(config):
    # 数据集路径配置
    folder_path = {
        'live': '/home/live/databaserelease2',
        'csiq': '/home/csiq',
        'tid2013': '/home/tid2013',
        'livec': '/home/clive',
        'koniq-10k': '/home/koniq10k',
        'bid': '/home/BID',
        'RLIE': '/home/RLIE',
        'SQUARE-LOL': '/home/SQUARE-LOL',
    }

    # 各数据集的索引范围
    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
        'LEDataset': list(range(0, 1000)),
        'RLIE': list(range(0, 1540)),
        'SQUARE-LOL': list(range(0, 2900)),
        'RNTIEQA': list(range(0, 2000)),
        'AGIQA-3K': list(range(0, 2982)),
    }
    
    # 
    if  (config.dataset == 'RLIE') | (config.dataset == 'SQUARE-LOL') :
        total_groups = len(img_num[config.dataset]) // 10
        if len(img_num[config.dataset]) % 10 != 0:
            total_groups += 1
        
        all_groups = list(range(total_groups))
        
        test_group_count = 30
        max_test_rounds = total_groups - test_group_count + 1
        if max_test_rounds < 1:
            config.train_test_num = 1
            max_test_rounds = 1
        elif config.train_test_num > max_test_rounds:
            config.train_test_num = max_test_rounds
        
        test_groups_list = []
        used_groups = set()
        while len(test_groups_list) < config.train_test_num:
            selected = random.sample(all_groups, test_group_count)
            selected_sorted = tuple(sorted(selected))
            if selected_sorted not in used_groups:
                used_groups.add(selected_sorted)
                test_groups_list.append(selected_sorted)
    else:
        sel_num = img_num[config.dataset]

    srcc_all = np.zeros(config.train_test_num, dtype=float)
    plcc_all = np.zeros(config.train_test_num, dtype=float)

    print(f'Training and testing on {config.dataset} dataset for {config.train_test_num} rounds...')
    for i in range(config.train_test_num):
        print(f'Round {i+1}')
        
        if  (config.dataset == 'RLIE') | (config.dataset == 'SQUARE-LOL'):
            current_test_groups = test_groups_list[i]
            
            train_index = []
            test_index = []
            
            for group in all_groups:
                start_idx = group * 10
                end_idx = min((group + 1) * 10, len(img_num[config.dataset]))
                
                group_indices = list(range(start_idx, end_idx))
                
                if group in current_test_groups:
                    test_index.extend(group_indices)
                else:
                    train_index.extend(group_indices)
        else:
            sel_num = img_num[config.dataset].copy()
            random.shuffle(sel_num)
            train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
            test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
        
        print(f'训练集大小: {len(train_index)}, 测试集大小: {len(test_index)}')
        
        solver = Solver(config, folder_path[config.dataset], train_index, test_index, num_worker=16)
        srcc_all[i], plcc_all[i] = solver.train()

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    print(f'Testing median SRCC {srcc_med:.4f},\tmedian PLCC {plcc_med:.4f}')

if __name__ == '__main__':                                                                                                                                                                                                                                                                                                                                                                                                                                 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='csiq', help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--dataset1', dest='dataset1', type=str, default='live', help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--dataset2', dest='dataset2', type=str, default='csiq', help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=20, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=15, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=4e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10 , help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.95, help='Gamma value for learning rate scheduling')
    config = parser.parse_args()
    main(config)
