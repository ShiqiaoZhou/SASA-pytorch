import os
import math

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import numpy as np
from data_loader import get_dataset_size, data_generator

import argparse
import torch

from SASA_Model import SASA
from dataset_config import get_dataset_config_class
from hyparams_config import get_hyparams_config_class
import random


def setSeed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-cuda_device', type=str, default='0', help='which gpu to use ')
    parser.add_argument('-dataset', type=str, default='Building', help='which dataset ')
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-seed", type=int, default=10)
    parser.add_argument('-epochs', type=int, default=40)
    parser.add_argument('-target_train', type=str, default='train_240')
    

    args = parser.parse_args()

    dataset_config = get_dataset_config_class(args.dataset)()
    hyparams_config = get_hyparams_config_class(args.dataset)()
    args = parser.parse_args()
    device = torch.device("cuda:" + args.cuda_device) if torch.cuda.is_available() else torch.device('cpu')
    setSeed(args.seed)

    root = "logs"
    os.makedirs(root, exist_ok=True)
    record_file = open(os.path.join(root, "record_d0.0.txt"), mode="a+")
    for src_id, trg_id in dataset_config.scenarios:
        print(f'source :{src_id}  target:{trg_id}')
        print('data preparing..')
        src_train_generator = data_generator(data_path=os.path.join(dataset_config.data_base_path, src_id, 'train.csv'),
                                             segments_length=dataset_config.segments_length,
                                             window_size=dataset_config.window_size,
                                             batch_size=args.batch_size, dataset=args.dataset, is_shuffle=True)
        tgt_train_generator = data_generator(data_path=os.path.join(dataset_config.data_base_path, trg_id, args.target_train + '.csv'),
                                             segments_length=dataset_config.segments_length,
                                             window_size=dataset_config.window_size,
                                             batch_size=args.batch_size, dataset=args.dataset, is_shuffle=True)

        tgt_test_generator = data_generator(data_path=os.path.join(dataset_config.data_base_path, trg_id, 'test.csv'),
                                            segments_length=dataset_config.segments_length,
                                            window_size=dataset_config.window_size,
                                            batch_size=args.batch_size, dataset=args.dataset, is_shuffle=False)

        tgt_test_set_size = get_dataset_size(os.path.join(dataset_config.data_base_path, trg_id, 'test.csv'),
                                             args.dataset, dataset_config.window_size)

        print('model preparing..')
        model = SASA(max_len=dataset_config.window_size, coeff=hyparams_config.coeff,
                     segments_num=dataset_config.segments_num, input_dim=dataset_config.input_dim,
                     class_num=dataset_config.class_num,
                     h_dim=hyparams_config.h_dim, dense_dim=hyparams_config.dense_dim,
                     drop_prob = hyparams_config.drop_prob,
                     lstm_layer = hyparams_config.lstm_layer)

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=hyparams_config.learning_rate,
                                     weight_decay=hyparams_config.weight_decay)

        global_step = 0
        total_train_label_loss = 0.0
        total_train_domain_loss = 0.0

        best_rmse = 0
        best_r2 = 0
        best_mae = 0
        best_mape = 0
        best_step = 0
        print('start training...')
        First = True
        while global_step < hyparams_config.training_steps: # 一次训练步一个bacth
            model.train()
            src_train_batch_x, src_train_batch_y, src_train_batch_l = src_train_generator.__next__() # 没有epochs，只有根据batch的training steps

            tgt_train_batch_x, tgt_train_batch_y, tgt_train_batch_l = tgt_train_generator.__next__()

            if src_train_batch_y.shape[0] != tgt_train_batch_y.shape[0]:  #
                continue
            src_x = torch.tensor(src_train_batch_x).to(device)
            tgt_x = torch.tensor(tgt_train_batch_x).to(device)
            src_y = torch.tensor(src_train_batch_y).to(device) # 分类时，转换成long类型

            batch_y_pred, batch_total_loss = model.forward(src_x=src_x, src_y=src_y, tgt_x=tgt_x)

            optimizer.zero_grad()
            # batch_total_loss = torch.sqrt(batch_total_loss)
            batch_total_loss.backward()
            optimizer.step()
            global_step += 1
            
            if global_step % hyparams_config.test_per_step == 0 and global_step != 0:

                total_tgt_test_label_loss = 0.0

                tgt_test_epoch = int(math.ceil(tgt_test_set_size / float(args.batch_size)))
                tgt_test_y_pred_list = list()
                tgt_test_y_true_list = list()
                for _ in range(tgt_test_epoch):
                    model.eval()
                    with torch.no_grad():
                        test_batch_tgt_x, test_batch_tgt_y, test_batch_tgt_l = tgt_test_generator.__next__()

                        test_x = torch.tensor(test_batch_tgt_x).to(device)
                        test_y = torch.tensor(test_batch_tgt_y).long().to(device)

                        batch_tgt_y_pred, batch_tgt_total_loss =  model.forward(src_x=test_x, src_y=test_y, tgt_x=torch.clone(test_x))

                        total_tgt_test_label_loss += batch_tgt_total_loss.detach().cpu().numpy()

                        tgt_test_y_pred_list.extend(batch_tgt_y_pred.detach().cpu().numpy())
                        tgt_test_y_true_list.extend(test_y.detach().cpu().numpy())

                mean_tgt_test_label_loss = total_tgt_test_label_loss / tgt_test_epoch
                tgt_test_y_pred_list = np.asarray(tgt_test_y_pred_list)
                tgt_test_y_true_list = np.asarray(tgt_test_y_true_list)
                # print(tgt_test_y_true_list)
                # print(tgt_test_y_pred_list)
                rmse = np.sqrt(mean_squared_error(tgt_test_y_true_list, tgt_test_y_pred_list)) # 回归问题；分类问题用roc_auc_score
                r2 = r2_score(tgt_test_y_true_list, tgt_test_y_pred_list)
                mae = mean_absolute_error(tgt_test_y_true_list, tgt_test_y_pred_list)
                mape = mean_absolute_percentage_error(tgt_test_y_true_list, tgt_test_y_pred_list)
                if First:
                    best_rmse = rmse
                    best_r2 = r2
                    best_mae = mae
                    best_mape = mape
                    First = False
                if best_rmse > rmse:
                    best_rmse = rmse
                    best_r2 = r2
                    best_mae = mae
                    best_mape = mape

                print("global_steps", global_step, "score", best_rmse)
                print("total loss",mean_tgt_test_label_loss)
                print("best_rmse", best_rmse, )
                print("best_r2", best_r2, )
                print("best_mae", best_mae, )
                print("best_mape", best_mape, '\n')

        print("src:%s -- trg:%s trg_train:%s, best_rmse: %g , best_r2: %g , best_mae: %g , best_mape: %g  \n\n" % (src_id, trg_id, args.target_train, best_rmse, best_r2, best_mae, best_mape), file=record_file)
        record_file.flush()