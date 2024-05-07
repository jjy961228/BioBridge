# num_factors=4, metric_factors=[valid_loss, AUPRC, AUROC,Brier]
import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import ipdb

class EarlyStopping:
    '''
    patience: The number of stone epochs, when there is no performance improvement over the epoch.
              default = 5
    verbose : Whether to print information about the progress of EarlyStopping
              If verbose is set to True, output the number of epochs where the verification loss is not improved, even if the model is saved.
              default = True
    delta : delta  (float): The minimum change that counts as an improvement. default: 0.0
    '''
    def __init__(self, patience=5, 
                 verbose=False, 
                 delta=0, 
                 num_factors=4, 
                 save_path='models/'):
        self.patience = patience
        self.verbose = verbose
        
        self.counter = [0 for i in range(num_factors)] # [Loss, AUPRC, AUROC, BRIER, F1]
        self.best_score = [None for i in range(num_factors)] # [Loss, AUPRC, AUROC, BRIER, F1]
        self.ckpt_mark = [0 for i in range(num_factors)]
        self.early_stop_flag = [False for i in range(num_factors)]
        
        self.early_stop = False        
        self.delta = delta
        self.num_factors = num_factors
        self.save_path = save_path
        
        self.factors = {0 : 'Loss', 1:'AUPRC', 2:'AUROC', 3:'BRIER', 4: 'F1'}

    def __call__(self, metric_factors, model):
        """
        :param metric_factors: [Loss_value, AUPRC_value, AUROC_value] : type=list
        :param model: your model
        """
        
        for i, val in enumerate(metric_factors):
            if self.best_score[i] is None:
                self.save_checkpoint(i, val, model)
                self.best_score[i] = val
            else:
                if self.factors[i] == 'Loss' or self.factors[i] == 'BRIER' : # Loss는 계산을 다르게함
                    score = -val
                    best_score = -self.best_score[i]
                else:
                    score = val
                    best_score = self.best_score[i]

                if score < best_score + self.delta: # Best 값을 갱신하지 못했을 경우
                    self.counter[i] += 1
                    if self.verbose:
                        print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter[i] >= self.patience:
                        self.early_stop_flag[i] = True
                else: # Best값을 갱신한 경우
                    self.save_checkpoint(i, val, model)
                    self.best_score[i] = val
                    self.counter[i] = 0
        self.early_stop = all(self.early_stop_flag)
            
    def save_checkpoint(self, factor_num, metric_value, model):
        ckpt_savepath = os.path.join(self.save_path, f'best_{self.factors[factor_num]}_model')
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score[factor_num]} --> {metric_value}). Saving model ...')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)   
        torch.save(model.state_dict(), ckpt_savepath + '.pt')