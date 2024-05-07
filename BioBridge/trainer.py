import ipdb
import torch
import argparse
import json
import random
import numpy as np
from torch import nn
from tqdm import tqdm
import os

class CustomTrainer:

    def __init__(self,model, save_path,
                train_loader,
                valid_loader,
                test_loader,
                optimizer, device, scheduler, 
                args):
        
        self.model = model
        self.save_path = save_path
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.args = args
                
    def train_epoch(self):
        train_flag = True
        self.model.train()
        train_losses = []
        train_total_examples = 0
        train_correct_pred= 0
        
        #----------train the model---------- #
        pbar = tqdm(self.train_loader)
        for data in pbar:
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                labels = data['labels'].to(self.device) 
                tok_type_ids = data['tok_type_ids'].to(self.device)

                if (self.args.model in ['mbert_cased','mbert_uncased','krbert','kobert','xlmr_base'] and 
                    (self.args.method == 'base' or self.args.method == 'bridging')):
                    loss, logits = self.model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            tok_type_ids=tok_type_ids,
                                            labels=labels,
                                            train_flag = train_flag)
                elif (self.args.model in ['mbert_cased','mbert_uncased','krbert','kobert','xlmr_base'] and
                      (self.args.method == 'bribio' or self.args.method == 'bio')):
                    eng_tok_ids = data['eng_tok_ids']
                    loss,logits = self.model(input_ids = input_ids,
                                            attention_mask = attention_mask,
                                            labels = labels,
                                            tok_type_ids = tok_type_ids,
                                            eng_tok_ids = eng_tok_ids,
                                            train_flag = train_flag) 
                pred_values,pred_labels = torch.max(logits ,dim=1)
                train_correct_pred += torch.sum(pred_labels == labels)
                train_total_examples += labels.size(0)
                train_losses.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                pbar.set_postfix(loss='{:.4f}'.format(loss))
        pbar.close() 
        #----------validate the model-----------#
        train_flag = True
        self.model.eval()
        valid_losses = []
        valid_correct_pred= 0
        valid_total_examples = 0
        valid_true_labels = []
        valid_logits = []
        with torch.no_grad():
            for data in tqdm(self.valid_loader):
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                labels = data['labels'].to(self.device) 
                tok_type_ids = data['tok_type_ids'].to(self.device)
                if (self.args.model in ['mbert_cased','mbert_uncased','krbert','kobert','xlmr_base']  and 
                    (self.args.method == 'base' or self.args.method == 'bridging')):
                    loss, logits = self.model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            tok_type_ids=tok_type_ids,
                                            labels=labels,
                                            train_flag=train_flag)
                elif (self.args.model in ['mbert_cased','mbert_uncased','krbert','kobert','xlmr_base'] and
                      (self.args.method == 'bribio' or self.args.method == 'bio')):
                    eng_tok_ids = data['eng_tok_ids']
                    loss,logits = self.model(input_ids = input_ids,
                                            attention_mask = attention_mask,
                                            labels = labels,
                                            tok_type_ids = tok_type_ids,
                                            eng_tok_ids = eng_tok_ids,
                                            train_flag=train_flag)  
                valid_total_examples += labels.size(0)
                valid_losses.append(loss.item())             
                pred_values,pred_labels = torch.max(logits ,dim=1)
                valid_correct_pred += torch.sum(pred_labels == labels)
                valid_true_labels.append(labels.to('cpu').numpy())
                valid_logits.append(logits.detach().cpu().numpy())
            valid_loss = np.mean(valid_losses) 
            train_loss = np.mean(train_losses)
            print('train_total_examples: ',train_total_examples)
            print('valid_total_examples: ',valid_total_examples)
            
            train_acc =  train_correct_pred.double() / train_total_examples
            valid_acc = valid_correct_pred.double() / valid_total_examples
            return train_loss, valid_loss, train_acc, valid_acc, valid_logits,valid_true_labels
    
    def evaluator(self, metric_name):
        train_flag = True
        metric_save_path = os.path.join(self.save_path,metric_name) 
        self.model.load_state_dict(torch.load(metric_save_path+'.pt'))
        self.model.eval()
        # initial_params3 = {n: p.clone().detach() for n, p in self.model.named_parameters() if 'bert' in n}
        test_losses = []
        test_correct_pred= 0
        test_total_examples = 0
        test_true_labels = []
        test_logits = []
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                labels = data['labels'].to(self.device) 
                tok_type_ids = data['tok_type_ids'].to(self.device)
                if (self.args.model in ['mbert_cased','mbert_uncased','krbert','kobert','xlmr_base'] and 
                    (self.args.method == 'base' or self.args.method == 'bridging')):
                    loss, logits = self.model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            tok_type_ids=tok_type_ids,
                                            labels=labels,
                                            train_flag = train_flag)
                elif (self.args.model in ['mbert_cased','mbert_uncased','krbert','kobert','xlmr_base'] and
                      (self.args.method == 'bribio' or self.args.method == 'bio')):
                    eng_tok_ids = data['eng_tok_ids']
                    loss,logits = self.model(input_ids = input_ids,
                                            attention_mask = attention_mask,
                                            labels = labels,
                                            tok_type_ids = tok_type_ids,
                                            eng_tok_ids = eng_tok_ids,
                                            train_flag = train_flag) 

                test_total_examples += labels.size(0)
                test_losses.append(loss.item())             
                pred_values,pred_labels = torch.max(logits ,dim=1)
                test_correct_pred += torch.sum(pred_labels == labels)
                test_true_labels.append(labels.to('cpu').numpy())
                test_logits.append(logits.detach().cpu().numpy())
                    
            test_loss = np.mean(test_losses) 
            print('test_total_examples: ',test_total_examples)
            
            test_acc = test_correct_pred.double() / test_total_examples
            return test_loss,test_acc, test_logits,test_true_labels
