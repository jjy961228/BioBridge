from transformers import (get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup,
                        BertTokenizerFast, 
                        XLMTokenizer, 
                        XLMRobertaTokenizer,
                        BertConfig,
                        AutoConfig)        
from tqdm import tqdm
from torch import nn, optim
import numpy as np
import pandas as pd
import torch
import json
import os
import ipdb
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
import os
import copy
import wandb
from dotenv import load_dotenv
import random
import os

#----------made module-----------#
from utils import EarlyStopping
import sent2vec
from dataset import Load_dataset
from customLoader import customDataLoader
from model import CustomBertForSequenceClassification
from trainer import CustomTrainer

class TrainingConfig:
    def __init__(self, config):
        self.epochs = config.get('EPOCHS')
        self.batch_size = config.get('BS')
        self.patience = config.get('PATIENCE')
        self.max_len = config.get('MAX_LEN')

class Arguments:
    def __init__(self,args):
        self.run_wandb = args.run_wandb
        self.random_seed = args.random_seed
        self.model = args.model
        self.ckpt_dir = args.ckpt_dir
        self.method = args.method
        self.schedular = args.schedular
        self.warmup_step = args.warmup_step
        self.lr = args.lr
        self.eps = args.eps
        self.classifier_lr = args.classifier_lr
        self.BioEmbLr = args.BioEmbLr

def fix_seed(seed):
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_wandb", type=bool)
    parser.add_argument("--random_seed", type=int, default=44) 
    parser.add_argument("--model", type=str, default='krbert') # krbert,kobert,mbert_cased,mbert_uncased,xlmr_base 
    parser.add_argument("--method", type= str, default='base') 
    parser.add_argument("--ckpt_dir", type=str,default='./output')
    parser.add_argument("--schedular",type = str, default='linear') # linear , coisine, restart_cosine
    parser.add_argument("--warmup_step", type = float, default=0.1) # 0.1, 0.15, 0.2 
    parser.add_argument("--lr", type = float, default=5e-5) 
    parser.add_argument("--eps", type = float, default=1e-8)
    parser.add_argument("--classifier_lr", type = float, default=2e-2) 
    parser.add_argument("--BioEmbLr", type = float, default=2e-2) 
    args = parser.parse_args()
    args = Arguments(args)
    
    if args.model in ['mbert_cased','mbert_uncased', 'krbert','kobert','xlmr_base']:
        with open('./config.json','r') as f:
            train_config = json.load(f)
    elif args.model in ['xlm']:
        with open('./config_large.json','r') as f:
            train_config = json.load(f)  
    train_config = TrainingConfig(train_config)
    combined_config = {**vars(args), **vars(train_config)}
    print('======random_seed====== :  ', args.random_seed)
    print(vars(train_config))
    print(vars(args))
    
    save_dir = os.path.join(args.ckpt_dir,'seed' + str(args.random_seed) , args.method , args.model)
    save_filename = (args.schedular + 
                        '_warm_' + str(args.warmup_step) + 
                        '_lr_' + str(args.lr) +
                        '_classifier_lr_' + str(args.classifier_lr) +
                        'BioEmbLr'+str(args.BioEmbLr))
    namer = ('seed'+'_'+ str(args.random_seed) + 
                '_' + args.model + 
                '_' +args.method + 
                '_' +args.schedular + 
                '_warm_' + str(args.warmup_step) + 
                '_lr_' + str(args.lr) +
                '_classifier_lr_' + str(args.classifier_lr)+
                'BioEmbLr' + str(args.BioEmbLr))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_filename, 'best_model')
    device = torch.device('cuda')
    
    fix_seed(args.random_seed)
    
    if args.run_wandb == True:
        try:
            import wandb
            from dotenv import load_dotenv
            load_dotenv()
            WANDB_API_KEY = os.getenv('Insert your Wandibi API key')
            wandb.login(key=WANDB_API_KEY)
            wandb.init(project = 'Insert your project name',  
            name = namer,
            entity="Insert your entity name", config=combined_config)
 
        except Exception as e:
            print(f'wandb error:{e}')
            pass
    
    train_data,valid_data, test_data, threshold  = Load_dataset.load_dataset()
    
    ## Load model
    # mBERT
    if args.model == 'mbert_cased':
        model_init_path = "bert-base-multilingual-cased"
        tokenizer = BertTokenizerFast.from_pretrained(model_init_path)
        config = AutoConfig.from_pretrained(model_init_path)
    
    if args.model == 'mbert_uncased':
        model_init_path = "bert-base-multilingual-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(model_init_path)
        config = AutoConfig.from_pretrained(model_init_path)
    
    # KR-BERT
    if args.model == 'krbert':
        model_init_path = "snunlp/KR-Medium"
        tokenizer = BertTokenizerFast.from_pretrained(model_init_path)
        config = AutoConfig.from_pretrained(model_init_path)
    
    # KoBERT
    if args.model == 'kobert':
        from tokenization_kobert import KoBertTokenizer
        model_init_path = "monologg/kobert"
        tokenizer = KoBertTokenizer.from_pretrained(model_init_path)
        config = AutoConfig.from_pretrained(model_init_path)    
    
    # xlm    
    if args.model == 'xlm':
        model_init_path = "xlm-mlm-100-1280"
        tokenizer = XLMTokenizer.from_pretrained(model_init_path)
        config = AutoConfig.from_pretrained(model_init_path)       

    # xlmr_base
    if args.model == 'xlmr_base':
        model_init_path = "xlm-roberta-base"
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_init_path)
        config = AutoConfig.from_pretrained(model_init_path) 
    
    # Bridging        
    if args.method == 'bridging' or args.method == 'bribio':
        if args.model in ['mbert_cased','mbert_uncased', 'krbert','kobert']:
            SPECIAL_TOKENS_DICT = {'additional_special_tokens' : ['[B-KO]','[B-ENG]']}
            tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
            # tokenizer.all_special_tokens => ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]', '[B-KO]', '[B-ENG]']
            # self.tokenizer.all_special_ids => [100, 102, 0, 101, 103, 119547, 119548, 119549, 119550]
        elif args.model in ['xlmr_base','xlm']:
            SPECIAL_TOKENS_DICT = {'additional_special_tokens' : ['<B-KO>','<B-ENG>']}
            tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        
    # Tokenizer & DataLoader
    if args.method == 'bridging' or args.method == 'bribio':
        train_loader = customDataLoader(train_data, tokenizer, train_config.max_len, train_config.batch_size, args, SPECIAL_TOKENS_DICT)
        valid_loader = customDataLoader(valid_data, tokenizer, train_config.max_len, train_config.batch_size, args, SPECIAL_TOKENS_DICT)
        test_loader = customDataLoader(test_data, tokenizer, train_config.max_len, train_config.batch_size, args, SPECIAL_TOKENS_DICT)
    else :
        train_loader = customDataLoader(train_data, tokenizer, train_config.max_len, train_config.batch_size, args)
        valid_loader = customDataLoader(valid_data, tokenizer, train_config.max_len, train_config.batch_size, args)
        test_loader = customDataLoader(test_data, tokenizer, train_config.max_len, train_config.batch_size, args)
    
    model = CustomBertForSequenceClassification(config = config,
                                                tokenizer = tokenizer,
                                                model_init_path = model_init_path,
                                                num_labels=2,
                                                device=device,
                                                args = args)
    if args.method == 'bribio' or args.method == 'bio' :
        if args.model in ['mbert_cased','mbert_uncased', 'krbert','kobert']:
            encoder_param = [p for n, p in model.model.bert.named_parameters() if "bio_embeddings" not in n]
            # for n,p in model.model.bert.named_parameters() :
            #     if 'bio_embeddings' in n:
            #         print('bio !!')
            #         ipdb.set_trace()
            optimizer = optim.AdamW([{'params': encoder_param,'lr': args.lr},
                                    {'params': model.model.classifier.parameters(),'lr': args.classifier_lr},
                                    {'params': model.model.bert.embeddings.bio_embeddings.parameters(),'lr': args.BioEmbLr}
                                    # {'params': model.model.bert.embeddings.bio_embeddings2.parameters(),'lr': args.BioEmbLr}
                                    ], 
                                        eps=1e-8) 
        if args.model in ['xlmr_base']:
            encoder_param = [p for n, p in model.model.roberta.named_parameters() if "bio_embeddings" not in n]
            optimizer = optim.AdamW([{'params': encoder_param,'lr': args.lr}, 
                                    {'params': model.model.classifier.parameters(),'lr': args.classifier_lr},
                                    {'params': model.model.roberta.embeddings.bio_embeddings.parameters(),'lr': args.BioEmbLr}],
                                        eps=1e-8)
        if args.model in ['xlm']:
            encoder = [p for n, p in model.model.transformer.named_parameters() if "bio_embeddings" not in n]
            optimizer = optim.AdamW([{'params': encoder,'lr': args.lr}, 
                                    {'params': model.model.sequence_summary.parameters(),'lr': args.classifier_lr},
                                    {'params': model.model.transformer.embeddings.bio_embeddings.parameters(),'lr': args.BioEmbLr}],
                                        eps=1e-8)
        
    if args.method == 'base' or args.method == 'bridging':
        if args.model in ['mbert_cased','mbert_uncased', 'krbert','kobert']:
            optimizer = optim.AdamW([{'params': model.model.bert.parameters(),'lr': args.lr},
                                    {'params': model.model.classifier.parameters(),'lr': args.classifier_lr},],
                                        eps=1e-8)
        if args.model in ['xlmr_base']:
            optimizer = optim.AdamW([{'params': model.model.roberta.parameters(),'lr': args.lr},
                                    {'params': model.model.classifier.parameters(),'lr': args.classifier_lr},],
                                        eps=1e-8)
        if args.model in ['xlm']:
            optimizer = optim.AdamW([{'params': model.model.transformer.parameters(),'lr': args.lr},
                                    {'params': model.model.sequence_summary.parameters(),'lr': args.classifier_lr},],
                                        eps=1e-8)

    total_steps = len(train_loader) * train_config.epochs
    num_warmup_steps = int(args.warmup_step * total_steps)

    if args.schedular == 'linear' : 
        schedular_type = get_linear_schedule_with_warmup
    elif args.schedular == 'cosine' :
        schedular_type = get_cosine_schedule_with_warmup
    elif args.schedular == 'restart_cosine' :
        schedular_type = get_cosine_with_hard_restarts_schedule_with_warmup
    
    scheduler = schedular_type(optimizer, 
                                num_warmup_steps = num_warmup_steps,
                                num_training_steps = total_steps)
    
    customTrainer = CustomTrainer(model= model,
                                        save_path=save_path,
                                        train_loader=train_loader,
                                        valid_loader=valid_loader,
                                        test_loader=test_loader,
                                        optimizer=optimizer,
                                        device=device,
                                        scheduler=scheduler,
                                        args = args)
    ES_obj = EarlyStopping(patience=5,
                           verbose=True,
                           delta=0, 
                           num_factors=5, # [Loss, AUPRC, AUROC, BRIER,F1]
                           save_path=save_path)
    ##Train
    # prt_task = str(args.random_seed) +'_' + args.model
    for epoch in range(train_config.epochs):
        print(f'Epoch:{epoch+1}/{train_config.epochs}')
        print(f'----------Train roop: {namer}-----------')
        train_loss ,valid_loss, train_acc, valid_acc, logits, true_labels = customTrainer.train_epoch()
        # initial_params4 = {n: p.clone().detach() for n, p in model.named_parameters() if 'bert' in n}
        # ipdb.set_trace()
                    
        if valid_loader is not None:
            from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score, f1_score, recall_score, brier_score_loss, precision_score
            predictions = np.concatenate(logits, axis=0) 
            true_labels = np.concatenate(true_labels, axis=0)
            pred_probas = torch.softmax(torch.tensor(predictions), dim=-1)[:,1] 
            pred_labels = np.where(pred_probas >= threshold , 1, 0)
            AUROC = roc_auc_score(true_labels, pred_probas)
            AUPRC = average_precision_score(true_labels, pred_probas)
            TH_ACC = accuracy_score(true_labels, pred_labels)
            RECALL = recall_score(true_labels, pred_labels)
            PRECISION = precision_score(true_labels, pred_labels)
            F1 = f1_score(true_labels, pred_labels)
            BRIER = brier_score_loss(true_labels, pred_probas)
            confunsion = confusion_matrix(true_labels, pred_labels)
            print('==========Validation==========')
            print('AUC score \n', AUROC,'\n')
            print('AUPRC score\n', AUPRC,'\n')
            print('Accuracy score \n', TH_ACC,'\n')
            print('recall score \n', RECALL,'\n')
            print('f1 score \n', F1,'\n')
            print('confusion matrix \n', confunsion,'\n')

            ES_obj(metric_factors=[valid_loss, AUPRC, AUROC, BRIER, F1], model=model)
            if args.run_wandb == True:
                    wandb.log({'train_loss' : train_loss,
                        'valid_loss': valid_loss,
                       'valid_acc': valid_acc,
                       'valid_thacc': TH_ACC,
                       'valid_auroc': AUROC,
                       'valid_auprc': AUPRC,
                       'valid_recall': RECALL,
                       'valid_precision': PRECISION,
                       'valid_f1_score': F1,
                       'valid_brier': BRIER},
                        step=epoch) 
            if ES_obj.early_stop:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch' : epoch,
                    'train_loss' : train_loss,
                    'train_acc' : train_acc,
                    'valid_loss' : valid_loss,
                    'valid_acc' : valid_acc,
                }, os.path.join(save_path, f'checkpoint_{epoch}.tar'))
                break
    
    ckpt_name = ['best_Loss_model','best_AUPRC_model','best_AUROC_model','best_BRIER_model', 'best_F1_model']
    for metric_name in ckpt_name:
        test_loss, test_acc, logits, true_labels = customTrainer.evaluator(metric_name)
        if test_loader is not None:
            from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score, f1_score, recall_score, brier_score_loss, precision_score
            predictions = np.concatenate(logits, axis=0) 
            true_labels = np.concatenate(true_labels, axis=0)
            pred_probas = torch.softmax(torch.tensor(predictions), dim=-1)[:,1] 
            pred_labels = np.where(pred_probas >= threshold , 1, 0)
            AUROC = roc_auc_score(true_labels, pred_probas)
            AUPRC = average_precision_score(true_labels, pred_probas)
            TH_ACC = accuracy_score(true_labels, pred_labels)
            RECALL = recall_score(true_labels, pred_labels)
            PRECISION = precision_score(true_labels, pred_labels)
            F1 = f1_score(true_labels, pred_labels)
            BRIER = brier_score_loss(true_labels, pred_probas)
            confunsion = confusion_matrix(true_labels, pred_labels)
            print(f'==========Evaluation:{metric_name}==========')
            print('AUC score \n', AUROC,'\n')
            print('AUPRC score \n', AUPRC,'\n')
            print('Accuracy score \n', TH_ACC,'\n')
            print('recall score \n', RECALL,'\n')
            print('f1 score \n', F1,'\n')
            print('confusion matrix \n', confunsion,'\n')
            
            if args.run_wandb == True and metric_name == 'best_Loss_model':
                    wandb.log({'(loss)loss' : test_loss,
                       '(loss)thacc': TH_ACC,
                       '(loss)auroc': AUROC,
                       '(loss)auprc': AUPRC,
                       '(loss)recall': RECALL,
                       '(loss)precision': PRECISION,
                       '(loss)f1_score': F1,
                       '(loss)brier': BRIER},
                        step=epoch) 
            if args.run_wandb == True and metric_name == 'best_AUPRC_model':
                    wandb.log({'(PRC)loss' : test_loss,
                       '(PRC)thacc': TH_ACC,
                       '(PRC)auroc': AUROC,
                       '(PRC)auprc': AUPRC,
                       '(PRC)recall': RECALL,
                       '(PRC)precision': PRECISION,
                       '(PRC)f1_score': F1,
                       '(PRC)brier': BRIER},
                        step=epoch) 
            if args.run_wandb == True and metric_name == 'best_AUROC_model':
                    wandb.log({'(ROC)loss' : test_loss,
                       '(ROC)thacc': TH_ACC,
                       '(ROC)auroc': AUROC,
                       '(ROC)auprc': AUPRC,
                       '(ROC)recall': RECALL,
                       '(ROC)precision': PRECISION,
                       '(ROC)f1_score': F1,
                       '(ROC)brier': BRIER},
                        step=epoch) 
            if args.run_wandb == True and metric_name == 'best_BRIER_model':
                    wandb.log({'(BRI)loss' : test_loss,
                       '(BRI)thacc': TH_ACC,
                       '(BRI)auroc': AUROC,
                       '(BRI)auprc': AUPRC,
                       '(BRI)recall': RECALL,
                       '(BRI)precision': PRECISION,
                       '(BRI)f1_score': F1,
                       '(BRI)brier': BRIER},
                        step=epoch) 
            if args.run_wandb == True and metric_name == 'best_F1_model':
                    wandb.log({'(F1)loss' : test_loss,
                       '(F1)thacc': TH_ACC,
                       '(F1)auroc': AUROC,
                       '(F1)auprc': AUPRC,
                       '(F1)recall': RECALL,
                       '(F1)precision': PRECISION,
                       '(F1)f1_score': F1,
                       '(F1)brier': BRIER},
                        step=epoch) 

if __name__ == '__main__':
    main()