from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
import torch
import ipdb
import re
import numpy as np


class CustomTokenization(Dataset):
  def __init__(self, data, labels, tokenizer, max_len, args, SPECIAL_TOKENS_DICT=None):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.SPECIAL_TOKENS_DICT = SPECIAL_TOKENS_DICT
        self.args=args
  def __len__(self):
        return len(self.data)
  
  def __getitem__(self, item):
    data = self.data[item]
    labels = self.labels[item]
    
    if self.args.method == 'bribio' or self.args.method == 'bridging' or self.args.method == 'bio':
        origin_data = " ".join(data.split())
    
        word_origin = [word for word in origin_data.split(' ')]

        #----------Find how many words English is broken into (for unified bio embedding phase)----------#
        eng_tok_ids = []
        eng_tok_max_len = 500 
        for word in word_origin:
            if word.encode().isalpha(): #When English words appear
                sub_tokens = self.tokenizer.tokenize(word)
                sub_token_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)

                eng_tok_ids.append(sub_token_ids)
                sub_token_len = len(sub_token_ids)
                
        for idx,eng_tok in enumerate(eng_tok_ids):
            while len(eng_tok) < eng_tok_max_len:
                eng_tok_ids[idx].append(0)

        # [PAD]
        while len(eng_tok_ids) < 512:
            zero_list = [0] * eng_tok_max_len
            eng_tok_ids.append(zero_list)
        # truncation
        if len(eng_tok_ids) > 512:
            eng_tok_ids = eng_tok_ids[:512]
        eng_tok_ids = torch.tensor(eng_tok_ids)
    
    #-----------Add modality-specific segment tokens for "Bridging Modality in Context" phase---------#
    if self.args.method == 'bribio' or self.args.method == 'bridging':
        
        special_tokens = self.SPECIAL_TOKENS_DICT['additional_special_tokens']
        word_tmp = []
        eng_flag = False
        kor_flag = False

        for word in word_origin:
            # When Korean appears 
            if bool(re.search(r'[\u3131-\uD79D]', word)) and not kor_flag:
                word_tmp.append(special_tokens[0]) 
                kor_flag = True
                eng_flag = False  

            # When English appears 
            elif bool(re.search(r'[a-zA-Z]', word)) and not eng_flag:
                word_tmp.append(special_tokens[1])
                kor_flag = False  
                eng_flag = True

            word_tmp.append(word)
            
        data = " ".join(word_tmp)
    
    if self.args.method == 'base' or self.args.method == 'bio':
        encoding = self.tokenizer(
        str(data),
        truncation=True,
        padding='max_length',
        max_length=self.max_len,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt'
        )
    
    if self.args.method == 'bribio' or self.args.method == 'bridging':
        encoding = self.tokenizer.encode_plus(
        str(data),     
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=True,
        padding='max_length', 
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt',
        )
            
    if self.args.method == 'base':
        return {
        'input_ids': encoding['input_ids'].flatten(), 
        'attention_mask': encoding['attention_mask'].flatten(),
        'tok_type_ids': encoding['token_type_ids'].flatten(),
        'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    if self.args.method == 'bribio' or self.args.method == 'bridging':
        return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': torch.tensor(labels, dtype=torch.long),
        'tok_type_ids': encoding['token_type_ids'].flatten(),
        'eng_tok_ids' : eng_tok_ids
        }    
    
    if self.args.method == 'bio':
        return {
        #'text': data,
        'input_ids': encoding['input_ids'].flatten(), 
        'attention_mask': encoding['attention_mask'].flatten(),
        'tok_type_ids': encoding['token_type_ids'].flatten(),
        'labels': torch.tensor(labels, dtype=torch.long),
        'eng_tok_ids' : eng_tok_ids
        }

def customDataLoader(df, tokenizer, max_len, batch_size, args, SPECIAL_TOKENS_DICT = None):
    ds = CustomTokenization(
                    data = df['ER_DHX'].to_numpy(), #ER_DHX means Present Illness(PI) of the EMR dataset. 
                    labels= df['LABEL'].to_numpy(),
                    tokenizer=tokenizer,
                    max_len=max_len,
                    args = args,
                    SPECIAL_TOKENS_DICT = SPECIAL_TOKENS_DICT                   
                    )
    return DataLoader(ds,batch_size=batch_size, sampler = RandomSampler(ds), drop_last=True)

    