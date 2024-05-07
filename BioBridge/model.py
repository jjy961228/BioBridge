from transformers import (BertForSequenceClassification,
                          XLMForSequenceClassification ,
                          XLMRobertaForSequenceClassification,
                          XLMForSequenceClassification,
                          MBartForSequenceClassification,
                          BertModel)
import sent2vec
import torch.nn as nn
import torch
import ipdb
import numpy as np           
            
class CustomBertForSequenceClassification(nn.Module):

        def __init__(self,config,tokenizer ,model_init_path, num_labels, device , args):
            super(CustomBertForSequenceClassification, self).__init__()
            self.config = config
            self.tokenizer = tokenizer
            self.device = device
            self.args = args

            ## model initiallize
            if self.args.model == 'xlm':
                self.model = XLMForSequenceClassification.from_pretrained(model_init_path, num_labels=num_labels).to(self.device)
            elif self.args.model in ['mbert_cased','mbert_uncased', 'krbert','kobert']:
                self.model = BertForSequenceClassification.from_pretrained(model_init_path, num_labels=num_labels).to(self.device)
            elif self.args.model in ['xlmr_base'] :
                self.model = XLMRobertaForSequenceClassification.from_pretrained(model_init_path,num_labels=num_labels).to(self.device)

            if args.method == 'bridging' or args.method == 'bribio':
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    self.model = self.model.to(self.device)
                    
            ## Linear mapper
            if args.method == 'bribio' or args.method == 'bio':
                self.bioSent2Vec = sent2vec.Sent2vecModel()
                self.bioSent2Vec.load_model('../bioSent2Vec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin')
                if self.args.model in [ 'mbert_cased','mbert_uncased', 'krbert','kobert']:  
                    self.model.bert.embeddings.bio_embeddings = nn.Linear(700, self.config.hidden_size).to(self.device)
                    self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
                    self.token_embeddings = self.model.bert.embeddings.word_embeddings.weight
                elif self.args.model in ['xlmr_base']:
                    self.model.roberta.embeddings.bio_embeddings = nn.Linear(700, self.config.hidden_size).to(self.device)
                    self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
                    self.token_embeddings = self.model.roberta.embeddings.word_embeddings.weight 
                elif self.args.model == 'xlm':
                    self.model.transformer.embeddings.bio_embeddings = nn.Linear(700, self.config.hidden_size).to(self.device)
                    self.dropout = nn.Dropout(self.config.attention_dropout)
                    self.token_embeddings = self.model.transformer.embeddings.weight
                
        def forward(self, input_ids,attention_mask, labels, tok_type_ids, train_flag = None, eng_tok_ids = None, flag_ids = None):     
            
            if (self.args.method == 'bribio' or self.args.method == 'bio') and train_flag == True:
                tensor_list = eng_tok_ids.tolist()
                batch_eng_tok_ids = [
                            [[value for value in sublist2 if value != 0] for sublist2 in sublist1 if sublist2.count(0) != len(sublist2)] 
                                for sublist1 in tensor_list if any(sublist2.count(0) != len(sublist2) for sublist2 in sublist1)
                            ]
                batch_eng_words, batch_word_eng_ids = [] , []
                for b_eng_tok_ids in batch_eng_tok_ids:
                    for eng_tok_ids in b_eng_tok_ids:
                        tok = self.tokenizer.convert_ids_to_tokens(eng_tok_ids)
                        word = ''
                        for id,t in zip(eng_tok_ids,tok):
                            if self.args.model in [ 'mbert_cased','mbert_uncased', 'krbert']:
                                t = t.replace('##','').lower()
                            elif self.args.model in ['kobert','xlmr_base'] :
                                t = t.replace('▁','').lower()   
                            elif self.args.model in ['xlm']:
                                t = t.replace('</w>','').lower()
                            word = word+t
                        
                        if len(word) >1 :
                            batch_word_eng_ids.append(eng_tok_ids)
                            batch_eng_words.append(word)
                        
                batch_bio_embedding = np.array([self.bioSent2Vec.embed_sentence(batch_eng_word) for batch_eng_word in batch_eng_words]) # (Number of English word, 1, 700)
                batch_bio_embedding = torch.tensor(batch_bio_embedding).float().to(self.device)
                
                if self.args.model in ['mbert_cased','mbert_uncased', 'krbert','kobert']:
                    batch_bio_embedding = self.model.bert.embeddings.bio_embeddings(batch_bio_embedding) # (Number of English word , 1, 768)
                    batch_bio_embedding = self.dropout(batch_bio_embedding)
                    # token_embeddings = self.model.bert.embeddings.word_embeddings.weight # [26990, 768] = (vocab_size,768)
                elif self.args.model in ['xlmr_base']:
                    batch_bio_embedding = self.model.roberta.embeddings.bio_embeddings(batch_bio_embedding)
                    batch_bio_embedding = self.dropout(batch_bio_embedding)
                    # token_embeddings = self.model.roberta.embeddings.word_embeddings.weight 
                elif self.args.model == 'xlm':
                    batch_bio_embedding = self.model.transformer.embeddings.bio_embeddings(batch_bio_embedding)
                    batch_bio_embedding = self.dropout(batch_bio_embedding)
                    # token_embeddings = self.model.transformer.embeddings.weight 
                    
                token_embeddings_clone = self.token_embeddings.detach().clone()   
                for i,eng_tok_ids in enumerate(batch_word_eng_ids):
                    if len(eng_tok_ids) > 1:
                        for tok_idx in eng_tok_ids:
                            token_embeddings_clone[tok_idx, : ] = batch_bio_embedding[i].squeeze()
                    else:
                        token_embeddings_clone[eng_tok_ids, : ] = batch_bio_embedding[i].squeeze()
                
                self.token_embeddings = nn.Parameter(token_embeddings_clone)
                
                
                
            if self.args.model in ['mbert_cased','mbert_uncased', 'krbert','kobert','xlmr_base','xlm']:
                loss,logits = self.model(
                                        input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        labels = labels,
                                        token_type_ids=tok_type_ids,
                                        # train_flag = None
                                        )[:2]             
                
            return loss, logits

