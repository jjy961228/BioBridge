## BioBridge: Unified Bio Embeddings with Bridging Modality in Code-Switched EMR. ##

### Requirements ### 
First, install PyTorch by following the instructions from the official website. To faithfully reproduce our results, please use the correct 1.11.0 version corresponding to your platforms/CUDA versions. PyTorch version higher than 1.11.0 should also work.
```
conda create --name=BioBridge python=3.8
conda activate BioBridge

git clone https://github.com/jjy961228/ConCSE.git
pip install -r requirements.txt
```

### Getting Started ###

1. You will need to download the Sent2Vec library from the Sent2Vec author's repository.

    URL : https://github.com/epfml/sent2vec

2. You should refer to the BioSentVec author's repository to download the weights of the pre-trained BioSentVec model. 21GB(700dim, trained on PubMed+MIMIC-III). 

   URL : https://github.com/ncbi-nlp/BioSentVec
```

### Train BioBridge ###
In the following section, we describe how to train a BioBridge model by using our code.
```
bash bert_base.sh
```

```
bash xlm.sh
```
