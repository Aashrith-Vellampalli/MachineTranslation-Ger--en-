import spacy
from datasets import load_dataset
from collections import Counter
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence


spacy_de=spacy.load("de_core_news_sm")
spacy_en=spacy.load("en_core_web_sm")

def Tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

def Tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

dataset=load_dataset("bentrevett/multi30k")

train_data=dataset["train"]
test_data=dataset["test"]

PAD_TOKEN="<pad>"
UNK_TOKEN="<unk>"
SOS_TOKEN="<sos>"
EOS_TOKEN="<eos>"

PAD_IDX=0
UNK_IDX=1
SOS_IDX=2
EOS_IDX=3

#building vocab-de
counter_de=Counter()
for sentence in train_data["de"]:
    tokens=Tokenize_de(sentence)
    counter_de.update(tokens)

vocab_de={
    PAD_TOKEN:PAD_IDX,
    UNK_TOKEN:UNK_IDX,
    SOS_TOKEN:SOS_IDX,
    EOS_TOKEN:EOS_IDX
}
start=4

for tok,freq in counter_de.items():
    if(freq>=2):
        vocab_de[tok]=start
        start+=1

#building vocab-en
counter_en=Counter()
for sentence in train_data["en"]:
    tokens=Tokenize_en(sentence)
    counter_en.update(tokens)

vocab_en={
    PAD_TOKEN:PAD_IDX,
    UNK_TOKEN:UNK_IDX,
    SOS_TOKEN:SOS_IDX,
    EOS_TOKEN:EOS_IDX
}

start=4
for tok,freq in counter_en.items():
    if(freq>=2):
        vocab_en[tok]=start
        start+=1


#defining snetence to vec func
def StoT_de(sentence):
    tokens=Tokenize_de(sentence)
    return [vocab_de.get(tok,UNK_IDX) for tok in tokens]

def StoT_en(sentence):
    tokens=Tokenize_en(sentence)
    return [vocab_en.get(tok,UNK_IDX) for tok in tokens]


#dataset class
class TranslationDataset(Dataset):
    def __init__(self,data,StoT_en,StoT_de,SOS_IDX,EOS_IDX):
        self.StoT_en=StoT_en
        self.StoT_de=StoT_de
        self.SOS_IDX=SOS_IDX
        self.EOS_IDX=EOS_IDX
        self.data=data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        en_sent=self.data["en"][index]
        de_sent=self.data["de"][index]

        en_vec=[self.SOS_IDX]+self.StoT_en(en_sent)+[self.EOS_IDX]
        de_vec=[self.SOS_IDX]+self.StoT_de(de_sent)+[self.EOS_IDX]

        return de_vec,en_vec
    

itow_en={idx:token for token,idx in vocab_en.items()}
itow_de={idx:token for token,idx in vocab_de.items()}

def collate_fn(batch):
    de_texts=[]
    en_texts=[]

    for(de,en) in batch:
        de_texts.append(torch.tensor(de,dtype=torch.long))
        en_texts.append(torch.tensor(en,dtype=torch.long))

    de_batch=pad_sequence(de_texts,batch_first=True,padding_value=0)
    en_batch=pad_sequence(en_texts,batch_first=True,padding_value=0)
    return de_batch,en_batch
        





    







