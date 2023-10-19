import torch
from tree import RTree
from RAE import RAE
import os as os
import numpy as np
from logger import Logger
from sklearn.model_selection import train_test_split
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext import transforms as T
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
    
def build_dataset(reviews, labels, vocab, max_len=512):
    text_transform = T.Sequential(
        T.VocabTransform(vocab=vocab),
        T.Truncate(max_seq_len=max_len),
        T.ToTensor(padding_value=vocab['<pad>']),
        T.PadTransform(max_length=max_len, pad_value=vocab['<pad>']),
    )
    dataset = TensorDataset(text_transform(reviews), torch.tensor(labels))
    return dataset

def load_imdb(datasets:str):
    trainset=np.load(datasets+'/imdb_train.npz')
    valset=np.load(datasets+'/imdb_train.npz')
    testset=np.load(datasets+'/imdb_train.npz')
    torkenizer=get_tokenizer('basic_english')
    reviews_train, labels_train = [torkenizer(s) for s in trainset['x'].tolist()],trainset['y'].tolist()
    reviews_val,labels_val = [torkenizer(s) for s in valset['x'].tolist()],valset['y']
    reviews_test, labels_test = [torkenizer(s) for s in testset['x'].tolist()],testset['y']
    
    vocab = build_vocab_from_iterator(reviews_train, min_freq=3, specials=['<pad>', '<unk>', '<cls>', '<sep>'])
    vocab.set_default_index(vocab['<unk>'])
    train_data = build_dataset(reviews_train, labels_train, vocab)
    val_data = build_dataset(reviews_val,labels_val,vocab)
    test_data = build_dataset(reviews_test, labels_test, vocab)
    return train_data, val_data,test_data, vocab



if __name__ == '__main__':
    train_data, val_data,test_data, vocab=load_imdb('I:/自然语言学习/datasets/aclImdb')
    
    log=Logger()
    if torch.cuda.is_available() :
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    train_dl=DataLoader(dataset=train_data,batch_size=200)
    val_dl=DataLoader(dataset=val_data,batch_size=1)
    model=RAE(device=device,
            vocab_size=vocab.__len__(),
            K=2).to(device)
    params=model.parameters()
    lossFunc2=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(params=params,lr=1e-4)
    val_loss=-1
    log.logout('#########START TRAIN#########')
    for epoch in range(1,200):
        for step ,(x,y) in enumerate(train_dl):
            x=x.to(device)
            y=y.to(device)
            pred,loss1=model(x)
            loss2=lossFunc2(pred,y)
            loss=loss1+loss2
            loss=torch.abs(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log.logout(str(100*step/len(train_dl))+'%'+'  train_loss:'+str(loss.item()))
        if epoch %50 == 0 :
            with torch.no_grad():
                for _,(vx,vy) in enumerate(val_data):
                    vx=vx.to(device)
                    vy=vy.to(device)
                    pred,vloss1=model(vx)
                    vloss2=lossFunc2(pred,y)
                    vloss=vloss1+vloss2
                    vloss=torch.abs(vloss)
                    pred=pred.tolist()
                    ty=ty.tolist()
        log.logout('EPOCH '+str(epoch)+',  val_loss:'+str(vloss.item())+'  acc:'+str(accuracy_score(ty,pred)))
        log.writer.add_scalar('Train loss',loss.item(),epoch)
        log.writer.add_scalar('Val loss',vloss.item(),epoch)
        log.writer.flush()
    log.logout('#########Train Complete#########')
    log.logout('#########START Test#########')
    test_dl=DataLoader(dataset=test_data,batch_size=200)
    accs=[]
    f1s=[]
    pers=[]
    recalls=[]
    with torch.no_grad():
        for _,(tx,ty) in enumerate(test_data):
            tx=tx.to(device)
            ty=ty.to(device)
            pred,loss=model(tx)
            pred=pred.tolist()
            ty=ty.tolist()
            accs.append(accuracy_score(ty,pred))
            recalls.append(recall_score(ty,pred))
            pers.append(precision_score(ty,pred))
            f1s.append(f1_score(ty,pred))
        acc=np.mean(accs)
        log.logout('acc:'+str(loss.item()))  
            