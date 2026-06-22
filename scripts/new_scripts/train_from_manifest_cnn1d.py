from __future__ import annotations

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score
from htc import Config
from adapters.spectrum_adapter import extract_spectrum_vector

class SpectrumDataset(Dataset):
    def __init__(self, X, y): self.X=torch.tensor(X,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx].unsqueeze(0), self.y[idx]

class CNN1D(nn.Module):
    def __init__(self, input_len, n_classes):
        super().__init__()
        self.features=nn.Sequential(nn.Conv1d(1,16,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(16,32,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(16))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(64*16,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,n_classes))
    def forward(self,x): return self.classifier(self.features(x))

def build_label_space(df):
    labels=sorted(df['class_name'].astype(str).unique().tolist()); name_to_index={n:i for i,n in enumerate(labels)}; return labels,name_to_index,Config({'label_mapping':name_to_index})

def split_manifest(df,test_size,random_state,group_by_subject):
    if group_by_subject and df['subject_name'].nunique()>=2:
        gss=GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=random_state); tr,te=next(gss.split(df,y=df['class_name'],groups=df['subject_name'])); return df.iloc[tr].copy(),df.iloc[te].copy()
    sss=StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=random_state); tr,te=next(sss.split(df,df['class_name'])); return df.iloc[tr].copy(),df.iloc[te].copy()

def load_feature_matrix(df,sheet_name,normalize,target_len=None):
    X,ok_rows,errors,inferred_len=[],[],[],target_len
    for _,row in df.iterrows():
        try:
            vec=extract_spectrum_vector(row['spectrum_path'],sheet_name=sheet_name,target_len=inferred_len,normalize=normalize)
            if inferred_len is None: inferred_len=vec.size
            X.append(vec); ok_rows.append(row)
        except Exception as e:
            err=row.to_dict(); err['status']=f'error: {e}'; errors.append(err)
    if not X: raise ValueError('No usable feature vectors could be extracted')
    return np.stack(X,axis=0),pd.DataFrame(ok_rows),pd.DataFrame(errors),inferred_len

def aggregate_subject_rows(df_eval,y_true,y_pred,n_classes):
    rows_by_subject={}
    for i,row in df_eval.reset_index(drop=True).iterrows():
        subject=row['subject_name']; cm=confusion_matrix([int(y_true[i])],[int(y_pred[i])],labels=list(range(n_classes))); acc=float(accuracy_score([int(y_true[i])],[int(y_pred[i])]))
        rows_by_subject.setdefault(subject,[]).append({'timestamp':row['timestamp'],'class_name':row['class_name'],'confusion_matrix':cm,'accuracy':acc})
    rows=[]
    for subject,items in rows_by_subject.items():
        cms=np.stack([it['confusion_matrix'] for it in items],axis=0)
        rows.append({'subject_name':subject,'image_names':[it['timestamp'] for it in items],'n_images':len(items),'condition':items[0]['class_name'],'confusion_matrix':cms.sum(axis=0),'accuracy':float(np.mean([it['accuracy'] for it in items]))})
    return pd.DataFrame(rows)

def main(args):
    manifest_path,out_dir=Path(args.manifest),Path(args.output_dir); out_dir.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    df=pd.read_csv(manifest_path); required={'class_name','subject_name','timestamp','sample_dir','spectrum_path'}; missing=required-set(df.columns)
    if missing: raise SystemExit(f'Manifest missing columns: {sorted(missing)}')
    labels,name_to_index,config=build_label_space(df); df['label_index']=df['class_name'].map(name_to_index)
    train_df,test_df=split_manifest(df,args.test_size,args.random_state,not args.no_group_split)
    X_train,ok_train_df,train_err_df,target_len=load_feature_matrix(train_df,args.sheet,args.normalize,None)
    X_test,ok_test_df,test_err_df,_=load_feature_matrix(test_df,args.sheet,args.normalize,target_len)
    y_train=ok_train_df['label_index'].to_numpy(dtype=np.int64); y_test=ok_test_df['label_index'].to_numpy(dtype=np.int64)
    train_loader=DataLoader(SpectrumDataset(X_train,y_train),batch_size=args.batch_size,shuffle=True); test_loader=DataLoader(SpectrumDataset(X_test,y_test),batch_size=args.batch_size,shuffle=False)
    model=CNN1D(input_len=target_len,n_classes=len(labels)).to(device); loss_fn=nn.CrossEntropyLoss(); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    hist=[]
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for xb,yb in train_loader:
            xb,yb=xb.to(device), yb.to(device); opt.zero_grad(); logits=model(xb); loss=loss_fn(logits,yb); loss.backward(); opt.step(); losses.append(float(loss.item()))
        hist.append({'epoch':epoch+1,'train_loss':float(np.mean(losses))}); print(f"epoch {epoch+1}/{args.epochs} train_loss={np.mean(losses):.4f}")
    model.eval(); preds=[]; probs=[]
    with torch.no_grad():
        for xb,_ in test_loader:
            xb=xb.to(device); logits=model(xb); p=torch.softmax(logits,dim=1); pred=torch.argmax(p,dim=1); preds.extend(pred.cpu().numpy().tolist()); probs.extend(torch.max(p,dim=1).values.cpu().numpy().tolist())
    y_pred=np.asarray(preds,dtype=np.int64)
    aggregate_subject_rows(ok_test_df,y_test,y_pred,len(labels)).to_pickle(out_dir/'test_table.pkl.xz'); config.save_config(out_dir/'config.json')
    pred_df=ok_test_df.copy(); pred_df['true_label_index']=y_test; pred_df['pred_label_index']=y_pred; pred_df['true_label_name']=[labels[i] for i in y_test]; pred_df['pred_label_name']=[labels[i] for i in y_pred]; pred_df['pred_confidence']=probs; pred_df.to_csv(out_dir/'predictions.csv',index=False)
    qc_df=pd.concat([train_err_df,test_err_df],ignore_index=True) if (not train_err_df.empty or not test_err_df.empty) else pd.DataFrame(columns=list(df.columns)+['status']); qc_df.to_csv(out_dir/'qc_report.csv',index=False)
    torch.save({'state_dict':model.state_dict(),'labels':labels,'feature_length':target_len}, out_dir/'model.pt')
    with open(out_dir/'label_map.json','w',encoding='utf-8') as f: json.dump({'labels':labels,'name_to_index':name_to_index},f,indent=2)
    with open(out_dir/'run_info.json','w',encoding='utf-8') as f: json.dump({'mode':'folder->manifest->CNN1D','manifest':str(manifest_path),'n_total':int(len(df)),'n_train_manifest_rows':int(len(train_df)),'n_test_manifest_rows':int(len(test_df)),'n_train_used':int(len(ok_train_df)),'n_test_used':int(len(ok_test_df)),'feature_length':int(target_len),'sheet':args.sheet,'normalize':bool(args.normalize),'labels':labels,'test_size':float(args.test_size),'group_split':bool(not args.no_group_split),'epochs':int(args.epochs),'batch_size':int(args.batch_size),'lr':float(args.lr),'device':str(device),'mean_sample_accuracy':float((y_test==y_pred).mean())},f,indent=2)
    pd.DataFrame(hist).to_csv(out_dir/'train_history.csv',index=False); ok_train_df.to_csv(out_dir/'train_manifest.csv',index=False); ok_test_df.to_csv(out_dir/'test_manifest.csv',index=False)
    print(f'Mean sample accuracy: {float((y_test==y_pred).mean()):.4f}')
    print(f'Saved run dir: {out_dir}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Train a local 1D CNN from folder-derived manifest and write HTC-style outputs')
    p.add_argument('--manifest',required=True,type=str); p.add_argument('--output-dir',required=True,type=str); p.add_argument('--sheet',default='0_derivative',type=str); p.add_argument('--normalize',action='store_true'); p.add_argument('--test-size',default=0.3,type=float); p.add_argument('--random-state',default=42,type=int); p.add_argument('--epochs',default=15,type=int); p.add_argument('--batch-size',default=64,type=int); p.add_argument('--lr',default=1e-3,type=float); p.add_argument('--cpu',action='store_true'); p.add_argument('--no-group-split',action='store_true'); main(p.parse_args())
