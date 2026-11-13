"""Recompute metrics from predictions, seed statistics, paired tests and diagnostic plots."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import read_json,write_json,verify_folder,require,scores,sha

def paired(a,b):
    require(a.sample_id.tolist()==b.sample_id.tolist() and a.label.equals(b.label),'Unaligned paired predictions')
    y=a.label.to_numpy();p=a.prediction.to_numpy();q=b.prediction.to_numpy()
    left=int(((p==y)&(q!=y)).sum());right=int(((p!=y)&(q==y)).sum())
    pv=binomtest(left,left+right,.5).pvalue if left+right else 1.
    counts=np.bincount(y*4+p*2+q,minlength=8)
    draws=np.random.default_rng(20261111).multinomial(len(y),counts/len(y),10000).reshape(-1,2,2,2)
    def metric(cm):
        tp=np.diagonal(cm,axis1=1,axis2=2);d=cm.sum(1)+cm.sum(2)
        return tp.sum(1)/len(y),np.divide(2*tp,d,out=np.zeros_like(tp,dtype=float),where=d!=0).mean(1)
    x,f=metric(draws.sum(3));z,g=metric(draws.sum(2))
    return {'baseline_only_correct':left,'candidate_only_correct':right,'p_raw':float(pv),'delta_accuracy_ci95':np.quantile(z-x,[.025,.975]).tolist(),'delta_macro_f1_ci95':np.quantile(g-f,[.025,.975]).tolist()}

def main(root):
    lock=read_json(root/'FINAL_LOCK.json');selection=read_json(root/'SELECTION.json');rows=[];curves=[];classes=[];errors=[]
    for n,r in lock['runs'].items():
        dest=root/'test'/n;verify_folder(dest);f=pd.read_csv(dest/'predictions.csv',dtype={'sample_id':str})
        require(len(f)==38000 and f.sample_id.is_unique,'Incomplete test CSV')
        met=scores(f[['logit_0','logit_1']].to_numpy(),f.label.to_numpy());cm=np.array(met['confusion_matrix_true_rows'])
        hist=[]
        for ep in (root/'runs'/n).glob('epoch-*'):
            verify_folder(ep);hist.append(read_json(ep/'metrics.json'))
        rows.append({'run':n,'config':r['id'],'seed':r['seed'],'selected_epoch':r['epoch'],'train_rows':503979,'val_rows':55998,'train_seconds':sum(m['train_seconds'] for m in hist),**{k:met[k] for k in ['accuracy','macro_f1','nll','brier_positive_class','ece_15_equal_width']}})
        curves.extend({'run':n,**m} for m in hist)
        for k in [0,1]:
            tp=cm[k,k];support=int(cm[k].sum());predicted=int(cm[:,k].sum())
            classes.append({'run':n,'label':k,'support':support,'precision':float(tp/predicted) if predicted else 0.,'recall':float(tp/support),'f1':float(2*tp/(support+predicted)),'tn':int(cm[1-k,1-k]),'fp':int(cm[1-k,k]),'fn':int(cm[k,1-k]),'tp':int(tp)})
        bad=f[f.prediction!=f.label].copy();bad['confidence']=bad[['prob_0','prob_1']].max(axis=1)
        errors.extend({'run':n,**x} for x in bad.sort_values('confidence',ascending=False).head(30).to_dict('records'))
    df=pd.DataFrame(rows);df.to_csv(root/'PER_SEED.csv',index=False)
    mean=df.groupby('config')[['accuracy','macro_f1','nll','train_seconds']].agg(['mean','std']);mean.to_csv(root/'MEAN_SD.csv')
    pd.DataFrame(curves).to_csv(root/'CURVES.csv',index=False);pd.DataFrame(classes).to_csv(root/'PER_CLASS.csv',index=False);pd.DataFrame(errors).to_csv(root/'ERROR_IDS.csv',index=False)
    tests={}
    for family,base in [('rnn','lstm_random'),('transformer','distilbert_base')]:
        target=selection['roles'][family+'_winner']
        a=pd.read_csv(root/'test'/f'{base}_s42'/'predictions.csv',dtype={'sample_id':str});b=pd.read_csv(root/'test'/f'{target}_s42'/'predictions.csv',dtype={'sample_id':str})
        tests[family]={'baseline':base,'candidate':target,**paired(a,b)}
    order=sorted(tests,key=lambda k:tests[k]['p_raw']);prev=0.
    for i,k in enumerate(order):prev=max(prev,min(1.,(len(order)-i)*tests[k]['p_raw']));tests[k]['p_holm']=prev
    write_json(root/'PAIRED_SEED42.json',{'tests':tests,'assumptions':'review IID; Holm adjusts two prespecified tests; not selection bias, business/user clustering or historical test exposure; CIs are marginal'})
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    cv=pd.DataFrame(curves)
    for name,g in cv.groupby('run'):
        if name.endswith('_s42'):
            g=g.sort_values('epoch');ax=axes[1 if name.startswith(('distilbert','deberta')) else 0];ax.plot(g.epoch,g.macro_f1,label=name.removesuffix('_s42'))
    for ax in axes:ax.set(xlabel='Epoch',ylabel='Full validation Macro-F1');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(root/'VALIDATION_CURVES.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6));x=np.arange(len(mean));ax.bar(x,mean[('accuracy','mean')],yerr=mean[('accuracy','std')]);ax.set_xticks(x,mean.index,rotation=70,ha='right');ax.set_ylabel('Test accuracy (mean ± sample SD)');fig.tight_layout();fig.savefig(root/'TEST_COMPARISON.png',dpi=180);plt.close(fig)
    text='# Full experiment results\n\n503,979 train / 55,998 validation / 38,000 historical test rows. 3 seeds, sample SD (ddof=1).\n\n'+df.to_csv(index=False)+'\n\nPaired tests are in PAIRED_SEED42.json. The test split has historical exposure, so it is not claimed as a fully blind evaluation. The error CSV lists only sample IDs; manual inspection targeted negation, contrast, sarcasm, truncation and label ambiguity. It must not be used to keep tuning and re-report on the same test set.\n'
    (root/'RESULTS.md').write_text(text,encoding='utf-8');print(mean.to_string())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
