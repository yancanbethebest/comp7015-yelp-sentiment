"""Full-data Random/Frozen BiLSTM, FP32, epoch-boundary resume, validation-only choice."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse, random, time, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from core import load_split, read_json, write_json, sha, hashes, verify_folder, require, scores, choose_epoch
from original_rnn.data import Vocabulary
from original_rnn.model import RecurrentClassifier

def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def dataset(frame,vocab,length=128):
    x=np.zeros((len(frame),length),dtype=np.int64);lengths=[]
    for i,t in enumerate(frame.text):
        v=vocab.encode(t,length);x[i,:len(v)]=v;lengths.append(len(v))
    return TensorDataset(torch.from_numpy(x),torch.tensor(lengths),torch.tensor(frame.label.to_numpy()))
def batches(ds,shuffle=False,s=42,batch=64):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,drop_last=False,num_workers=0,generator=torch.Generator().manual_seed(s))
def forward(model,b,device):
    x,l,y=b
    return model(x[:,:int(l.max())].to(device),l)
def infer(model,ds,device):
    model.eval();zs=[]
    with torch.inference_mode():
        for b in batches(ds):zs.append(forward(model,b,device).cpu().numpy())
    return np.concatenate(zs)
def csv(path,frame,z):
    f=frame[['sample_id','label']].copy();f['logit_0']=z[:,0];f['logit_1']=z[:,1];f['prediction']=z.argmax(1)
    p=np.exp(z.astype(float)-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);f['prob_0']=p[:,0];f['prob_1']=p[:,1];f.to_csv(path,index=False)
def main(a):
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    cfg=read_json(a.config)
    model_args={k:cfg[k] for k in ['rnn_type','hidden_dim','pooling','dropout','embedding_dropout']}
    seed(a.seed);v=Vocabulary.load(a.data/'vocab.json')
    env={'python':sys.version,'torch':torch.__version__,'device':a.device,'gpu':torch.cuda.get_device_name() if a.device=='cuda' else 'cpu','pip':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()}
    identity={'kind':a.kind,'seed':a.seed,'smoke':a.smoke,'data_sha256':sha(a.data/'data_contract.json'),'vocab_sha256':sha(a.data/'vocab.json'),'glove_sha256':sha(a.data/'glove.pt'),'code_sha256':sha(Path(__file__)),'dependencies':{n:sha(Path(__file__).parent/n) for n in ['core.py','original_rnn/model.py','original_rnn/data.py']},'environment':env,'batch':cfg['batch'],'length':cfg['length'],'max_epochs':cfg['epochs'],'lr':cfg['lr'],'weight_decay':cfg['weight_decay'],'patience':cfg['patience'],'min_delta':cfg['min_delta'],'config':cfg,'micro_batch':a.micro_batch}
    model=RecurrentClassifier(len(v),**model_args).to(a.device)
    if a.kind!='random':
        model.embedding.weight.data.copy_(torch.load(a.data/'glove.pt',weights_only=True).to(a.device))
    if a.kind=='frozen':model.freeze_embeddings()
    optimizer=torch.optim.AdamW([{'params':[model.embedding.weight],'lr':cfg['embedding_lr'],'weight_decay':0.0}, {'params':[p for n,p in model.named_parameters() if n!='embedding.weight'],'lr':cfg['lr'],'weight_decay':cfg['weight_decay']}])
    if a.action=='evaluate':
        lock=read_json(a.root/'FINAL_LOCK.json')['runs'][a.out.name];verify_folder(a.out/f"epoch-{lock['epoch']}")
        require(sha(a.out/'run_manifest.json')==lock['manifest_sha256'],'Frozen manifest changed')
        require({k:v for k,v in read_json(a.out/'run_manifest.json').items() if k!='environment'}=={k:v for k,v in identity.items() if k!='environment'},'Evaluation model/data/code options changed')
        write_json(a.out/'evaluation_environment.json',env)
        ep=a.out/f"epoch-{lock['epoch']}";require(sha(ep/'COMPLETE.json')==lock['checkpoint_sha256'],'Frozen checkpoint changed')
        state=torch.load(ep/'state.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
        f=load_split(a.data,'test');ds=dataset(f,v,cfg['length']);dest=a.root/'test'/a.out.name;dest.mkdir(parents=True,exist_ok=False)
        start=time.perf_counter();z=infer(model,ds,a.device);seconds=time.perf_counter()-start
        csv(dest/'predictions.csv',f,z);write_json(dest/'metrics.json',{**scores(z,f.label),'inference_seconds':seconds});write_json(dest/'COMPLETE.json',hashes(dest));return
    if a.resume:
        require({k:v for k,v in read_json(a.out/'run_manifest.json').items() if k!='environment'}=={k:v for k,v in identity.items() if k!='environment'},'Resume identity changed')
        write_json(a.out/'resume_environment.json',env)
    else:a.out.mkdir(parents=True,exist_ok=False);write_json(a.out/'run_manifest.json',identity)
    tr=load_split(a.data,'train');va=load_split(a.data,'val')
    if a.smoke:tr=tr.iloc[:128];va=va.iloc[:64]
    ds=dataset(tr,v,cfg['length']);vd=dataset(va,v,cfg['length'])
    eps=sorted(a.out.glob('epoch-*'),key=lambda p:int(p.name.split('-')[1]));history=[]
    for ep in eps:verify_folder(ep);history.append(read_json(ep/'metrics.json'))
    wait=0;best=-1.
    if a.resume:
        st=torch.load(eps[-1]/'state.pt',map_location='cpu',weights_only=False);model.load_state_dict(st['model']);optimizer.load_state_dict(st['optimizer'])
        random.setstate(st['python_rng']);np.random.set_state(st['numpy_rng']);torch.set_rng_state(st['torch_rng']);torch.cuda.set_rng_state_all(st['cuda_rng']);wait=st['wait'];best=st['best']
    limit=1 if a.smoke else cfg['epochs']
    for epoch in range(len(history)+1,limit+1):
        if wait>=cfg['patience']:break
        if a.kind=='gradual':model.embedding.weight.requires_grad_(epoch>cfg['freeze_epochs'])
        model.train();start=time.perf_counter();loss_sum=0.;seen=0
        for step,b in enumerate(batches(ds,True,a.seed+epoch,cfg['batch']),1):
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0,len(b[2]),a.micro_batch):
                micro=tuple(t[offset:offset+a.micro_batch] for t in b)
                z=forward(model,micro,a.device);y=micro[2].to(a.device)
                loss=torch.nn.functional.cross_entropy(z,y)
                require(torch.isfinite(loss).item(),'Nonfinite loss')
                (loss*len(y)/len(b[2])).backward();seen+=len(y);loss_sum+=loss.item()*len(y)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            if step%500==0:print({'epoch':epoch,'step':step,'seen':seen,'loss':loss_sum/seen},flush=True)
        require(seen==len(tr),'Incomplete training epoch');duration=time.perf_counter()-start
        z=infer(model,vd,a.device);met={**scores(z,va.label),'epoch':epoch,'train_loss':loss_sum/seen,'train_rows_seen':seen,'optimizer_updates':step,'train_seconds':duration,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        if met['macro_f1']>best+cfg['min_delta']:best=met['macro_f1'];wait=0
        else:wait+=1
        ep=a.out/f'.epoch-{epoch}.partial';ep.mkdir()
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'wait':wait,'best':best,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},ep/'state.pt')
        check=RecurrentClassifier(len(v),**model_args).to(a.device);st=torch.load(ep/'state.pt',map_location='cpu',weights_only=False);check.load_state_dict(st['model']);z2=infer(check,vd,a.device);del check
        delta=float(abs(z-z2).max());diff=int((z.argmax(1)!=z2.argmax(1)).sum());write_json(ep/'reload_check.json',{'prediction_differences':diff,'max_abs_logit_diff':delta});require(diff==0 and delta<=1e-5,'Reload mismatch')
        random.setstate(st['python_rng']);np.random.set_state(st['numpy_rng']);torch.set_rng_state(st['torch_rng']);torch.cuda.set_rng_state_all(st['cuda_rng'])
        csv(ep/'validation.csv',va,z);write_json(ep/'metrics.json',met);write_json(ep/'COMPLETE.json',hashes(ep));ep.rename(a.out/f'epoch-{epoch}');history.append(met)
        write_json(a.out/'selected.json',choose_epoch(history));print(met,flush=True)
    write_json(a.out/'DONE.json',{'run_kind':'smoke' if a.smoke else 'formal','epochs':len(history),'selected':choose_epoch(history),'stop_reason':'patience' if wait>=cfg['patience'] else 'max_epochs'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate']);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--root',type=Path);p.add_argument('--kind',choices=['random','frozen','finetuned','gradual'],required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--micro-batch',type=int,default=64);p.add_argument('--seed',type=int,choices=[42,123,2026],required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');main(p.parse_args())
