"""Single CUDA GPU experiments; epoch-boundary resume, immutable checkpoints, no test in train."""
import argparse
import gc
import math
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import random
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from core import MODELS, choose_epoch, hashes, load_split, read_json, require, scores, sha, truncate, verify_folder, write_json

def event(folder, kind, **fields):
    import json
    with (Path(folder)/'execution.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'time_utc':datetime.now(timezone.utc).isoformat(),'event':kind,**fields})+'\n')

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def runtime(precision):
    require(torch.cuda.is_available(), 'This executable requires a CUDA GPU, not a TPU or CPU runtime')
    require(torch.cuda.device_count() == 1, 'Protocol requires one visible CUDA GPU')
    require(transformers.__version__ == '4.57.1', 'Install requirements.txt and restart runtime')
    if precision == 'bf16':
        require(torch.cuda.is_bf16_supported(including_emulation=False), 'Native BF16 required')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {'python':sys.version, 'torch':torch.__version__, 'transformers':transformers.__version__,
            'numpy':np.__version__, 'pandas':pd.__version__, 'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(), 'precision':precision,
            'package_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()}

def amp(precision):
    return torch.autocast('cuda', dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[precision], enabled=precision!='fp32')

def encode(frame, tok, cfg):
    extra = tok.num_special_tokens_to_add(pair=False)
    require(extra == 2, 'Expected two special tokens')
    xs = np.empty((len(frame), cfg['length']), dtype=np.int64)
    masks = np.empty_like(xs)
    lengths = np.empty(len(frame), dtype=np.int64)
    for start in range(0, len(frame), 512):
        raw = tok(frame.text.iloc[start:start+512].tolist(), add_special_tokens=False, truncation=False)['input_ids']
        for offset, seq in enumerate(raw):
            i = start + offset
            lengths[i] = len(seq) + extra
            body = truncate(seq, cfg['length'], extra, cfg['policy'])
            x = tok.build_inputs_with_special_tokens(body)
            n = cfg['length'] - len(x)
            xs[i] = x + [tok.pad_token_id]*n
            masks[i] = [1]*len(x) + [0]*n
    dataset = TensorDataset(torch.from_numpy(xs), torch.from_numpy(masks), torch.tensor(frame.label.to_numpy(),dtype=torch.long))
    stats = {'rows':len(frame), 'percentiles':dict(zip(['p50','p90','p95','p99'],np.percentile(lengths,[50,90,95,99]).tolist())),
             'over_limit_fraction':{str(n):float((lengths>n).mean()) for n in [128,256,384,512]}}
    return dataset, lengths, stats

def loader(dataset, batch, shuffle=False, seed=0):
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,drop_last=False,num_workers=0,
                      generator=torch.Generator().manual_seed(seed),pin_memory=True)

def infer(model, dataset, precision):
    model.eval()
    rows=[]
    with torch.inference_mode():
        for x,m,y in loader(dataset,16):
            with amp(precision):
                z=model(input_ids=x.cuda(),attention_mask=m.cuda()).logits
            rows.append(z.float().cpu().numpy())
    return np.concatenate(rows)

def model_from(source, revision=None, local=False):
    return AutoModelForSequenceClassification.from_pretrained(source, revision=revision, local_files_only=local,
             num_labels=2, id2label={0:'negative',1:'positive'}, label2id={'negative':0,'positive':1},
             attn_implementation='eager').cuda()

def prediction_csv(path, frame, z, lengths):
    out=frame[['sample_id','label']].copy()
    out['logit_0']=z[:,0]
    out['logit_1']=z[:,1]
    out['prediction']=z.argmax(1)
    shifted=z.astype(np.float64)-z.max(1,keepdims=True)
    prob=np.exp(shifted)
    prob/=prob.sum(1,keepdims=True)
    out['prob_0']=prob[:,0]
    out['prob_1']=prob[:,1]
    out['original_token_length']=lengths
    out.to_csv(path,index=False,lineterminator='\n')

def train(a):
    require(a.micro_batch in [1,2,4,8,16], 'Microbatch must divide effective batch 16')
    env=runtime(a.precision)
    cfg={'model':a.model,'model_id':MODELS[a.model][0],'revision':MODELS[a.model][1], 'length':a.length,
         'policy':a.policy,'lr':a.lr,'seed':a.seed,'epochs':3,'effective_batch':16,'micro_batch':a.micro_batch,
         'precision':a.precision,'weight_decay':0.01,'warmup_ratio':0.1,'attention':'eager',
         'gradient_checkpointing':a.gradient_checkpointing,'tokenizer_fast':False,'selection':'global best macro_f1 within 1e-4; lowest nll; earlier epoch',
         'run_kind':'smoke' if a.smoke else 'formal'}
    require(a.policy=='head' or a.model=='distilbert', 'Head-tail is specified only for DistilBERT')
    data=Path(a.data)
    out=Path(a.out)
    identity={'config':cfg,'environment':env,'data_contract_sha256':sha(data/'data_contract.json'),
              'scripts':{f:sha(Path(__file__).parent/f) for f in ['core.py','run_experiment.py']}}
    if a.resume:
        require({k:v for k,v in read_json(out/'run_manifest.json').items() if k!='environment'}=={k:v for k,v in identity.items() if k!='environment'},'Resume model/data/code options mismatch')
        write_json(out/'resume_environment.json',env)
    else:
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/'run_manifest.json',identity)
    event(out,'resume' if a.resume else 'start',command=sys.argv,cwd=str(Path.cwd()))
    wall_start=time.perf_counter()
    train_frame=load_split(data,'train')
    val_frame=load_split(data,'val')
    require(len(train_frame)==503979 and len(val_frame)==55998,'Full training requires 503979/55998')
    require(not set(train_frame.sample_id)&set(val_frame.sample_id),'Train/val overlap')
    if a.smoke:
        require(not a.resume,'Smoke cannot resume as formal')
        train_frame=train_frame.iloc[:64].copy()
        val_frame=val_frame.iloc[:32].copy()
    seed_all(a.seed)
    tok=AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['revision'],use_fast=False)
    ds,_,train_stats=encode(train_frame,tok,cfg)
    val_ds,val_lengths,val_stats=encode(val_frame,tok,cfg)
    write_json(out/'length_stats.json',{'train':train_stats,'val':val_stats})
    committed=sorted(p for p in out.glob('epoch-*') if (p/'COMPLETE.json').is_file())
    for p in committed:
        verify_folder(p)
    history=[read_json(p/'metrics.json') for p in committed]
    require([r['epoch'] for r in history]==list(range(1,len(history)+1)), 'Noncontiguous committed epochs')
    if a.resume:
        require(bool(committed),'No completed epoch: start a new run')
        model=model_from(committed[-1]/'model',local=True)
    else:
        model=model_from(cfg['model_id'],cfg['revision'])
    if a.gradient_checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    require(cfg['length']<=model.config.max_position_embeddings,'Unsupported input length')
    groups=[{'params':[p for n,p in model.named_parameters() if p.ndim>=2 and not n.endswith('bias')],'weight_decay':0.01},
            {'params':[p for n,p in model.named_parameters() if p.ndim<2 or n.endswith('bias')],'weight_decay':0.0}]
    optimizer=torch.optim.AdamW(groups,lr=a.lr,betas=(0.9,0.999),eps=1e-8)
    steps=math.ceil(len(train_frame)/16)
    scheduler=get_linear_schedule_with_warmup(optimizer,int(steps*3*0.1),steps*3)
    scaler=torch.amp.GradScaler('cuda',enabled=a.precision=='fp16')
    if a.resume:
        # Only load own trusted training state. Never load an unknown pickle checkpoint.
        state=torch.load(committed[-1]/'training_state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
    max_epoch=1 if a.smoke else 3
    for epoch in range(len(history)+1,max_epoch+1):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start=time.perf_counter()
        total_loss=0.0
        seen=0
        iterator=iter(loader(ds,16,shuffle=True,seed=a.seed+epoch))
        for step,(x,m,y) in enumerate(iterator,1):
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0,len(y),a.micro_batch):
                xb=x[offset:offset+a.micro_batch].cuda()
                mb=m[offset:offset+a.micro_batch].cuda()
                yb=y[offset:offset+a.micro_batch].cuda()
                with amp(a.precision):
                    loss=model(input_ids=xb,attention_mask=mb,labels=yb).loss
                require(torch.isfinite(loss).item(),'Nonfinite training loss')
                scaler.scale(loss*(len(yb)/len(y))).backward()
                total_loss+=loss.detach().item()*len(yb)
                seen+=len(yb)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            old_scale=scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            require(scaler.get_scale()>=old_scale,'Skipped optimizer update; reject this run')
            scheduler.step()
            if step%100==0:
                print({'epoch':epoch,'step':step,'steps':steps,'loss':total_loss/seen,'lr':scheduler.get_last_lr()[0]},flush=True)
        torch.cuda.synchronize()
        train_s=time.perf_counter()-start
        start=time.perf_counter()
        z=infer(model,val_ds,a.precision)
        val_s=time.perf_counter()-start
        metrics=scores(z,val_frame.label.to_numpy())
        metrics.update(epoch=epoch,train_loss=total_loss/seen,train_seconds=train_s,val_seconds=val_s,
                       train_rows_seen=seen,optimizer_updates=step,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       parameters=sum(p.numel() for p in model.parameters()))
        require(seen==len(train_frame),'Incomplete epoch')
        pending=out/f'.epoch-{epoch}-{uuid.uuid4().hex}.partial'
        pending.mkdir()
        model.save_pretrained(pending/'model',safe_serialization=True)
        tok.save_pretrained(pending/'model')
        write_json(pending/'preprocessing.json',cfg)
        prediction_csv(pending/'validation.csv',val_frame,z,val_lengths)
        write_json(pending/'metrics.json',metrics)
        torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                    'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all()},pending/'training_state.pt')
        # Reload a separate copy. Never move the live optimizer's model off its device.
        reloaded=model_from(pending/'model',local=True)
        z2=infer(reloaded,val_ds,a.precision)
        delta=float(np.abs(z2-z).max())
        mismatches=int((z.argmax(1)!=z2.argmax(1)).sum())
        write_json(pending/'reload_check.json',{'rows':len(z),'prediction_differences':mismatches,'max_abs_logit_diff':delta})
        require(mismatches==0 and delta<=1e-5,'Saved/reloaded model mismatch; do not accept or test')
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()
        # Reload verification can consume RNG; restore the exact continuation state.
        state=torch.load(pending/'training_state.pt',map_location='cpu',weights_only=False)
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        write_json(pending/'COMPLETE.json',hashes(pending))
        verify_folder(pending)
        final=out/f'epoch-{epoch}'
        require(not final.exists(),'Epoch already exists; use explicit resume')
        pending.rename(final)
        event(out,'epoch_committed',epoch=epoch,checkpoint_sha256=sha(final/'COMPLETE.json'))
        history.append(metrics)
        write_json(out/'selected.json',choose_epoch(history))
        print({'committed_epoch':epoch,'validation':metrics,'selected':choose_epoch(history)},flush=True)
    selected=choose_epoch(history)
    write_json(out/'DONE.json',{'run_kind':cfg['run_kind'],'selected':selected,'epochs':len(history)})
    event(out,'done',invocation_wall_seconds=time.perf_counter()-wall_start)

def evaluate(a):
    root=Path(a.root)
    lock=read_json(root/'FINAL_LOCK.json')
    require(a.run in lock['runs'],'Run was not frozen before test evaluation')
    out=root/'runs'/a.run
    manifest=read_json(out/'run_manifest.json')
    require(sha(out/'run_manifest.json')==lock['runs'][a.run]['manifest_sha256'],'Frozen run changed')
    cfg=manifest['config']
    require(cfg['run_kind']=='formal','Smoke is forbidden for test')
    env=runtime(cfg['precision'])
    write_json(out/'evaluation_environment.json',env)
    epoch=lock['runs'][a.run]['epoch']
    checkpoint=out/f'epoch-{epoch}'
    verify_folder(checkpoint)
    require(sha(checkpoint/'COMPLETE.json')==lock['runs'][a.run]['checkpoint_sha256'],'Frozen checkpoint changed')
    require(sha(Path(a.data)/'data_contract.json')==manifest['data_contract_sha256'],'Different data contract')
    frame=load_split(a.data,'test')
    require(len(frame)==38000 and frame.sample_id.is_unique,'Test must contain exactly 38000 unique IDs')
    dest=root/'test'/a.run
    dest.mkdir(parents=True,exist_ok=False)
    event(dest,'test_start',command=sys.argv,lock_sha256=sha(root/'FINAL_LOCK.json'))
    tok=AutoTokenizer.from_pretrained(checkpoint/'model',use_fast=False,local_files_only=True)
    ds,lengths,_=encode(frame,tok,cfg)
    model=model_from(checkpoint/'model',local=True)
    start=time.perf_counter()
    z=infer(model,ds,cfg['precision'])
    duration=time.perf_counter()-start
    require(len(z)==38000,'Evaluation stopped early')
    prediction_csv(dest/'predictions.csv',frame,z,lengths)
    check=pd.read_csv(dest/'predictions.csv',dtype={'sample_id':str})
    require(check.sample_id.tolist()==frame.sample_id.tolist(),'Prediction rows/IDs changed')
    write_json(dest/'metrics.json',{**scores(z,frame.label.to_numpy()),'inference_seconds':duration,
                                   'lock_sha256':sha(root/'FINAL_LOCK.json'),'historically_exposed_test':True})
    event(dest,'test_complete',rows=len(z),inference_seconds=duration)
    write_json(dest/'COMPLETE.json',hashes(dest))
    print(read_json(dest/'metrics.json'),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    subs=parser.add_subparsers(dest='action',required=True)
    p=subs.add_parser('train')
    p.add_argument('--data',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--length',type=int,choices=[256,384,512],required=True)
    p.add_argument('--policy',choices=['head','headtail'],required=True)
    p.add_argument('--lr',type=float,choices=[1e-5,2e-5,3e-5],required=True)
    p.add_argument('--seed',type=int,choices=[42,123,2026],required=True)
    p.add_argument('--precision',choices=['bf16','fp16','fp32'],default='bf16')
    p.add_argument('--micro-batch',type=int,default=16)
    p.add_argument('--gradient-checkpointing',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p=subs.add_parser('evaluate')
    p.add_argument('--data',required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--run',required=True)
    args=parser.parse_args()
    train(args) if args.action=='train' else evaluate(args)
