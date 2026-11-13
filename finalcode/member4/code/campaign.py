"""Full-data candidate search, predeclared confirmation, freeze, test, report."""
import argparse,datetime,json,subprocess,sys
from pathlib import Path
from core import read_json,write_json,sha,require,verify_folder
HERE=Path(__file__).resolve().parent

def runname(c,seed):return f"{c['id']}_s{seed}"
def execute(root,cmd,label):
    logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    log=logs/f'{label}_{stamp}.log'
    with log.open('x',encoding='utf-8') as f:
        f.write(json.dumps({'command':cmd,'cwd':str(HERE)})+'\n');f.flush()
        p=subprocess.Popen(cmd,cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        tail=[]
        for line in p.stdout:
            print(line,end='',flush=True);f.write(line);f.flush();tail=(tail+[line])[-100:]
        status=p.wait();f.write(f'EXIT_CODE={status}\n')
    if status:
        text=''.join(tail)
        reason='OTHER'
        for marker,kind in [('CUDA out of memory','OOM'),('No space left','DISK_FULL'),('Nonfinite','NONFINITE'),('Saved/reloaded model mismatch','RELOAD'),('Reload mismatch','RELOAD'),('Resume identity','RESUME_IDENTITY'),('ConnectionError','NETWORK'),('KeyboardInterrupt','INTERRUPTED')]:
            if marker in text:reason=kind;break
        write_json(log.with_suffix('.incident.json'),{'reason':reason,'exit_code':status,'log':str(log),'command':cmd,'resume':'last COMPLETE epoch only; unfinished epoch must repeat','time':stamp})
        raise RuntimeError(f'{reason}: process exited {status}; see {log}')

def command(a,c,seed,out,action='train'):
    if c['family']=='rnn':
        return [sys.executable,str(HERE/'rnn_full.py'),action,'--data',str(a.data),'--root',str(a.root),'--out',str(out),'--config',str(a.root/'configs'/f"{c['id']}.json"),'--kind',c['kind'],'--seed',str(seed),'--device',a.device,'--micro-batch',str(a.rnn_micro)]
    if action=='evaluate':return [sys.executable,str(HERE/'run_experiment.py'),'evaluate','--data',str(a.data),'--root',str(a.root),'--run',out.name]
    cmd=[sys.executable,str(HERE/'run_experiment.py'),'train','--data',str(a.data),'--out',str(out),'--model',c['model'],'--length',str(c['length']),'--policy',c['policy'],'--lr',str(c['lr']),'--seed',str(seed),'--precision',a.precision,'--micro-batch',str(a.tf_micro)]
    if a.gradient_checkpointing:cmd+=['--gradient-checkpointing']
    return cmd

def train_one(a,c,seed,smoke=False):
    out=a.root/('smoke' if smoke else 'runs')/runname(c,seed)
    if (out/'DONE.json').exists():
        for p in out.glob('epoch-*'):verify_folder(p)
        return
    cmd=command(a,c,seed,out)
    if smoke:cmd+=['--smoke']
    if out.exists():
        require(a.resume,'Incomplete run exists. Read log and use --resume or recover.py restart.')
        require(any(out.glob('epoch-*/COMPLETE.json')),'No complete epoch to resume; use recover.py restart.')
        cmd+=['--resume']
    execute(a.root,cmd,('smoke_' if smoke else '')+out.name)

def selected(a,c,seed):
    f=a.root/'runs'/runname(c,seed);d=read_json(f/'DONE.json')
    require(d['run_kind']=='formal','Smoke cannot be selected')
    return d['selected']

def best(candidates,a):
    rows=[(c,selected(a,c,42)) for c in candidates];top=max(m['macro_f1'] for _,m in rows)
    return min([(c,m) for c,m in rows if top-m['macro_f1']<=1e-4],key=lambda x:(x[1]['nll'],x[0]['id']))[0]

def main(a):
    a.root=a.root.resolve();a.data=a.data.resolve();a.root.mkdir(parents=True,exist_ok=True)
    require(sha(a.data/'data_contract.json')==read_json(a.data/'PREPARED.json')['data_contract'],'Incomplete or changed data preparation')
    matrix=read_json(HERE/'matrix.json');configs=matrix['configs'];byid={c['id']:c for c in configs}
    contract={'matrix':matrix,'data_contract_sha256':sha(a.data/'data_contract.json'),'code':{p.relative_to(HERE).as_posix():sha(p) for p in sorted(HERE.rglob('*.py'))},'precision':a.precision,'tf_micro':a.tf_micro,'rnn_micro':a.rnn_micro,'gradient_checkpointing':a.gradient_checkpointing,'rnn_device':a.device}
    protocol=a.root/'PROTOCOL.json'
    if protocol.exists():require(read_json(protocol)==contract,'Protocol/options changed. Use original options or new root; do not rewrite this file.')
    else:
        write_json(protocol,contract);(a.root/'configs').mkdir()
        for c in configs:write_json(a.root/'configs'/f"{c['id']}.json",c)
    if a.action in ['smoke','screen','confirm']:require(not (a.root/'FINAL_LOCK.json').exists(),'Campaign already frozen')
    if a.action=='smoke':
        for c in configs:train_one(a,c,42,True)
    elif a.action=='screen':
        for c in configs:
            require(read_json(a.root/'smoke'/runname(c,42)/'DONE.json')['run_kind']=='smoke','Missing matching smoke')
            train_one(a,c,42)
        roles={n:n for n in matrix['mandatory_confirm']}
        roles['rnn_winner']=best([c for c in configs if c['family']=='rnn'],a)['id']
        roles['transformer_winner']=best([c for c in configs if c['family']=='transformer'],a)['id']
        choice={'roles':roles,'confirm_ids':list(dict.fromkeys(roles.values())),'based_on':'seed42 full validation only; frozen before other seeds','seeds':matrix['seeds']}
        path=a.root/'SELECTION.json'
        if path.exists():require(read_json(path)==choice,'Selection changed')
        else:write_json(path,choice)
        print(choice)
    elif a.action=='confirm':
        for seed in matrix['seeds']:
            for name in read_json(a.root/'SELECTION.json')['confirm_ids']:train_one(a,byid[name],seed)
    elif a.action=='freeze':
        require(not (a.root/'FINAL_LOCK.json').exists(),'Already frozen');runs={}
        for name in read_json(a.root/'SELECTION.json')['confirm_ids']:
            for seed in matrix['seeds']:
                c=byid[name];n=runname(c,seed);out=a.root/'runs'/n;met=selected(a,c,seed);epoch=met['epoch']
                for ep in out.glob('epoch-*'):
                    verify_folder(ep);m=read_json(ep/'metrics.json');require(m['train_rows_seen']==503979 and m['rows']==55998,'Not full data')
                if c['family']=='transformer':require(read_json(out/'DONE.json')['epochs']==3,'Transformer training incomplete')
                runs[n]={'id':name,'seed':seed,'epoch':epoch,'manifest_sha256':sha(out/'run_manifest.json'),'checkpoint_sha256':sha(out/f'epoch-{epoch}'/'COMPLETE.json')}
        write_json(a.root/'FINAL_LOCK.json',{'runs':runs,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(protocol),'selection_sha256':sha(a.root/'SELECTION.json'),'test_has_historical_exposure':True})
    elif a.action=='test':
        lock=read_json(a.root/'FINAL_LOCK.json');require(sha(protocol)==lock['protocol_sha256'],'Changed protocol')
        for n,r in lock['runs'].items():
            dest=a.root/'test'/n
            if (dest/'COMPLETE.json').exists():verify_folder(dest);continue
            execute(a.root,command(a,byid[r['id']],r['seed'],a.root/'runs'/n,'evaluate'),'test_'+n)
    elif a.action=='report':execute(a.root,[sys.executable,str(HERE/'report.py'),'--root',str(a.root)],'report')
    elif a.action=='status':
        choice=read_json(a.root/'SELECTION.json') if (a.root/'SELECTION.json').exists() else {'confirm_ids':[]}
        print({'screen_complete':sum((a.root/'runs'/runname(c,42)/'DONE.json').exists() for c in configs),'screen_total':len(configs),'confirmation_unique_models':len(choice['confirm_ids']),'test_complete':len(list((a.root/'test').glob('*/COMPLETE.json'))),'frozen':(a.root/'FINAL_LOCK.json').exists()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','screen','confirm','freeze','test','report','status']);p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--precision',choices=['bf16','fp16','fp32'],default='bf16');p.add_argument('--tf-micro',type=int,default=8);p.add_argument('--rnn-micro',type=int,default=32);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--gradient-checkpointing',action='store_true');p.add_argument('--resume',action='store_true');main(p.parse_args())
