"""Copy pinned full splits; build train-only shared RNN vocabulary and GloVe matrix."""
import argparse, gzip, shutil, urllib.request, urllib.error, time
from pathlib import Path
import numpy as np
import torch
from core import COUNTS, PARENT_SHA, read_frame, sha, write_json, require
from original_rnn.data import Vocabulary
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--resume',action='store_true')
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=a.resume)
files={}; ids={}; texts={}
for split in COUNTS:
    src=a.repo/'processed_data'/f'{split}.csv'
    require(sha(src)==PARENT_SHA[split],f'Wrong full CSV or LFS pointer: {src}')
    f=read_frame(src);require(len(f)==COUNTS[split],f'Wrong {split} row count')
    ids[split]=set(f.sample_id);texts[split]=set(f.text)
    shutil.copyfile(src,a.out/f'{split}.csv')
    files[split]={'sha256':sha(src),'rows':len(f),'class_counts':f.label.value_counts().to_dict()}
    if split=='train':
        vocab=Vocabulary.build(f.text,max_size=30000,min_freq=1);vocab.save(a.out/'vocab.json')
for x,y in [('train','val'),('train','test'),('val','test')]:
    require(not ids[x]&ids[y],f'ID overlap {x}/{y}')
    require(not texts[x]&texts[y],f'Exact text overlap {x}/{y}')
write_json(a.out/'data_contract.json',{'files':files,'split_seed':42,'scope':'full','vocab_sha256':sha(a.out/'vocab.json'),'audit':'IDs and exact text disjoint; no semantic duplicate claim'})
# Raw official gensim-data release, streamed; avoids loading 400k vectors in RAM.
url='https://github.com/piskvorky/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz'
raw=a.out/'glove-wiki-gigaword-100.gz'
if not raw.exists():
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url,raw.with_suffix('.download'))
            raw.with_suffix('.download').replace(raw)
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt==2:raise
            time.sleep(2**attempt)

g=torch.Generator().manual_seed(42);w=torch.empty(len(vocab),100).uniform_(-.1,.1,generator=g);w[0].zero_();found=0
with gzip.open(raw,'rt',encoding='utf-8') as f:
    header=f.readline().split();require(len(header)==2 and int(header[1])==100,'Wrong vector dimension/header')
    for line in f:
        word,*values=line.rstrip().split(' ')
        if word in vocab.token_to_id and word not in ['<pad>','<unk>']:
            v=np.asarray(values,dtype=np.float32);require(v.shape==(100,) and np.isfinite(v).all(),'Malformed vector')
            w[vocab.token_to_id[word]]=torch.from_numpy(v);found+=1
require(found>0,'No GloVe matches')
torch.save(w,a.out/'glove.pt')
write_json(a.out/'embedding_source.json',{'url':url,'raw_sha256':sha(raw),'matrix_sha256':sha(a.out/'glove.pt'),'vocab_sha256':sha(a.out/'vocab.json'),'matched':found,'vocab_size':len(vocab),'coverage':found/(len(vocab)-2),'oov':'uniform(-0.1,0.1), seed42; PAD zero'})
print(files,flush=True)

write_json(a.out/'PREPARED.json',{'data_contract':sha(a.out/'data_contract.json'),'embedding_source':sha(a.out/'embedding_source.json')})
