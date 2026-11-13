"""Explicit incident actions; archives failures and never suppresses training errors."""
import argparse,datetime,shutil
from pathlib import Path
from core import read_json,require,verify_folder
p=argparse.ArgumentParser();p.add_argument('action',choices=['inspect','archive-partial','restart','archive-test']);p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
require(Path(a.run).name==a.run and a.run not in ['.','..'],'Use a run name, not a path')
root=a.root.resolve();out=root/('smoke' if a.smoke else 'runs')/a.run
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
if a.action=='inspect':
    print({'path':str(out),'disk_free_gb':shutil.disk_usage(root).free/1e9,'complete_epochs':sorted(p.name for p in out.glob('epoch-*') if (p/'COMPLETE.json').exists()),'partial':[p.name for p in out.glob('*.partial')],'logs':[str(p) for p in (root/'logs').glob(f'*{a.run}*')]})
    for ep in out.glob('epoch-*'):verify_folder(ep)
elif a.action=='archive-partial':
    dest=root/'failed'/f'{a.run}_{stamp}';dest.mkdir(parents=True)
    for ep in out.glob('*.partial'):ep.rename(dest/ep.name)
    print(dest)
elif a.action=='restart':
    require(not (root/'FINAL_LOCK.json').exists(),'Frozen run cannot be restarted in this campaign')
    require(not (out/'DONE.json').exists(),'Completed runs are reused')
    dest=root/'failed'/f'{a.run}_{stamp}';dest.parent.mkdir(exist_ok=True);out.rename(dest);print(dest)
else:
    target=root/'test'/a.run;require(not (target/'COMPLETE.json').exists(),'Completed test is reused')
    dest=root/'failed'/f'test_{a.run}_{stamp}';dest.parent.mkdir(exist_ok=True);target.rename(dest);print(dest)
