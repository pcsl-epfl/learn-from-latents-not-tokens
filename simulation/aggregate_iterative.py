"""Aggregate historical statistician_e2e result dictionaries into figure inputs."""
import argparse
import json
import re
from pathlib import Path
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r'statistician_e2e_v(?P<v>\d+)_m(?P<m>\d+)_s(?P<s>\d+)_L(?P<L>\d+)_rs(?P<seed>\d+)\.pt$')

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'data/iterative_ilc/results')
    p.add_argument('--output',type=Path,default=ROOT/'data/aggregated/df_e2e_statistician.parquet')
    p.add_argument('--allow-incomplete',action='store_true')
    args=p.parse_args()
    rows=[]
    for f in sorted(args.results.glob('statistician_e2e_*.pt')):
        match=PATTERN.fullmatch(f.name)
        if not match: continue
        metadata={k:int(v) for k,v in match.groupdict().items()}
        payload=torch.load(f,map_location='cpu',weights_only=False)
        for P,accuracy in payload.items():
            rows.append(dict(**metadata,P=int(P),accuracy=float(accuracy)))
    if not rows: raise RuntimeError('No iterative-clustering results found')
    df=pd.DataFrame(rows).sort_values(['v','s','L','m','P','seed']).reset_index(drop=True)
    if df.duplicated(['v','s','L','m','P','seed']).any(): raise ValueError('Duplicate measurements')
    if not df.accuracy.between(0,1).all(): raise ValueError('Invalid accuracy')
    counts=df.groupby(['v','s','L','m','seed']).size()
    issues=[f'{k}: {n}/16 sample counts' for k,n in counts.items() if n!=16]
    if set(zip(df.v,df.s,df.L))=={(8,3,5)}:
        observed=set(zip(df.m,df.seed))
        issues += [f'missing m={m}, seed={seed}' for m in [6,8,12,16,22,32] for seed in range(3) if (m,seed) not in observed]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.coverage.json').write_text(json.dumps(dict(rows=len(df),issues=issues),indent=2)+'\n')
    if issues and not args.allow_incomplete: raise RuntimeError('Incomplete results; inspect coverage report')
    groups=df.groupby(['v','m','s','L','P']).accuracy
    for stat in ['mean','std','min','max']:
        df[f'accuracy_{stat}']=groups.transform(stat)
    df['n_seeds']=groups.transform('count')
    df['accuracy_sem']=df.accuracy_std/df.n_seeds**0.5
    df['P_over_vm3']=df.P/(df.v*df.m**3)
    df.to_parquet(args.output,index=False)
    print(f'Wrote {len(df)} measurements; {len(issues)} coverage issues')

if __name__=='__main__': main()
