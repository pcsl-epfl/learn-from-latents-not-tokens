"""Export only main-text figure inputs from a make_summary_data.ipynb output directory.

This reads saved aggregates, not raw simulation data. To regenerate aggregates,
use aggregate_gradient.py and aggregate_iterative.py instead.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FILES=['df_ptc_m_sweep_best.parquet','df_e2e_statistician.parquet','data2vec.pkl','data2vec_sensitivity.pkl']
GRADIENT_COLUMNS=['experiment_name','simulation_id','rhm_L','rhm_v','rhm_m','rhm_s',
 'rhm_Ntrain','rhm_Ntest','rhm_seed_rules','rhm_seed_sample','checkpoint_step',
 'classifier_best_val_acc','classifier_test_acc_at_best_val']

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'data/aggregated')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for name in FILES:
        source=a.source/name;output=a.output/name
        if source.resolve()==output.resolve():raise ValueError('Use a separate output directory')
        record={'file':name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        if name=='df_ptc_m_sweep_best.parquet':
            df=pd.read_parquet(source)
            df=df[df.experiment_name=='ptc_m_sweep_L4'][GRADIENT_COLUMNS]
            df.to_parquet(output,index=False)
            record['rows']=len(df)
        else:
            shutil.copyfile(source,output)
        record['sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
        manifest.append(record)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Exported {len(FILES)} summary inputs to {a.output}')

if __name__=='__main__':main()
