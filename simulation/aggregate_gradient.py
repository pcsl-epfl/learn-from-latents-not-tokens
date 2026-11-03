"""Extract one validation-selected root-classifier measurement per simulation.

Matches analysis_ptc_m_sweep.ipynb: greatest best_val_acc; ties choose the
latest checkpoint step. Never select by test accuracy. No bootstrap samples.
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['experiment_name','simulation_id','rhm_L','rhm_v','rhm_m','rhm_s',
           'rhm_Ntrain','rhm_Ntest','rhm_seed_rules','rhm_seed_sample',
           'checkpoint_step','classifier_best_val_acc','classifier_test_acc_at_best_val']

def collect(results, expected_sweep=None):
    rows, failures = [], []
    runs = sorted(p for p in results.iterdir() if p.is_dir())
    for run in runs:
        source = run / 'classifier_results.pkl'
        if not source.exists():
            failures.append(f'{run.name}: missing classifier_results.pkl')
            continue
        with source.open('rb') as handle:
            payload = pickle.load(handle)
        checkpoints = payload.get('checkpoint_results', [])
        ok = [r for r in checkpoints if r.get('status') == 'ok'
              and np.isfinite(r.get('best_val_acc', np.nan))
              and np.isfinite(r.get('test_acc_at_best_val', np.nan))]
        if len(ok) != len(checkpoints):
            failures.append(f'{run.name}: {len(checkpoints)-len(ok)} failed/invalid checkpoints')
        if not ok:
            failures.append(f'{run.name}: no valid classifier checkpoints')
            continue
        best = max(ok, key=lambda r:(r['best_val_acc'],r['step']))
        rows.append(dict(experiment_name='ptc_m_sweep_L4',simulation_id=run.name,
                         **{k:best['checkpoint_args'][k] for k in COLUMNS if k.startswith('rhm_')},
                         checkpoint_step=best['step'], classifier_best_val_acc=best['best_val_acc'],
                         classifier_test_acc_at_best_val=best['test_acc_at_best_val']))
    frame = pd.DataFrame(rows,columns=COLUMNS)
    if not frame.empty:
        if frame.duplicated(['rhm_m','rhm_Ntrain','rhm_seed_rules']).any():
            failures.append('Duplicate parameter/seed measurements')
        expected = pd.read_csv(expected_sweep or ROOT/'simulation/gradient_sweep.csv', header=None,
                               names=['m','P','seed','epochs'])
        if not ((frame.rhm_L == 4) & (frame.rhm_v == 10) & (frame.rhm_s == 3)).all():
            failures.append('Unexpected grammar parameters: expected L=4, v=10, s=3')
        observed = set(zip(frame.rhm_m,frame.rhm_Ntrain,frame.rhm_seed_rules))
        planned = set(zip(expected.m,expected.P,expected.seed))
        failures += [f'missing m={m}, P={P}, seed={seed}' for m,P,seed in sorted(planned-observed)]
        failures += [f'unexpected m={m}, P={P}, seed={seed}' for m,P,seed in sorted(observed-planned)]
    return frame, failures

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'data/gradient_ilc/results')
    p.add_argument('--output',type=Path,default=ROOT/'data/aggregated/df_ptc_m_sweep_best.parquet')
    p.add_argument('--expected-sweep',type=Path,default=None,help='Alternative m,P,seed,epochs CSV for a subset experiment')
    p.add_argument('--allow-incomplete',action='store_true')
    args=p.parse_args()
    df,failures=collect(args.results, args.expected_sweep)
    if df.empty:
        raise RuntimeError('No successful simulations found')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(simulations=len(df),issues=failures,source=str(args.results.resolve()))
    args.output.with_suffix('.coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    if failures and not args.allow_incomplete:
        raise RuntimeError('Incomplete results; inspect coverage report. Use --allow-incomplete explicitly.')
    df.sort_values(['rhm_m','rhm_Ntrain','rhm_seed_rules']).to_parquet(args.output,index=False)
    print(f'Wrote {len(df)} measurements to {args.output}; {len(failures)} coverage issues')

if __name__=='__main__':
    main()
