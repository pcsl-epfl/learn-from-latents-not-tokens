"""One training + validation-selected classifier task from the published sweep."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent / 'gradient_ilc'

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--sweep', type=Path, default=ROOT / 'simulation/gradient_sweep.csv')
    parser.add_argument('--max-training-seconds', type=float, default=0)
    parser.add_argument('--checkpoint-min-step', type=int, default=0)
    parser.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '1')))
    parser.add_argument('--smoke', action='store_true', default=os.environ.get('SMOKE', '0') == '1')
    parser.add_argument('--device', default=os.environ.get('DEVICE', 'cuda'))
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data/gradient_ilc')
    args = parser.parse_args()
    rows = list(csv.reader(args.sweep.open()))
    if not 1 <= args.task <= len(rows):
        parser.error(f'task must be in 1..{len(rows)}')
    m, p, seed, epochs = map(int, rows[args.task-1])
    v, L, ntest, val, width, hidden = 10, 4, 2000, 3000, 128, 150
    if args.smoke:
        v, m, L, p, epochs, ntest, val, width, hidden = 4, 2, 3, 32, 1, 16, 16, 8, 12
    out = args.output_root.resolve() / ('smoke' if args.smoke else 'results') / f'task_{args.task:03d}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'observations.pt').exists():
        raise FileExistsError(f'Refusing to overwrite an existing run: {out}')
    flags = ['--rhm_L',str(L),'--rhm_m',str(m),'--rhm_Ntrain',str(p),
             '--rhm_seed_rules',str(seed),'--Nepoch',str(epochs),'--rhm_v',str(v),
             '--rhm_s','3','--rhm_Ntest',str(ntest),'--rhm_seed_sample','57823',
             '--cache_on_device','--ema_alpha','0.985','--lr','3e-3','--seed_torch','42',
             '--wd','1e-3','--use_autojac','--token_dimension',str(width),'--ptc',
             '--ptc_hidden_dim',str(hidden),'--ptc_cluster_pdfs',
             '--ptc_cluster_similarity_margin','0.8','--ptc_cluster_similarity_penalty','0.5',
             '--ptc_cluster_subtract_mean','--ptc_cluster_sparsity_reward','1e-2',
             '--observe_checkpoint_step_linear_N','10','--observe_checkpoint_step_min','1000000',
             '--observe_N_batches','8','--ptc_cluster_maximum_comparisons','300',
             '--device',args.device,'--output_filename_prefix',str(out)+'/']
    if args.max_training_seconds > 0:
        flags += ['--max_training_seconds', str(args.max_training_seconds)]
    if args.checkpoint_min_step > 0:
        flags += ['--observe_checkpoint_step_min', str(args.checkpoint_min_step),
                  '--observe_checkpoint_step_geom_factor', '2.0',
                  '--observe_checkpoint_step_linear_N', '0']
    commands = [[sys.executable,str(SOURCE/'run.py'),*flags],
                [sys.executable,str(SOURCE/'run_checkpoint_classifiers.py'),
                 '--simulation_dir',str(out),'--device',args.device,
                 '--classifier_batch_size','128','--representation_batch_size','512',
                 '--validation_size',str(val),'--validation_seed','519823','--classifier_seed','42']]
    if args.smoke:
        commands[1] += ['--classifier_epochs','1']
    revision = subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True)
    (out/'run_metadata.json').write_text(json.dumps(dict(task=args.task,smoke=args.smoke,
                git_sha=revision.stdout.strip(),commands=commands),indent=2)+'\n')
    for command in commands:
        print('Running:',*command,flush=True)
        subprocess.run(command,cwd=SOURCE,check=True)
    print(f'Completed training and classifier: {out}')

if __name__ == '__main__':
    main()
