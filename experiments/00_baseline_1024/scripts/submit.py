"""Submit only verified models. Do not blindly resubmit after ambiguous acknowledgement."""
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from llm_evals.checkpointing import atomic, locked
root = Path(__file__).resolve().parents[1]
models = yaml.safe_load((root / 'configs/models.yaml').read_text())
state = root / 'outputs/runs/submissions.json'
with locked(root / 'outputs/runs/submission.lock'):
    saved = json.loads(state.read_text()) if state.exists() else {}
    for key, model in models.items():
        if not model['ready']:
            print(f'{key}: blocked: {model["problems"]}')
            continue
        if key in saved:
            print(f'{key}: already submitted; inspect {saved[key]} before explicit resubmission')
            continue
        saved[key] = {'state': 'submission_pending'}
        atomic(state, saved)
        env = dict(os.environ, PROJECT_ROOT=str(root), MODEL_ID=key)
        out = subprocess.run(['sbatch', '--parsable', '--output=' + str(root / 'outputs/runs/%x-%j.out'),
                              '--error=' + str(root / 'outputs/runs/%x-%j.err'),
                              str(root / 'scripts/slurm/worker.sh')], env=env,
                             capture_output=True, text=True, timeout=45)
        if out.returncode:
            saved[key] = {'state': 'rejected', 'stderr': out.stderr}
        else:
            saved[key] = {'state': 'accepted', 'scheduler_acknowledgement': out.stdout.strip()}
        atomic(state, saved)
        print(key, saved[key])
