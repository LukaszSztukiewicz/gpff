"""Recompute consistent ratios/CIs from raw trials. Run only on Hydra."""
import csv
import json
from pathlib import Path
import statistics
import numpy as np

ROOT=Path('/home/lsztuk/gpff/scratch/speedup-audit-20261029')
summary=[]
for path in sorted((ROOT/'results').glob('*.json')):
    rows=json.loads(path.read_text())
    env=next((x['environment'] for x in rows if 'environment' in x),{})
    for row in rows:
        if 'before_samples_ms' in row:
            a=np.array(row['before_samples_ms'])
            b=np.array(row['after_samples_ms'])
            indices=np.random.default_rng(999).integers(0,len(a),(10000,len(a)))
            ratios=np.median(a[indices],axis=1)/np.median(b[indices],axis=1)
            point=float(np.median(a)/np.median(b))
            summary.append({'file':path.name,'gpu':env.get('gpu'),'job':env.get('job'),
                'test':row['test'],'before_ms':float(np.median(a)),'after_ms':float(np.median(b)),
                'speedup':point,'time_reduction_pct':100*(1-1/point),
                'ci95':np.quantile(ratios,[.025,.975]).tolist(),
                'batch':row.get('batch',env.get('args',{}).get('batch')),
                'width':row.get('width',env.get('args',{}).get('width')),
                'compiled':row.get('compiled',env.get('args',{}).get('compile',False))})
        elif 'samples_ms' in row:
            summary.append({'file':path.name,'gpu':env.get('gpu'),'job':env.get('job'),
                'test':row['test'],'median_ms':statistics.median(row['samples_ms']),
                'mean_ms':statistics.mean(row['samples_ms']),
                'first_call_ms':row.get('first_call_ms'),'batch':row.get('batch'),
                'width':row.get('width'),'compiled':row.get('compiled',False)})
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (ROOT/'summary.csv').open('w',newline='') as f:
    fields=['file','gpu','job','test','before_ms','after_ms','speedup','time_reduction_pct','ci95','median_ms','mean_ms','first_call_ms','batch','width','compiled']
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    w.writerows(summary)
print(json.dumps(summary,indent=2))
