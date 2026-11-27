"""Compare an independently compiled C99 EQ01 reader against released test predictions."""
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import platform
import struct
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(exe, head, rows, count, check=True):
    return subprocess.run([str(exe), str(head), str(rows), str(count)], capture_output=True, check=check)


def boundary_tests(exe, head, rows, d):
    q = np.full(d, 127, np.int8)
    header = struct.pack('<4sHff', b'EQ01', d, 0.5, 0.0)
    head.write_bytes(header + q.tobytes())
    rows.write_bytes(np.stack([q, -q, q * 0]).tobytes())
    obs = np.loadtxt(io.BytesIO(run(exe, head, rows, 3).stdout), ndmin=2)
    limit = d * 127 * 127
    assert obs[:,0].tolist()==[limit,-limit,0] and obs[:,1].tolist()==[limit,limit,0]
    assert obs[:,3].tolist()==[1,0,1] and obs[2,2]==0
    valid_head=head.read_bytes(); valid_row=q.tobytes()
    bad = {
        'bad_magic':(b'BAD!' + valid_head[4:],valid_row,1),
        'little_endian_dimension_swapped':(valid_head[:4] + valid_head[4:6][::-1] + valid_head[6:],valid_row,1),
        'truncated_head':(valid_head[:-1],valid_row,1), 'extra_head_byte':(valid_head+b'0',valid_row,1),
        'truncated_row':(valid_head,valid_row[:-1],1), 'extra_row_byte':(valid_head,valid_row+b'0',1),
        'invalid_weight_128':(valid_head[:-1]+b'\x80',valid_row,1),
        'invalid_activation_128':(valid_head,valid_row[:-1]+b'\x80',1),
        'nonfinite_scale':(struct.pack('<4sHff',b'EQ01',d,float('nan'),0)+valid_row,valid_row,1),
        'zero_declared_rows':(valid_head,b'',0),
    }
    for name,(h,x,n) in bad.items():
        head.write_bytes(h); rows.write_bytes(x)
        if run(exe,head,rows,n,check=False).returncode==0:
            raise ValueError('Malformed case accepted: '+name)
    return {'positive_and_negative_extreme_dot':limit,'zero_row_score':0.,'zero_row_tie_prediction':1,'malformed_inputs_rejected':list(bad)}


def audit(args):
    root=args.root; predictions=root/'results/edge_adaptation_predictions.npz'; heads=root/'results/edge_adaptation_heads.zip'
    activations=root/'data/features/edge_fgvc.npz'; source=Path(__file__).with_name('packed_head.c')
    with np.load(predictions,allow_pickle=False) as p, np.load(activations,allow_pickle=False) as a:
        paths=list(p['test_paths']); lookup={s:i for i,s in enumerate(a['source_path'])};assert len(paths)==len(set(paths))
        x=a['activation'][[lookup[s] for s in paths]]; assert x.dtype==np.float32 and np.isfinite(x).all()
        largest=np.max(np.abs(x),axis=1); denom=np.ones_like(largest); active=largest>0
        denom[active]=largest[active]/np.float32(127.)
        q=np.clip(np.rint(x/denom[:,None]),-127,127).astype(np.int8); d=q.shape[1]
        method=list(p['methods']).index('local_int8'); refs=p['score'][:,:,method,:]; labels=p['pred'][:,:,method,:]; seeds=p['seeds']; budgets=p['budgets']
    x64=q.astype(np.int64); norms=np.sum(x64**2,axis=1,dtype=np.int64)
    maxerr=0.; dot_max=0; rows_checked=0; mismatch=0; per_budget={str(int(b)):0 for b in budgets}
    with TemporaryDirectory() as tmp, ZipFile(heads) as z:
        temp=Path(tmp); exe=temp/'packed_head'; rows=temp/'rows.i8'; head=temp/'head.bin'
        subprocess.run([args.cc,'-O2','-std=c99','-Wall','-Wextra','-Werror','-pedantic',str(source),'-o',str(exe),'-lm'],check=True)
        bounds=boundary_tests(exe,head,rows,d); rows.write_bytes(q.tobytes())
        for si,seed in enumerate(seeds):
            for bi,budget in enumerate(budgets):
                blob=z.read(f's{seed:02d}-b{budget:03d}/local_int8.bin'); head.write_bytes(blob)
                assert len(blob)==14+d and blob[:4]==b'EQ01' and int.from_bytes(blob[4:6],'little')==d
                out=np.loadtxt(io.BytesIO(run(exe,head,rows,len(q)).stdout),ndmin=2)
                dots=x64@np.frombuffer(blob,np.int8,offset=14).astype(np.int64)
                if not np.array_equal(out[:,0],dots) or not np.array_equal(out[:,1],norms):
                    raise ValueError(f'Accumulator mismatch in seed {seed}, budget {budget}')
                maxerr=max(maxerr,float(np.max(np.abs(out[:,2]-refs[si,bi]))));dot_max=max(dot_max,int(np.abs(dots).max()))
                mismatch+=int(np.sum(out[:,3]!=labels[si,bi]));rows_checked+=len(q);per_budget[str(int(budget))]+=len(q)
        expected=len(q)*len(seeds)*len(budgets)
        if rows_checked!=expected or mismatch or maxerr>1e-12: raise ValueError('C99 replay mismatch')
    report=dict(scope='Host C99 final-head consistency on saved features; no new device execution',compiler=subprocess.check_output([args.cc,'--version'],text=True).splitlines()[0],
                arch=platform.machine(),input_dimension=d,head_bytes=14+d,n_test_vectors=len(q),seeds=seeds.tolist(),budgets=budgets.tolist(),
                n_int8_heads=len(seeds)*len(budgets),n_test_decisions=rows_checked,counts_per_budget=per_budget,integer_dot_mismatches=0,
                norm_squared_mismatches=0,label_mismatches=mismatch,max_abs_score_difference=maxerr,max_observed_abs_dot=dot_max,
                max_observed_norm_squared=int(norms.max()),theoretical_max_abs_dot=d*127**2,theoretical_max_norm_squared=d*127**2,
                theoretical_bound_for_largest_wire_dimension=65535*127**2,int32_max=int(np.iinfo(np.int32).max),boundary_cases=bounds,
                sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in [source,heads,predictions,activations]})
    args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:report[k] for k in ('n_int8_heads','n_test_decisions','label_mismatches','max_abs_score_difference','max_observed_abs_dot','max_observed_norm_squared')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--output',type=Path,default=ROOT/'results/c99_head_audit.json');parser.add_argument('--cc',default='cc');audit(parser.parse_args())
