"""Minimum JPEG payload required for a fixed backend menu to match target balanced accuracy."""
from itertools import product
import numpy as np


def minimum_bytes(y, edge, menu, target_tn, target_tp):
    """menu contains paired (binary predictions, integer byte costs) arrays; return None if infeasible."""
    y=np.asarray(y);edge=np.asarray(edge); n0=int((y==0).sum()); n1=int((y==1).sum())
    if min(n0,n1)==0:raise ValueError('balanced accuracy requires both classes')
    tn=int(((y==0)&(edge==y)).sum()); tp=int(((y==1)&(edge==y)).sum())
    # Multiply the desired balanced accuracy difference by 2*n0*n1 to work in integers.
    needed=(target_tn-tn)*n1+(target_tp-tp)*n0
    if needed<=0:return 0
    costs=np.full(y.shape,np.iinfo(np.int64).max,dtype=np.int64)
    for pred,size in menu:
        valid=(edge!=y)&(np.asarray(pred)==y)
        costs[valid]=np.minimum(costs[valid],np.asarray(size)[valid])
    finite=costs<np.iinfo(np.int64).max
    prefixes=[np.r_[0,np.cumsum(np.sort(costs[finite&(y==cl)]))] for cl in (0,1)]
    candidates=[]
    for fixed0,b0 in enumerate(prefixes[0]):
        remaining=needed-fixed0*n1
        fixed1=max(0,(remaining+n0-1)//n0)
        if fixed1<len(prefixes[1]):
            candidates.append(int(b0+prefixes[1][fixed1]))
    return min(candidates) if candidates else None


def verify():
    rng=np.random.default_rng(117)
    for n in range(2,8):
        for case in range(12):
            y=np.r_[0,1,rng.integers(2,size=n-2)]
            edge=rng.integers(2,size=n)
            menu=[(rng.integers(2,size=n),rng.integers(1,80,size=n)) for _ in range(2)]
            n0=int((y==0).sum());n1=n-n0
            best={}
            for choice in product(range(3),repeat=n):
                pred=np.array([edge[i] if k==0 else menu[k-1][0][i] for i,k in enumerate(choice)])
                cost=sum(int(menu[k-1][1][i]) for i,k in enumerate(choice) if k)
                val=int(((pred==0)&(y==0)).sum())*n1+int(((pred==1)&(y==1)).sum())*n0
                best[val]=min(cost,best.get(val,cost))
            for tn in range(n0+1):
                for tp in range(n1+1):
                    brute=min((b for v,b in best.items() if v>=tn*n1+tp*n0),default=None)
                    assert minimum_bytes(y,edge,menu,tn,tp)==brute
    print('Exact oracle agreed with exhaustive multi-backend assignment on 72 finite pools.')


if __name__=='__main__':
    verify()
