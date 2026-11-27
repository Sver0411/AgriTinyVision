"""Evaluate fixed sensor decisions under measured rural link snapshots and explicit weak-link stress cases."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import numpy as np
from joint_budget import OUT,table,write

URL='https://zenodo.org/records/16919567/files/data_forestry.xlsx?download=1'
MD5='e3b36c7df1ac82e6034feda882e33ad0'


def field_profiles(cache):
    if not cache.exists():
        with urllib.request.urlopen(URL,timeout=120) as r:cache.write_bytes(r.read())
    assert hashlib.md5(cache.read_bytes()).hexdigest()==MD5
    ns={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(cache) as archive:
        root=ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
    profiles=[]
    for row in root.findall('.//x:sheetData/x:row',ns)[1:]:
        v={c.attrib['r'].rstrip('0123456789'):float(c.find('x:v',ns).text) for c in row.findall('x:c',ns) if c.find('x:v',ns) is not None}
        if not all(k in v for k in 'FGHI'):continue
        profiles.append(dict(name=f"forest_{v['H']:g}m",series='COMMECT_forestry_snapshot',
                             up_mbps=v['G'],down_mbps=v['F'],rtt_ms=v['I'],distance_m=v['H']))
    assert len(profiles)==9
    return profiles


def busy_time(profile, jpeg_sizes, identifiers, cold=False, cloud_s=.1):
    # Measured application goodput already incorporates transport overheads; don't add them again.
    rtt=profile['rtt_ms']/1000
    up=jpeg_sizes+8
    back=12
    rtts=3 if cold else 1
    value=up*8/(profile['up_mbps']*1e6)+back*8/(profile['down_mbps']*1e6)+rtts*rtt+cloud_s
    if profile['series']=='synthetic':
        # One round of selective repeat per failed segment. All requests have a deterministic random stream.
        for i,(uid,n) in enumerate(zip(identifiers,up)):
            seed=int.from_bytes(hashlib.sha256(('field-link-v1/'+uid).encode()).digest()[:8],'little')
            rng=np.random.default_rng(seed)
            pieces=np.r_[np.full((int(n)-1)//512,512),((int(n)-1)%512)+1,back]
            tries=np.minimum(rng.geometric(.9,len(pieces)),5)
            # Ceiling at five attempts; callers time out any permanently undelivered transaction.
            if np.any((tries==5)&(rng.random(len(tries))<.1)):
                value[i]=np.inf
                continue
            original=np.r_[np.full(len(pieces)-1,profile['up_mbps']),profile['down_mbps']]
            value[i]+=np.sum(8*((tries-1)*pieces+tries*52)/(1e6*original))+np.sum(tries-1)*rtt
    return value


def service(order,ask,length,interval,deadline,outages):
    served=np.zeros(len(ask),dtype=bool)
    delays=[];free=0.
    for frame,i in enumerate(order):
        if not ask[i]:continue
        arrival=frame*interval; start=max(arrival,free)
        remaining=length[i]
        if start<arrival+deadline and np.isfinite(remaining):
            end=start
            if outages:
                # 60 s of each 120-s phase is disconnected. Pause a pending transfer during a gap.
                while remaining>1e-10 and end<arrival+deadline:
                    phase=end%120.
                    if phase>=60.:
                        end+=120.-phase
                    else:
                        step=min(60.-phase,remaining);end+=step;remaining-=step
            else:
                end+=remaining;remaining=0.
            served[i]=remaining<=1e-10 and end-arrival<=deadline
            free=min(end,arrival+deadline)
        elif start<arrival+deadline:
            free=arrival+deadline
        delays.append(free-arrival if served[i] else deadline)
    return served,np.array(delays)


def main(cache):
    rural=field_profiles(cache)
    write('rural_measured_kpis.csv',rural)
    profiles=rural+[dict(name='constrained_loss_outage',series='synthetic',up_mbps=.01,down_mbps=.02,rtt_ms=1200,distance_m='')]
    settings=[(2.,5.,False,.1),(30.,5.,False,.1),(2.,5.,True,.5),(30.,60.,True,.5)]
    rows=[]; setup=[]
    runs=table('joint_budget_runs.csv')
    for site in ('cornell','kashmir','latvia'):
        site_rows=[r for r in runs if r['site']==site]
        bykey={(int(r['seed']),r['method'],int(r['cap_percent']),r['review_location']):i for i,r in enumerate(site_rows)}
        with np.load(OUT/f'joint_budget_{site}.npz') as z:
            ids,y,preds,asks,costs=z['source_path'],z['y'],z['predictions'].astype(bool),z['queries'].astype(bool),z['application_bytes']
            order=np.argsort([hashlib.sha256(('field-link-stream/'+p).encode()).digest().hex() for p in ids])
            for profile in profiles:
                for deadline,interval,cold,compute in settings:
                    lengths=busy_time(profile,costs-20,ids,cold=cold,cloud_s=compute)
                    for seed in range(20):
                        for where in ('local','remote'):
                            j=bykey[seed,'joint',25,where]
                            a=bykey[seed,'update_only',25,where]
                            ask=asks[j];baseline=preds[a]
                            done,delay=service(order,ask,lengths,interval,deadline,profile['series']=='synthetic')
                            decision=np.where(done,preds[j],baseline)
                            fp=int(((y==0)&decision).sum());fn=int(((y==1)&~decision).sum())
                            rows.append(dict(site=site,seed=seed,reviewer=where,profile=profile['name'],series=profile['series'],
                                             deadline_s=deadline,acquisition_interval_s=interval,cold_connection=int(cold),
                                             assumed_cloud_seconds=compute,requested=int(ask.sum()),timely=int(done.sum()),
                                             fallback_count=int(ask.sum()-done.sum()),fp=fp,fn=fn,mean_wait_s=float(delay.mean()),
                                             p95_wait_s=float(np.quantile(delay,.95)),balanced_accuracy=1-.5*(fp/(y==0).sum()+fn/(y==1).sum())))
    write('rural_deadline_runs.csv',rows)
    summaries=[]
    keys=('site','reviewer','profile','series','deadline_s','acquisition_interval_s','cold_connection','assumed_cloud_seconds')
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    for key,rr in groups.items():
        summaries.append(dict(zip(keys,key),**{k:float(np.mean([r[k] for r in rr])) for k in
           ('requested','timely','fallback_count','balanced_accuracy','mean_wait_s','p95_wait_s')}))
    write('rural_deadline_summary.csv',summaries)
    for r in table('edge_adaptation_metrics.csv'):
        if r['budget']!='128' or r['method']!='local_int8':continue
        for who in ('local','remote'):
            gross=int(r[f'{who}_label_total_bytes'])
            for profile in profiles:
                # Optimistic bulk service excluding stalls; human effort is separate and explicit.
                uplink=gross-int(r['head_downlink_bytes'])
                seconds=8*(uplink/profile['up_mbps']+int(r['head_downlink_bytes'])/profile['down_mbps'])/1e6
                for human in (0,15,60):
                    setup.append(dict(seed=r['seed'],reviewer=who,profile=profile['name'],setup_bytes=gross,
                                      ideal_transfer_s=seconds,assumed_seconds_per_label=human,
                                      sequential_128_label_labor_s=128*human))
    write('rural_setup_cost.csv',setup)
    print('Rural site snapshots',len(rural),'deadline replays',len(rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source_xlsx',type=Path)
    main(parser.parse_args().source_xlsx)
