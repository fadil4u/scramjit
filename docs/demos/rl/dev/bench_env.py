"""The measurement that decides the demo: one environment, trained the same way, four ways."""
import json,os,subprocess,sys,tempfile,time
HERE=os.path.dirname(os.path.abspath(__file__))
RL=os.path.dirname(HERE)
sys.path[:0]=[RL,os.path.join(RL,'..','..','py')]
import numpy as np, envlab
path=sys.argv[1]
c,f=envlab.parse_env(open(path).read())
# compiled
t=time.perf_counter(); fast=envlab.load_run(envlab.build_source(f)); lab=envlab.Lab(c,fast); lab.go(1,1000,False); comp=time.perf_counter()-t
print(f'compile + first call: {comp:.2f} s',flush=True)
# identical results: same seed, fresh tables
plain=envlab.load_run(envlab.build_source(f,compiled=False))
a=envlab.Lab(c,fast); b=envlab.Lab(c,plain)
N=20000
ra=a.go(N,100,True); 
t=time.perf_counter(); rb=b.go(N,100,True); py=time.perf_counter()-t
same=np.array_equal(a.q,b.q) and a.seed==b.seed and ra==rb
print(f'plain Python: {N/py:,.0f} steps/s; Q table, rng and returns identical to compiled: {same}',flush=True)
# best of 3, interleaved against nothing else
best=0
for _ in range(3):
    l=envlab.Lab(c,fast); t=time.perf_counter(); l.go(2000000,100,True); best=max(best,2000000/(time.perf_counter()-t))
print(f'scramjit native: {best:,.0f} steps/s ({best*py/N:,.0f}x Python)',flush=True)
# per-episode fresh process: the cheapest thing that could be called a sandbox
src=envlab.build_source(f,compiled=False).replace('import math\n','import math\n',1)
d=tempfile.mkdtemp(); p=d+'/ep.py'
open(p,'w').write(src+f'''
import numpy as np
c={ {k:c[k] for k in ('W','H','ACTIONS','MAX_STEPS','OBS')} }
q=np.zeros((1<<10,c['ACTIONS'])); s=np.zeros(32,np.int64); g=np.zeros((c['H'],c['W']),np.int64); o=np.zeros(c['OBS'],np.int64); r=np.zeros(1024,np.int64); st=np.zeros(4)
run(q,s,g,o,r,st,c['W'],c['H'],c['ACTIONS'],1024,(1<<10)-1,c['MAX_STEPS'],c['MAX_STEPS'],7,100,0.2,0.99,True)
''')
t=time.perf_counter(); n=5
for _ in range(n): subprocess.run([sys.executable,'-I',p],check=True)
ep=(time.perf_counter()-t)/n
sb=c['MAX_STEPS']/ep
print(f'fresh process per episode ({c["MAX_STEPS"]} steps): {ep*1000:.0f} ms/episode = {sb:,.0f} steps/s ({best/sb:,.0f}x slower than scramjit); full episodes, so this is the best case',flush=True)
# learning
print(f'random return ...',flush=True)
l=envlab.Lab(c,fast); print(' random policy:',round(l.go(200000,1000,False)[0],3))
t=time.perf_counter(); tot=0
for ch in range(8):
    l.go(1000000,100,True); tot+=1000000
    g,e=l.go(100000,20,False)
    print(f' {tot/1e6:.0f}M steps, {time.perf_counter()-t:.2f} s: greedy return {g:.3f} ({e} episodes)',flush=True)
