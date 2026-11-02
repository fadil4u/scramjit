import sys,time
sys.path.insert(0,'/mnt')
import numpy as np, envlab
c,f=envlab.parse_env(open('/mnt/dev/keydoor.py').read())
run=envlab.load_run(envlab.build_source(f))
lab=envlab.Lab(c,run)
t=time.perf_counter(); lab.go(1,1000,False); print('compile + first call s',round(time.perf_counter()-t,2),flush=True)
lab=envlab.Lab(c,run)
print('random return',round(lab.go(300000,1000,False)[0],3),flush=True)
t0=time.perf_counter(); tt=0
for i in range(10):
    t=time.perf_counter(); lab.go(1000000,max(30,200-20*i),True,alpha=max(0.02,0.3*0.7**i)); tt+=time.perf_counter()-t
    g,e=lab.go(100000,30,False)
    print(f'{i+1}M steps {time.perf_counter()-t0:.2f}s greedy {g:.3f}',flush=True)
print(f'pyodide steps/s {10000000/tt:,.0f}',flush=True)
plain=envlab.load_run(envlab.build_source(f,compiled=False),'plain')
a=envlab.Lab(c,run); b=envlab.Lab(c,plain); n=5000
a.go(n,100,True); t=time.perf_counter(); b.go(n,100,True); py=time.perf_counter()-t
print(f'pyodide plain python {n/py:,.0f} steps/s; identical {np.array_equal(a.q,b.q) and a.seed==b.seed}',flush=True)
