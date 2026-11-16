from h import *
import numpy_financial as npf
A=prop("A",200000,600,furn=0,copro=0,tf=0)   # loan 194400, interest ~6.7k > rent 7.2k? check
B=prop("B",50000,900,furn=0,copro=0,tf=0,loan=0)  # cash
df,b=run(strat(A,B),3)
for r in b["fiscal_annuel"]: print({k:round(v) for k,v in r.items()})
# hand: A R = 7200 - int - ins ; B R = 10800 - CFE(y>=2: 300)
