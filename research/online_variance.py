#!/usr/bin/env python3
"""Score online pawn-key error tables (OV lines from the research/online-variance engine) against
the fixed risk heads: out-of-sample R2 with coefficients fitted on half the games.
Usage: python3 online_variance.py DIR   (DIR holds ov0.txt .. ov7.txt from the engine stderr)"""
import sys,glob,numpy as np
rows=[];gid=-1
for p,fn in enumerate(sorted(glob.glob(sys.argv[1]+'/ov[0-9].txt'))):
    for l in open(fn):
        if l.startswith('OVGAME'): gid+=1; continue
        if l.startswith('OV '): rows.append([gid]+[float(x) for x in l.split()[1:]])
R=np.array(rows); g,dep,bnd,ply,eRaw,eCor,dn,up,sgn,mag,cnt=R.T
print(f"{len(R)} logged nodes, {len(np.unique(g))} games; bounds: upper {np.mean(bnd==0):.0%}, exact {np.mean(bnd==1):.1%}, lower {np.mean(bnd==2):.0%}")
rng=np.random.default_rng(0); games=np.unique(g); test=np.isin(g,rng.choice(games,len(games)//2,replace=False)); train=~test
def r2(X,y):
    X=np.c_[np.ones(len(y)),X]; b=np.linalg.lstsq(X[train],y[train],rcond=None)[0]; e=y[test]-X[test]@b
    return 1-e.var()/y[test].var()
D=np.c_[np.minimum(dep,16),bnd==1,bnd==2]
seen=(cnt>0).astype(float)
print("\nCONTROL (signed error of raw eval; correction history's job):")
y=np.clip(eRaw,-400,400); sfc=eRaw-eCor
print(f"  SF's own correction history     R2 {r2(np.c_[sfc],y):.3f}")
print(f"  our online pawn table           R2 {r2(np.c_[sgn,seen],y):.3f}   (nodes with a seen slot: {seen.mean():.0%})")
print(f"  both                            R2 {r2(np.c_[sfc,sgn,seen],y):.3f}")
print("\nCANDIDATE (log |error| of corrected eval):")
y=np.log(np.abs(eCor)+10); m=np.where(cnt>0,mag,0)
base=r2(np.c_[D],y); heads=r2(np.c_[D,dn,up],y); tab=r2(np.c_[D,m,seen],y); both=r2(np.c_[D,dn,up,m,seen],y)
print(f"  depth+bound only                R2 {base:.3f}")
print(f"  + fixed heads                   R2 {heads:.3f}")
print(f"  + online magnitude table        R2 {tab:.3f}")
print(f"  + heads + online table          R2 {both:.3f}   (gain over heads {both-heads:+.3f})")
for lo,hi in [(1,3),(3,10),(10,50),(50,1e9)]:
    sel=(cnt>=lo)&(cnt<hi)
    if sel.sum()<2000: continue
    tr,te=train&sel,test&sel
    def r2s(X):
        X=np.c_[np.ones(len(y)),X]; b=np.linalg.lstsq(X[tr],y[tr],rcond=None)[0]; e=y[te]-X[te]@b; return 1-e.var()/y[te].var()
    print(f"    slot seen {lo}-{hi:<5g} n={sel.sum():7d}: heads {r2s(np.c_[D,dn,up]):.3f} -> +table {r2s(np.c_[D,dn,up,mag]):.3f}")

print("\nROBUSTNESS:")
y=np.log(np.abs(eCor)+10)
def r2m(X,sel):
    X=np.c_[np.ones(len(y)),X]; tr,te=train&sel,test&sel
    b=np.linalg.lstsq(X[tr],y[tr],rcond=None)[0]; e=y[te]-X[te]@b; return 1-e.var()/y[te].var()
ex=bnd==1
print(f"  exact-score nodes only (n={ex.sum()}): heads {r2m(np.c_[np.minimum(dep,16),dn,up],ex):.3f} -> +table {r2m(np.c_[np.minimum(dep,16),dn,up,m,seen],ex):.3f}")
allsel=np.ones(len(y),bool); bias=np.c_[np.abs(sgn),np.abs(eRaw-eCor),np.abs(sgn-(eRaw-eCor))]
print(f"  baseline with bias-size features: heads+bias {r2m(np.c_[D,dn,up,bias],allsel):.3f} -> +table {r2m(np.c_[D,dn,up,bias,m,seen],allsel):.3f}")
pl=np.minimum(ply,40)
print(f"  baseline with ply too:            {r2m(np.c_[D,dn,up,bias,pl],allsel):.3f} -> +table {r2m(np.c_[D,dn,up,bias,pl,m,seen],allsel):.3f}")
# per-game consistency of the gain
gains=[]
for gg in np.unique(g[test]):
    sel=(g==gg)
    Xa=np.c_[np.ones(len(y)),D,dn,up]; Xb=np.c_[Xa,m,seen]
    ba=np.linalg.lstsq(Xa[train],y[train],rcond=None)[0]; bb=np.linalg.lstsq(Xb[train],y[train],rcond=None)[0]
    ea=((y[sel]-Xa[sel]@ba)**2).mean(); eb=((y[sel]-Xb[sel]@bb)**2).mean(); gains.append(1-eb/ea)
gains=np.array(gains); print(f"  per test game MSE reduction: median {np.median(gains):.3f}, positive in {np.mean(gains>0):.0%} of {len(gains)} games")
