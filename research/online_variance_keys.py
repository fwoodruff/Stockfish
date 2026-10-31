import sys,glob,numpy as np
rows=[];gid=-1
for fn in sorted(glob.glob(sys.argv[1]+'/ov[0-9].txt')):
    for l in open(fn):
        if l.startswith('OVGAME'): gid+=1
        elif l.startswith('OV2 '): rows.append([gid]+[float(x) for x in l.split()[1:]])
R=np.array(rows); g,dep,bnd,t,P,M,N,W,B,G1,G2,cnt=R.T
rng=np.random.default_rng(0); games=np.unique(g); test=np.isin(g,rng.choice(games,len(games)//2,replace=False)); train=~test
D=np.c_[np.minimum(dep,16),bnd==1,bnd==2]
def r2(*cols):
    X=np.c_[np.ones(len(t)),D,*cols] if cols else np.c_[np.ones(len(t)),D]
    b=np.linalg.lstsq(X[train],t[train],rcond=None)[0]; e=t[test]-X[test]@b; return 1-e.var()/t[test].var(), b
base=r2()[0]
print(f"{len(t)} updates, {len(games)} games; target = residual log error (engine units); depth+bound R2 {base:.3f}")
for name,cols in [("global fast",[G1]),("global slow",[G2]),("material key",[M]),("minor-piece key",[N]),
                  ("non-pawn keys (W+B)",[W,B]),("pawn key (current engine)",[P]),
                  ("pawn + global slow",[P,G2]),("pawn + material + minor",[P,M,N]),
                  ("pawn + non-pawn W+B",[P,W,B]),("all keys",[P,M,N,W,B]),("all keys + globals",[P,M,N,W,B,G1,G2]),
                  ("sum of pawn+minor+nonpawn (corrhist-style, 1 coef)",[P+N+W+B])]:
    r,b=r2(*cols); print(f"  {name:52s} R2 {r:.3f}  (+{r-base:.3f})")
