import numpy as np
D=8; N=4096; DR=2048
WIN=[(0,512),(1536,2048),(2048,2560),(3584,4096)]
def streamA(seed):
    r=np.random.default_rng(7000+seed); X=r.uniform(-1,1,(N,D)); t=np.arange(N)
    y1=np.sin(3*X[:,0])*np.cos(2*X[:,1])+0.5*X[:,2]*X[:,3]
    y2=np.cos(3*X[:,1])*np.sin(2*X[:,0]+1)+0.5*X[:,4]*X[:,5]
    return X,np.where(t<DR,y1,y2)+r.normal(0,0.05,N)
def streamB(seed):
    r=np.random.default_rng(8000+seed); X=r.uniform(-1,1,(N,D)); t=np.arange(N)
    def vec():
        v=np.zeros(D); v[r.choice(D,3,replace=False)]=r.normal(size=3); return v/np.linalg.norm(v)*1.5
    a1,b1,a2,b2=vec(),vec(),vec(),vec()
    y1=np.tanh(2*X@a1)*np.sin(2*X@b1); y2=np.tanh(2*X@a2)*np.sin(2*X@b2)
    return X,np.where(t<DR,y1,y2)+r.normal(0,0.05,N)
