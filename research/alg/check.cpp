#include "hgln_var.hpp"
#include "hgln.hpp"
#include <cstdio>
int main(){ for(int d:{8,64}) for(int nt=0;nt<2;nt++){ hv::Var V; V.norm_track=nt; if(d>16){V.glr0=V.glr1=0.02;} hv::Net a; a.init(d,7,V); hgln::Net<double> b; b.init(d,7,d>16?hgln::Params(0.02,0.02):hgln::Params());
 hv::Rng r(5); std::vector<double> x(d); double mx=0; for(int t=0;t<3000;t++){ for(auto&v:x) v=2*r.uni()-1; double y=std::sin(3*x[0])*std::cos(2*x[1]); double p1=a.predict(x.data()), p2=b.predict(x.data()); mx=std::max(mx,std::fabs(p1-p2)); a.learn(y); b.learn(y);} printf("d=%d norm_track=%d max |var - lib| = %.3g\n",d,nt,mx);} }
