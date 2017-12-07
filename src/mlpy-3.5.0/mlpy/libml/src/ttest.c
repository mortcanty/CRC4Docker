#include <math.h>
#include <stdio.h>

static double betai(double a,double b,double x);
static double betacf(double a,double b,double x);
static  double gammln(double xx);


int ttest(double *data1,int n1,double *data2,int n2,double *t,double *prob)
{
  double var1,var2,df,ave1,ave2;
  int i;
  double dev;

  if(n1 <= 1){
    fprintf(stderr,"ttest: n1 must be > 1");
    return 0;
  }else{
    
    ave1=.0;
    for(i=0;i<n1;i++)
      ave1 += data1[i];
    ave1 /= n1;
    
    var1=.0;
    for(i=0;i<n1;i++){
      dev = data1[i]-ave1;
      var1 += dev*dev;
    }
    var1 /= (n1 - 1);
  }

  if(n2 <= 1){
    fprintf(stderr,"ttest: n2 must be > 1");
    return 0;
  }else{    
    ave2=.0;
    for(i=0;i<n2;i++)
      ave2 += data2[i];
    ave2 /= n2;

   var2=.0;
   for(i=0;i<n2;i++){
      dev = data2[i]-ave2;
      var2 += dev*dev;
   }
   var2 /= (n2 - 1);
  }

  *t=(ave1-ave2)/sqrt(var1/n1+var2/n2);

  df=(var1/n1+var2/n2)*(var1/n1+var2/n2)/((var1/n1)*(var1/n1)/(n1-1)+
					  (var2/n2)*(var2/n2)/(n2-1));
  *prob=betai(0.5*df,0.5,df/(df+(*t)*(*t)));

  return 1;
}

static double betai(double a,double b,double x)
{
  double bt;
  
  if(x<0.0 || x> 1.0){
    fprintf(stderr,"WARNING: bad x in BETAI\n");
  }
  if(x==0.0 || x == 1.0){
    bt = 0.0;
  }else{
    bt=exp(gammln(a+b)-gammln(a)-gammln(b)+a*log(x)+b*log(1.0-x));
  }
  if(x< (a+1.0)/(a+b+2.0)){
    return bt*betacf(a,b,x)/a;
  }else{
    return 1.0-bt*betacf(b,a,1.0-x)/b;
  }
}


#define ITMAX 1000000
#define EPS 3.0e-7

static double betacf(double a,double b,double x)
{
  double qap,qam,qab,em,tem,d;
  double bz,bm=1.0,bp,bpp;
  double az=1.0,am=1.0,ap,app,aold;
  int m;

  qab=a+b;
  qap=a+1.0;
  qam=a-1.0;

  bz=1.0-qab*x/qap;
  
  for(m=1;m<=ITMAX;m++){
    em=(double)m;
    tem=em+em;
    d=em*(b-em)*x/((qam+tem)*(a+tem));
    ap=az+d*am;
    bp=bz+d*bm;
    d = -(a+em)*(qab+em)*x/((qap+tem)*(a+tem));
    app=ap+d*az;
    bpp=bp+d*bz;
    aold=az;
    am=ap/bpp;
    bm=bp/bpp;
    az=app/bpp;
    bz=1.0;
    if(fabs(az-aold)<(EPS*fabs(az)))
      return az;
  }
  fprintf(stderr,"WARNING: a or b too big, or ITMAX too small in BETACF\n");
}

#undef ITMAX
#undef EPS


static  double gammln(double xx)
{
  double x,tmp,ser;
  static double cof[6]={76.18009173,-86.50532033,24.01409822,
			-1.231739516,0.120858003e-2,-0.536382e-5};
  int j;

  x=xx-1.0;
  tmp=x+5.5;
  tmp -= (x+0.5)*log(tmp);
  ser=1.0;
  for(j=0;j<=5;j++){
    x += 1.0;
    ser += cof[j]/x;
  }
  return -tmp+log(2.50662827465*ser);
}
