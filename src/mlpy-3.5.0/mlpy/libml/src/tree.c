#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ml.h"


static void split_node(Node *node,Node *nodeL,Node *nodeR,int classes[],
		       int nclasses);
static double gini_index(double p[],int n);

static int compute_tree_bagging(ETree *etree,int n,int d,double *x[],
				int y[], int nmodels,int stumps, int minsize);
static int compute_tree_aggregate(ETree *etree,int n,int d,double *x[],int y[],
				  int nmodels,int stumps, int minsize);
static int compute_tree_adaboost(ETree *etree,int n,int d,double *x[],
				 int y[], int nmodels,int stumps, int minsize);

int  compute_tree(Tree *tree,int n,int d,double *x[],
		   int y[],int stumps,int minsize)
     /*
       compute tree model.x,y,n,d are the input data.
       stumps takes values  1 (compute single split) or 
       0 (standard tree). minsize is the minimum number of
       cases required to split a leaf.

       Return value: 0 on success, 1 otherwise.
     */
{
  int i,j;
  int node_class_index;
  int max_node_points;
  int cn;
  double sumpriors;

  tree->n=n;
  tree->d=d;


  if(stumps != 0 && stumps != 1){
    fprintf(stderr,"compute_tree: parameter stumps must be 0 or 1\n");
    return 1;
  }

  if(minsize < 0){
    fprintf(stderr,"compute_tree: parameter minsize must be >= 0\n");
    return 1;
  }

  tree->nclasses=iunique(y,tree->n, &(tree->classes));

  if(tree->nclasses<=0){
    fprintf(stderr,"compute_tree: iunique error\n");
    return 1;
  }

  if(tree->nclasses==1){
    fprintf(stderr,"compute_tree: only 1 class recognized\n");
    return 1;
  }

  if(tree->nclasses==2)
    if(tree->classes[0] != -1 || tree->classes[1] != 1){
      fprintf(stderr,"compute_tree: for binary classification classes must be -1,1\n");
      return 1;
    }

  if(tree->nclasses>2)
    for(i=0;i<tree->nclasses;i++)
      if(tree->classes[i] != i+1){
        fprintf(stderr,"compute_tree: for %d-class classification classes must be 1,...,%d\n",tree->nclasses,tree->nclasses);
        return 1;
      }


  
  if(!(tree->x=dmatrix(n,d))){
    fprintf(stderr,"compute_tree: out of memory\n");
    return 1;
  }
  if(!(tree->y=ivector(n))){
    fprintf(stderr,"compute_tree: out of memory\n");
    return 1;
  }
  for(i=0;i<n;i++){
    for(j=0;j<d;j++)
      tree->x[i][j]=x[i][j];
    tree->y[i]=y[i];
  }

  tree->stumps = stumps;
  tree->minsize = minsize;
  
  tree->node=(Node *)malloc(sizeof(Node));

  tree->node[0].nclasses=tree->nclasses;
  tree->node[0].npoints = tree->n;
  tree->node[0].nvar = tree->d;
  tree->node[0].data=tree->x;
  tree->node[0].classes=tree->y;
  

  tree->node[0].npoints_for_class=ivector(tree->nclasses);
  tree->node[0].priors=dvector(tree->nclasses);

  
  for(i=0;i<tree->node[0].npoints;i++){
    for(j = 0; j < tree->nclasses;j++)
      if(tree->classes[j]==tree->node[0].classes[i]){
	tree->node[0].npoints_for_class[j] += 1;
	break;
      }
  }

  node_class_index=0;
  max_node_points=0;
  for(j = 0; j < tree->nclasses;j++)
    if(tree->node[0].npoints_for_class[j] > max_node_points){
      max_node_points = tree->node[0].npoints_for_class[j];
      node_class_index = j;
    }
  tree->node[0].node_class = tree->classes[node_class_index];
  
  sumpriors=.0;
  for(j=0;j < tree->nclasses;j++)
    sumpriors += tree->node[0].npoints_for_class[j];
  for(j = 0; j < tree->nclasses;j++)
    tree->node[0].priors[j] = tree->node[0].npoints_for_class[j]/sumpriors;
  
  tree->node[0].terminal=TRUE;
  if(gini_index(tree->node[0].priors,tree->nclasses)>0)
    tree->node[0].terminal=FALSE;

  tree->nnodes=1;
  for(cn=0;cn<tree->nnodes;cn++)
    if(!tree->node[cn].terminal){
      tree->node[cn].left=tree->nnodes;
      tree->node[cn].right=tree->nnodes+1;
      tree->node=(Node *)realloc(tree->node,(tree->nnodes+2)*sizeof(Node));
      split_node(&(tree->node[cn]),&(tree->node[tree->nnodes]),
		 &(tree->node[tree->nnodes+1]),tree->classes,tree->nclasses);
      
      if(tree->minsize>0){
	if(tree->node[tree->nnodes].npoints < tree->minsize)
	  tree->node[tree->nnodes].terminal = TRUE;
	if(tree->node[tree->nnodes+1].npoints < tree->minsize)
	  tree->node[tree->nnodes+1].terminal = TRUE;
      }
      if(tree->stumps){
	tree->node[tree->nnodes].terminal = TRUE;
	tree->node[tree->nnodes+1].terminal = TRUE;
      }
      tree->nnodes += 2;
    }

  return 0;
  
}

int compute_etree(ETree *etree,int n,int d,double *x[], int y[],
		  int method,int nmodels,int stumps,int minsize)
    /*
       compute ensamble of tree models.x,y,n,d are the input data.
       method is one of BAGGING,AGGREGATE,ADABOOST.
       stumps takes values  1 (compute single split) or 
       0 (standard tree). minsize is the minimum number of
       cases required to split a leaf.

       Return value: 0 on success, 1 otherwise.
     */
{
  switch(method){
  case BAGGING:
    if(compute_tree_bagging(etree,n,d,x,y,nmodels,stumps,minsize) != 0){
      fprintf(stderr,"compute_etree: compute_tree_bagging error\n");
      return 1;
    }
    break;
  case AGGREGATE:
    if(compute_tree_aggregate(etree,n,d,x,y,nmodels,stumps,minsize) != 0){
      fprintf(stderr,"compute_etree: compute_tree_aggregate error\n");
      return 1;
    }
    break;
  case ADABOOST:
    if( compute_tree_adaboost(etree,n,d,x,y,nmodels,stumps,minsize) != 0){
      fprintf(stderr,"compute_etree: compute_tree_adaboost error\n");
      return 1;
    }
    break;
  default:
    fprintf(stderr,"compute_etree: ensamble method not recognized\n");
    return 1;
    break;
  }

  return 0;
}

int predict_tree(Tree *tree, double x[],double **margin)
     /* 
	predicts tree model on a test point x. Priors of each class
	in the terminal node are stored within  the array margin
	(an array of length tree->nclasses).

	Return value: the predicted value on success (-1 or 1 for
	binary classification; 1,...,nclasses in the multiclass case),
	0 on succes with non unique classification, -2 otherwise.

     */
{
  int i;
  int act_node;
  int max_post;
  int which_max_post;
  
  act_node=0;

  for(;;){
    if(tree->node[act_node].terminal){

      if(!((*margin)=dvector(tree->nclasses))){
	fprintf(stderr,"predict_tree: out of memory\n");
	return -2;
      }
      for(i=0;i<tree->nclasses;i++)
	(*margin)[i]=tree->node[act_node].priors[i];
      
      max_post=0.0;
      which_max_post=0;
      for(i=0;i<tree->nclasses;i++)
	if((*margin)[i]>max_post){
	  max_post=(*margin)[i];
	  which_max_post=i;
	}
      for(i=0;i<tree->nclasses;i++)
	if(i != which_max_post)
	  if((*margin)[i] == (*margin)[which_max_post])
	    return 0;

      return tree->node[act_node].node_class;
    }
    else{
      if(x[tree->node[act_node].var]<tree->node[act_node].value)
	act_node=tree->node[act_node].left;
      else
	act_node=tree->node[act_node].right;
    }
  }
  
  return -2;
}




int predict_etree(ETree *etree, double x[],double **margin)
     /*
       predicts tree model on a test point x. Proportions of neighbours
       for each class will be stored within the array margin 
       (an array of length tree->nclasses). 

       
       Return value: the predicted value on success (-1 or 1 for
       binary classification; 1,...,nclasses in the multiclass case),
       0 on succes with non unique classification, -2 otherwise.
     */
{
  int i,b;
  int pred;
  double *tmpmargin;
  double maxmargin;
  
  if(!((*margin)=dvector(etree->nclasses))){
    fprintf(stderr,"predict_etree: out of memory\n");
    return -2;
  }

  if(etree->nclasses==2){
    for(b=0;b<etree->nmodels;b++){
      pred=predict_tree(&(etree->tree[b]), x,&tmpmargin);
      if(pred < -1){
	fprintf(stderr,"predict_etree: predict_tree error\n");
	return -2;
      }
      if(pred==-1)
	(*margin)[0] += etree->weights[b];
      else if(pred==1)
	(*margin)[1] += etree->weights[b];
 
      free_dvector(tmpmargin);
    }

    if((*margin)[0] > (*margin)[1])
      return -1;
    else if((*margin)[0] < (*margin)[1])
      return 1;
    else
      return 0;
  }else{
    for(b=0;b<etree->nmodels;b++){
      pred=predict_tree(&(etree->tree[b]), x,&tmpmargin);
      if(pred < -1){
	fprintf(stderr,"predict_etree: predict_tree error\n");
	return -2;
      }
      
      if(pred>0)
	(*margin)[pred-1] += etree->weights[b];
      
      free_dvector(tmpmargin);
    }

    maxmargin=0.0;
    pred=0;
    for(i=0;i<etree->nclasses;i++)
      if((*margin)[i]>maxmargin){
	maxmargin=(*margin)[i];
	pred=i;
      }

    for(i=0;i<etree->nclasses;i++)
      if(i != pred)
	if((*margin)[i] == maxmargin)
	  return 0;

    return pred+1;
  }

  return -2;
  
}


static int compute_tree_bagging(ETree *etree,int n,int d,double *x[],
				int y[], int nmodels,int stumps, int minsize)
{
  int i,b;
  int *samples;
  double **trx;
  int *try;

  if(nmodels<1){
    fprintf(stderr,"compute_tree_bagging: nmodels must be greater than 0\n");
    return 1;
  }

 if(stumps != 0 && stumps != 1){
    fprintf(stderr,"compute_tree_bagging: parameter stumps must be 0 or 1\n");
    return 1;
  }

  if(minsize < 0){
    fprintf(stderr,"compute_tree_bagging: parameter minsize must be >= 0\n");
    return 1;
  }

  etree->nclasses=iunique(y,n, &(etree->classes));


  if(etree->nclasses<=0){
    fprintf(stderr,"compute_tree_bagging: iunique error\n");
    return 1;
  }
  if(etree->nclasses==1){
    fprintf(stderr,"compute_tree_bagging: only 1 class recognized\n");
    return 1;
  }

  if(etree->nclasses==2)
    if(etree->classes[0] != -1 || etree->classes[1] != 1){
      fprintf(stderr,"compute_tree_bagging: for binary classification classes must be -1,1\n");
      return 1;
    }
  
  if(etree->nclasses>2)
    for(i=0;i<etree->nclasses;i++)
      if(etree->classes[i] != i+1){
	fprintf(stderr,"compute_tree_bagging: for %d-class classification classes must be 1,...,%d\n",etree->nclasses,etree->nclasses);
	return 1;
      }

  if(!(etree->tree=(Tree *)calloc(nmodels,sizeof(Tree)))){
    fprintf(stderr,"compute_tree_bagging: out of memory\n");
    return 1;
  }
  etree->nmodels=nmodels;
  if(!(etree->weights=dvector(nmodels))){
    fprintf(stderr,"compute_tree_bagging: out of memory\n");
    return 1;
  }

  for(b=0;b<nmodels;b++)
    etree->weights[b]=1.0 / (double) nmodels;
  
  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_tree_bagging: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_tree_bagging: out of memory\n");
    return 1;
  }
  
  for(b=0;b<nmodels;b++){
    if(sample(n, NULL, n, &samples, TRUE,b)!=0){
       fprintf(stderr,"compute_tree_bagging: sample error\n");
       return 1;
    }

    for(i =0;i<n;i++){
      trx[i] = x[samples[i]];
      try[i] = y[samples[i]];
    }

    if(compute_tree(&(etree->tree[b]),n,d,trx,try,stumps,minsize)!=0){
      fprintf(stderr,"compute_tree_bagging: compute_tree error\n");
      return 1;
    }
    free_ivector(samples);

  }

  free(trx);
  free_ivector(try);
    
  return 0;

}



static int compute_tree_aggregate(ETree *etree,int n,int d,double *x[],int y[],
				  int nmodels,int stumps, int minsize)
{
  int i,b;
  int *samples;
  double **trx;
  int *try;
  int indx;

  if(nmodels<1){
    fprintf(stderr,"compute_tree_aggregate: nmodels must be greater than 0\n");
    return 1;
  }

  if(nmodels > n){
    fprintf(stderr,"compute_tree_aggregate: nmodels must be less than n\n");
    return 1;
  }

 if(stumps != 0 && stumps != 1){
    fprintf(stderr,"compute_tree_bagging: parameter stumps must be 0 or 1\n");
    return 1;
  }

  if(minsize < 0){
    fprintf(stderr,"compute_tree_bagging: parameter minsize must be >= 0\n");
    return 1;
  }

  etree->nclasses=iunique(y,n, &(etree->classes));

  if(etree->nclasses<=0){
    fprintf(stderr,"compute_tree_aggregate: iunique error\n");
    return 1;
  }
  if(etree->nclasses==1){
    fprintf(stderr,"compute_tree_aggregate: only 1 class recognized\n");
    return 1;
  }

  if(etree->nclasses==2)
    if(etree->classes[0] != -1 || etree->classes[1] != 1){
      fprintf(stderr,"compute_tree_aggregate: for binary classification classes must be -1,1\n");
      return 1;
    }
  
  if(etree->nclasses>2)
    for(i=0;i<etree->nclasses;i++)
      if(etree->classes[i] != i+1){
	fprintf(stderr,"compute_tree_aggregate: for %d-class classification classes must be 1,...,%d\n",etree->nclasses,etree->nclasses);
	return 1;
      }

  if(!(etree->tree=(Tree *)calloc(nmodels,sizeof(Tree)))){
    fprintf(stderr,"compute_tree_aggregate: out of memory\n");
    return 1;
  }
  etree->nmodels=nmodels;
  if(!(etree->weights=dvector(nmodels))){
    fprintf(stderr,"compute_tree_aggregate: out of memory\n");
    return 1;
  }

  for(b=0;b<nmodels;b++)
    etree->weights[b]=1.0 / (double) nmodels;
  
  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_tree_aggregate: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_tree_aggregate: out of memory\n");
    return 1;
  }
  
  if(sample(nmodels, NULL, n, &samples, TRUE,0)!=0){
    fprintf(stderr,"compute_tree_aggregate: sample error\n");
    return 1;
  }

  for(b=0;b<nmodels;b++){
  
    indx=0;
    for(i=0;i<n;i++)
      if(samples[i] == b){
	trx[indx] = x[i];
	try[indx++] = y[i];
      }

    if(compute_tree(&(etree->tree[b]),indx,d,trx,try,stumps,minsize)!=0){
      fprintf(stderr,"compute_tree_aggregate: compute_tree error\n");
      return 1;
    }

  }

  free_ivector(samples);
  free(trx);
  free_ivector(try);
    
  return 0;

}

static int compute_tree_adaboost(ETree *etree,int n,int d,double *x[],int y[],
				 int nmodels,int stumps, int minsize)
{
  int i,b;
  int *samples;
  double **trx;
  int *try;
  double *prob;
  double *prob_copy;
  double sumalpha;
  double eps;
  int *pred;
  double *margin;
  double sumprob;
  

  if(nmodels<1){
    fprintf(stderr,"compute_tree_adaboost: nmodels must be greater than 0\n");
    return 1;
  }

 if(stumps != 0 && stumps != 1){
    fprintf(stderr,"compute_tree_bagging: parameter stumps must be 0 or 1\n");
    return 1;
  }

  if(minsize < 0){
    fprintf(stderr,"compute_tree_bagging: parameter minsize must be >= 0\n");
    return 1;
  }

  etree->nclasses=iunique(y,n, &(etree->classes));

  if(etree->nclasses<=0){
    fprintf(stderr,"compute_tree_adaboost: iunique error\n");
    return 1;
  }
  if(etree->nclasses==1){
    fprintf(stderr,"compute_tree_adaboost: only 1 class recognized\n");
    return 1;
  }

  if(etree->nclasses==2)
    if(etree->classes[0] != -1 || etree->classes[1] != 1){
      fprintf(stderr,"compute_tree_adaboost: for binary classification classes must be -1,1\n");
      return 1;
    }
  
  if(etree->nclasses>2){
    fprintf(stderr,"compute_tree_adaboost: multiclass classification not allowed\n");
    return 1;
  }

  if(!(etree->tree=(Tree *)calloc(nmodels,sizeof(Tree)))){
    fprintf(stderr,"compute_tree_adaboost: out of memory\n");
    return 1;
  }

  if(!(etree->weights=dvector(nmodels))){
    fprintf(stderr,"compute_tree_adaboost: out of memory\n");
    return 1;
  }

  if(!(trx=(double **)calloc(n,sizeof(double*)))){
    fprintf(stderr,"compute_tree_adaboost: out of memory\n");
    return 1;
  }
  if(!(try=ivector(n))){
    fprintf(stderr,"compute_tree_adaboost: out of memory\n");
    return 1;
  }
  
  if(!(prob_copy=dvector(n))){
    fprintf(stderr,"compute_tree_adaboost: out of memory\n");
    return 1;
  }
  if(!(prob=dvector(n))){
    fprintf(stderr,"compute_tree_adaboost: out of memory\n");
    return 1;
  }

  if(!(pred=ivector(n))){
    fprintf(stderr,"compute_tree_adaboost: out of memory\n");
    return 1;
  }

  for(i =0;i<n;i++)
    prob[i]=1.0/(double)n;

  etree->nmodels=nmodels;
  sumalpha=0.0;
  for(b=0;b<nmodels;b++){

    for(i =0;i<n;i++)
      prob_copy[i]=prob[i];
    if(sample(n, prob_copy, n, &samples, TRUE,b)!=0){
      fprintf(stderr,"compute_tree_adaboost: sample error\n");
      return 1;
    }

    for(i=0;i<n;i++){
      trx[i] = x[samples[i]];
      try[i] = y[samples[i]];
    }
    
    if(compute_tree(&(etree->tree[b]),n,d,trx,try,stumps,minsize)!=0){
      fprintf(stderr,"compute_tree_adaboost: compute_tree error\n");
      return 1;
    }
    free_ivector(samples);

    eps=0.0;
    for(i=0;i<n;i++){
      pred[i]=predict_tree(&(etree->tree[b]),x[i],&margin);
      if(pred[i] < -1 ){
	fprintf(stderr,"compute_tree_adaboost: predict_tree error\n");
	return 1;
      }
      if(pred[i]==0 || pred[i] != y[i])
	eps += prob[i];
      free_dvector(margin);
    }
    
    if(eps > 0.0 && eps < 0.5){
      etree->weights[b]=0.5 *log((1.0-eps)/eps);
      sumalpha+=etree->weights[b];
    }else{
      etree->nmodels=b;
      break;
    }
      
    sumprob=0.0;
    for(i=0;i<n;i++){
      prob[i]=prob[i]*exp(-etree->weights[b]*y[i]*pred[i]);
      sumprob+=prob[i];
    }

    if(sumprob <=0.0){
      fprintf(stderr,"compute_tree_adaboost: sumprob = 0\n");
      return 1;
    }
    for(i=0;i<n;i++)
      prob[i] /= sumprob;
    
  }
  
  if(etree->nmodels<=0){
    fprintf(stderr,"compute_tree_adaboost: no models produced\n");
    return 1;
  }

  if(sumalpha <=0){
      fprintf(stderr,"compute_tree_adaboost: sumalpha = 0\n");
      return 1;
  }
  for(b=0;b<etree->nmodels;b++)
    etree->weights[b] /= sumalpha;
  
  free(trx);
  free_ivector(try);
  free_ivector(pred);
  free_dvector(prob);
  free_dvector(prob_copy);
  return 0;

}



static void split_node(Node *node,Node *nodeL,Node *nodeR,int classes[],
		       int nclasses)
{
  int **indx;
  double *tmpvar;
  int i,j,k;
  int **npL , **npR;
  double **prL , **prR;
  int totL,totR;
  double a,b;
  double *decrease_in_inpurity;
  double max_decrease=0;
  int splitvar;
  int splitvalue;
  int morenumerous;

  nodeL->priors=dvector(nclasses);
  nodeR->priors=dvector(nclasses);
  nodeL->npoints_for_class=ivector(nclasses);
  nodeR->npoints_for_class=ivector(nclasses);
  indx=imatrix(node->nvar,node->npoints);
  tmpvar=dvector(node->npoints);
  decrease_in_inpurity=dvector(node->npoints-1);
  npL=imatrix(node->npoints,nclasses);
  npR=imatrix(node->npoints,nclasses);
  prL=dmatrix(node->npoints,nclasses);
  prR=dmatrix(node->npoints,nclasses);

  splitvar=0;
  splitvalue=0;
  max_decrease=0;

  for(i=0;i<node->nvar;i++){
    for(j=0;j<node->npoints;j++)
      tmpvar[j]=node->data[j][i];
    
    for(j=0;j<node->npoints;j++)
      indx[i][j]=j;
    dsort(tmpvar,indx[i],node->npoints,SORT_ASCENDING);

    for(k=0;k<nclasses;k++)
      if(node->classes[indx[i][0]]==classes[k]){
	npL[0][k] = 1;
	npR[0][k] = node->npoints_for_class[k]-npL[0][k];
      } else{
	npL[0][k] = 0;
	npR[0][k] = node->npoints_for_class[k];
      }
    
    for(j=1;j<node->npoints-1;j++)
      for(k=0;k<nclasses;k++)
	if(node->classes[indx[i][j]]==classes[k]){
	  npL[j][k] = npL[j-1][k] +1;
	  npR[j][k] = node->npoints_for_class[k] - npL[j][k];
	}
	else {
	  npL[j][k] = npL[j-1][k];
	  npR[j][k] = node->npoints_for_class[k] - npL[j][k];
	}


    for(j=0;j<node->npoints-1;j++){
      if(node->data[indx[i][j]][i] != node->data[indx[i][j+1]][i]){
	totL = totR = 0;
	
	for(k=0;k<nclasses;k++)
	  totL += npL[j][k];
	for(k=0;k<nclasses;k++)
	  prL[j][k] =  (double) npL[j][k] / (double) totL;
	
	for(k=0;k<nclasses;k++)
	  totR += npR[j][k];
	for(k=0;k<nclasses;k++)
	  prR[j][k] =  (double) npR[j][k] /(double)  totR;
	
	a = (double) totL / (double) node->npoints;
	b = (double) totR / (double) node->npoints ;
	
	decrease_in_inpurity[j] = gini_index(node->priors,nclasses) - 
	  a * gini_index(prL[j],nclasses) - b * gini_index(prR[j],nclasses);
      }
    }

    for(j=0;j<node->npoints-1;j++)
      if(decrease_in_inpurity[j] > max_decrease){
	max_decrease = decrease_in_inpurity[j];
	
	splitvar=i;
	splitvalue=j;

	for(k=0;k<nclasses;k++){
	  nodeL->priors[k]=prL[splitvalue][k];
	  nodeR->priors[k]=prR[splitvalue][k];
	  nodeL->npoints_for_class[k]=npL[splitvalue][k];
	  nodeR->npoints_for_class[k]=npR[splitvalue][k];
	}
      }
  }
  
  
  node->var=splitvar;
  node->value=(node->data[indx[splitvar][splitvalue]][node->var]+      
	       node->data[indx[splitvar][splitvalue+1]][node->var])/2.;

  nodeL->nvar=node->nvar;
  nodeL->nclasses=node->nclasses;
  nodeL->npoints=splitvalue+1;

  nodeL->terminal=TRUE;
  if(gini_index(nodeL->priors,nclasses) >0)
    nodeL->terminal=FALSE;

  nodeL->data=(double **) calloc(nodeL->npoints,sizeof(double *));
  nodeL->classes=ivector(nodeL->npoints);

  for(i=0;i<nodeL->npoints;i++){
    nodeL->data[i] = node->data[indx[splitvar][i]];
    nodeL->classes[i] = node->classes[indx[splitvar][i]];
  }
  
  
  morenumerous=0;
  for(k=0;k<nclasses;k++)
    if(nodeL->npoints_for_class[k] > morenumerous){
      morenumerous = nodeL->npoints_for_class[k];
      nodeL->node_class=classes[k];
    }
  


  nodeR->nvar=node->nvar;
  nodeR->nclasses=node->nclasses;
  nodeR->npoints=node->npoints-nodeL->npoints;

  nodeR->terminal=TRUE;
  if(gini_index(nodeR->priors,nclasses) >0)
    nodeR->terminal=FALSE;

  nodeR->data=(double **) calloc(nodeR->npoints,sizeof(double *));
  nodeR->classes=ivector(nodeR->npoints);

  for(i=0;i<nodeR->npoints;i++){
    nodeR->data[i] = node->data[indx[splitvar][nodeL->npoints+i]];
    nodeR->classes[i] = node->classes[indx[splitvar][nodeL->npoints+i]];
  }
  
  morenumerous=0;
  for(k=0;k<nclasses;k++)
    if(nodeR->npoints_for_class[k] > morenumerous){
      morenumerous = nodeR->npoints_for_class[k];
      nodeR->node_class=classes[k];
    }

  free_imatrix(indx,  node->nvar,node->npoints);
  free_imatrix(npL, node->npoints,nclasses);
  free_imatrix(npR, node->npoints,nclasses);
  free_dmatrix(prL, node->npoints,nclasses);
  free_dmatrix(prR, node->npoints,nclasses);
  free_dvector(tmpvar);
  free_dvector(decrease_in_inpurity);

}


static double gini_index(double p[],int n)
{
  int i;
  double gini=.0;
  
  for(i=0;i<n;i++)
    gini += p[i] * p[i];
  
  return 1.0 - gini;
  
} 



