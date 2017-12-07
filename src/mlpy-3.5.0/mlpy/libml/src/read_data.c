#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ml.h"

int read_classification_data(char file[],char sep,double ***x,int **y,int *r,
			     int *c)
     /*
       Read a "classification data" file with columns separated by sep

       x_11 x_12 ... x_1c y_1
       x_21 x_22 ... x_2c y_2
                 ...
       x_r1 x_r2 ... x_rc y_r	 

       r is the number of rows, c is the number of columns of the matrix x
       (the file contain c+1 columns, the last one contains the class labels y)
       
       Return value: 1 if an error occureed, 0 otherwise
     */
{
  char *line;
  int i;
  int out_get_line;
  FILE *fp;

  if(!(fp = fopen(file, "r"))){
    fprintf(stderr,"read_classification_data: error opening file %s for reading\n",file);
    return 1;
  }

  *c=0;
  out_get_line=get_line(&line,fp);
  if(out_get_line<3){
    switch(out_get_line){
    case 2:
      fprintf(stderr,"read_classification_data: WARNING: first (and unique) line of file %s does not end in newline\n",
	      file);
	break;
    case 1:
      fprintf(stderr,"read_classification_data: file %s starts with an empty line\n",file);
      return 1;
	break;
    case 0:
      fprintf(stderr,"read_classification_data: file %s is empty\n",file);
      return 1;
      break;
    case -1:
      fprintf(stderr,"read_classification_data: get_line error on file %s\n",
	      file);
      return 1;
    default:
      fprintf(stderr,"read_classification_data: unrecognized exit status of get_line on file %s\n",file);
      return 1;
      break;
    }
  }
  while((line = (char *)strchr(line, sep))){
    line++;
    (*c)++;
  }
  if(*c<1){
    fprintf(stderr,"read_classification_data: no columns recognized in file %s\n",file);
    return 1;
  }

  rewind(fp);
  *r=0;
  if(!(*x=dmatrix(*r+1,*c))){
    fprintf(stderr,"read_classification_data: out of memory on file %s\n",
	    file);
    return 1;
  }
  if(!(*y=ivector(*r+1))){
    fprintf(stderr,"read_classification_data: out of memory on file %s\n",
	    file);
    return 1;
  }

  while(out_get_line>=2){
    out_get_line=get_line(&line,fp);
    if(out_get_line<3){
      switch(out_get_line){
      case 2:
	fprintf(stderr,"read_classification_data: line %d of file %s does not end in newline\n",*r+1,file);
	break;
      case 1:
	fprintf(stderr,"read_classification_data: file %s contains an empty line\n",file);
	return 1;
	break;
      case 0:
	fclose(fp);
	return 0;
	break;
      case -1:
	fprintf(stderr,"read_classification_data: get_line error on file %s\n",
		file);
	return 1;
      default:
	fprintf(stderr,"read_classification_data: unrecognized exit status of get_line on file %s\n",file);
	return 1;
	break;
      }
    }
    for(i=0;i<*c;i++){
      if(line[0]==sep){
	fprintf(stderr,"read_classification_data: error reading line %d of file %s: missing value at col %d\n",*r+1,file,i+1);
	return 1;
      }

      sscanf(line,"%lf", &((*x)[*r][i]));
      if(!(line = (char *)strchr(line, sep))){
	fprintf(stderr,"read_classification_data: error reading line %d of file %s: only %d cols (%d expected)\n",*r+1,file,i+1,*c+1);
	return 1;
      }
      line++;
    }
    if(line[0]=='\0'){
      fprintf(stderr,"read_classification_data: error reading line %d of file %s: missing class label\n",*r+1,file);
      return 1;
    }
    sscanf(line,"%d", &((*y)[*r]));
    (*r)++;
    if(!(*x=(double**)realloc(*x,(*r+1)*sizeof(double*)))){
      fprintf(stderr,"read_classification_data: out of memory on file %s\n",
	      file);
      return 1;
    }
    if(!((*x)[*r]=dvector(*c))){
      fprintf(stderr,"read_classification_data: out of memory on file %s\n",
	      file);
      return 1;
    }
    if(!(*y=(int*)realloc(*y,(*r+1)*sizeof(int)))){
      fprintf(stderr,"read_classification_data: out of memory on file %s\n",
	      file);
      return 1;
    }
  }

  fclose(fp);
  return 0;
}

int read_regression_data(char file[],char sep,double ***x,double **y,int *r,
			 int *c)
     /*
       Read a "regression data" file with columns separated by sep

       x_11 x_12 ... x_1c y_1
       x_21 x_22 ... x_2c y_2
                 ...
       x_r1 x_r2 ... x_rc y_r	 

       r is the number of rows, c is the number of columns of the matrix x
       (the file contain c+1 columns, the last one contains the target 
       values y)
       
       Return value: 1 if an error occureed, 0 otherwise
     */
{  char *line;
  int i;
  int out_get_line;
  FILE *fp;

  if(!(fp = fopen(file, "r"))){
    fprintf(stderr,"read_regression_data: error opening file %s for reading\n",file);
    return 1;
  }

  *c=0;
  out_get_line=get_line(&line,fp);
  if(out_get_line<3){
    switch(out_get_line){
    case 2:
      fprintf(stderr,"read_regression_data: WARNING: first (and unique) line of file %s does not end in newline\n",
	      file);
	break;
    case 1:
      fprintf(stderr,"read_regression_data: file %s starts with an empty line\n",file);
      return 1;
	break;
    case 0:
      fprintf(stderr,"read_regression_data: file %s is empty\n",file);
      return 1;
      break;
    case -1:
      fprintf(stderr,"read_regression_data: get_line error on file %s\n",
	      file);
      return 1;
    default:
      fprintf(stderr,"read_regression_data: unrecognized exit status of get_line on file %s\n",file);
      return 1;
      break;
    }
  }
  while((line = (char *)strchr(line, sep))){
    line++;
    (*c)++;
  }
  if(*c<1){
    fprintf(stderr,"read_regression_data: no columns recognized in file %s\n",file);
    return 1;
  }

  rewind(fp);
  *r=0;
  if(!(*x=dmatrix(*r+1,*c))){
    fprintf(stderr,"read_regression_data: out of memory on file %s\n",
	    file);
    return 1;
  }
  if(!(*y=dvector(*r+1))){
    fprintf(stderr,"read_regression_data: out of memory on file %s\n",
	    file);
    return 1;
  }

  while(out_get_line>=2){
    out_get_line=get_line(&line,fp);
    if(out_get_line<3){
      switch(out_get_line){
      case 2:
	fprintf(stderr,"read_regression_data: line %d of file %s does not end in newline\n",*r+1,file);
	break;
      case 1:
	fprintf(stderr,"read_regression_data: file %s contains an empty line\n",file);
	return 1;
	break;
      case 0:
	fclose(fp);
	return 0;
	break;
      case -1:
	fprintf(stderr,"read_regression_data: get_line error on file %s\n",
		file);
	return 1;
      default:
	fprintf(stderr,"read_regression_data: unrecognized exit status of get_line on file %s\n",file);
	return 1;
	break;
      }
    }
    for(i=0;i<*c;i++){
      if(line[0]==sep){
	fprintf(stderr,"read_regression_data: error reading line %d of file %s: missing value at col %d\n",*r+1,file,i+1);
	return 1;
      }

      sscanf(line,"%lf", &((*x)[*r][i]));
      if(!(line = (char *)strchr(line, sep))){
	fprintf(stderr,"read_regression_data: error reading line %d of file %s: only %d cols (%d expected)\n",*r+1,file,i+1,*c+1);
	return 1;
      }
      line++;
    }
    if(line[0]=='\0'){
      fprintf(stderr,"read_regression_data: error reading line %d of file %s: missing target value\n",*r+1,file);
      return 1;
    }
    sscanf(line,"%lf", &((*y)[*r]));
    (*r)++;
    if(!(*x=(double**)realloc(*x,(*r+1)*sizeof(double*)))){
      fprintf(stderr,"read_regression_data: out of memory on file %s\n",
	      file);
      return 1;
    }
    if(!((*x)[*r]=dvector(*c))){
      fprintf(stderr,"read_regression_data: out of memory on file %s\n",
	      file);
      return 1;
    }
    if(!(*y=(double*)realloc(*y,(*r+1)*sizeof(double)))){
      fprintf(stderr,"read_regression_data: out of memory on file %s\n",
	      file);
      return 1;
    }
  }

  fclose(fp);
  return 0;
}

