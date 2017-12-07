#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef BLOCKSIZE
#undef BLOCKSIZE
#endif
#define BLOCKSIZE 500

int get_line(char **line,FILE *fp)
     /*
       Reads a line from the stream pointer fp

       Return values:
       3 non empty line ending in newline
       2 non empty line not ending in newline
       1 empty line not ending in EOF
       0 empty line ending in EOF
       -1 error
     */

{
  int c;
  int pos=0;
  int nblocks=1;
  int maxsize=BLOCKSIZE+1;

  (*line)=calloc(maxsize,sizeof(char));

  while((c=fgetc(fp)) != EOF && c != '\n'){
    if(pos==maxsize-1){
      maxsize=(++nblocks)*BLOCKSIZE+1;
      if(!((*line)=(char *)realloc(*line,maxsize*sizeof(char)))){
	fprintf(stderr,"get_line: out of memory\n");
	return -1;
      }
    }
    (*line)[pos++]=c;
  }
  if(c=='\n'){
    (*line)[pos]='\0';
    if(strlen(*line)>0)
      return 3;
    else
      return 1;
  }else if(c==EOF){
    (*line)[pos]='\0';
    if(strlen(*line)>0){
      return 2;
    }
    else
      return 0;
    }

    fprintf(stderr,"get_line: WARNING: unusual exit status\n");
    return -1;
}

#undef BLOCKSIZE
