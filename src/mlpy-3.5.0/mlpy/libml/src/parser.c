#include <stdio.h>
#include <string.h>
#include <stdlib.h>


int parser(int n,char *array[],char ***flag,char ***value,int *nflags)
     /*
       Given an array of length n (tipically argv and argc) fills the
       vectors flag and values storing in a) "flag" the flags (-c,-d) and
       in b) "values" the values (input_configuration_file,
       output_database_file,...). "nflags" will contain the number of
       flags.

       Return value: 1 if an error occurred, 0 otherwise
     */
{
  if(n <= 2){
    fprintf(stderr,"parser: not enough parameters\n");
    return 1;
  }

  if(!((*flag)=(char **)calloc(n,sizeof(char *)))){
    fprintf(stderr,"parser: out of memory\n");
    return 1;
  }
  if(!((*value)=(char **)calloc(n,sizeof(char *)))){
    fprintf(stderr,"parser: out of memory\n");
    return 1;
  }

  (*nflags)=0;
  while(--n){
    (*value)[(*nflags)]=array[n--];
    if(array[n][0]=='-')
      (*flag)[(*nflags)++]=array[n];
    else{
      fprintf(stderr,"parser: wrong command line format\n");
      return 1;
    }
  }
  return 0;
}


char *get_value(char *flag[],char *value[],int nflags,char opt[])
     /*
       being flag value and nflags as computed by the function parser,
       get_value returns a pointer to the string corresponding to the flag opt
       or NULL if no flag matches opt
     */
{
  int i;
  char *out=NULL;

  for(i=0;i<nflags;i++)
    if(strcmp(flag[i],opt)==0)
      out=value[i];

  return out;
}
