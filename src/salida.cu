/*
 * salida.cu
 *
 *  Created on: 13/01/2017
 *      Author: alfonso
 */


#include "distlocrec.h"

void escribeSalida(double radio);


extern PLocalidad ploc;
extern PDiccionario pdic;
extern int cantiloc;


/**
* Función que escribe la salida de todas las localidades con sus casos
*/
void escribeSalida(double radio){

  FILE * fh;
  int i,j;

  int e,m,l;

  fh=fopen("./salida.txt","w");
  for(i=0;i<cantiloc;i++){
    PLocalidad p=(ploc+i);

    for(j=0;j<CANTI_TIPO_REC;j++){
      e=(int)(p->c[j]/10000000);
      m=(int)((p->c[j]-e*10000000)/10000);
      l=p->c[j]-e*10000000-m*10000;
      fprintf(fh,"%d,%d,%d,%s,%d,%lf,%d,%d,%d,0\n",p->est,p->mun,p->loc,(pdic+j)->nombre,p->pob,radio*p->dist[j],e,m,l);
    }

  }

  fclose(fh);

}

