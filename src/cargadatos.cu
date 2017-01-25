/*
 * cargadatos.c
 *
 *  Created on: 13/01/2017
 *      Author: alfonso
 */

#include "distlocrec.h"

extern PLocalidad ploc;
extern PRecurso prec;
extern PDiccionario pdic;

void cargaArchivoLocs(char *);
void cargaArchivoRecs(char *);


int cantixtipo[CANTI_TIPO_REC];

int eml2conapo(int, int, int);
float deg2rad(float);
void cesfe2carte(float lat, float lng, float *res);


/**
* Función que carga los datos de localidades
*/
void cargaArchivoLocs(char * archlocs){

  printf("Carga Localidades\n");

  int est;
  int mun;
  int loc;

  float lat,lng;
  float latr,lngr;


  float *res2=(float *)malloc(3*sizeof(float));


  int pob;

  FILE *fh=fopen(archlocs,"r");

  PLocalidad p;

  int j=0;
  int i=0;
  while(fscanf(fh,"%d %d %d %f %f %d",&est,&mun,&loc,&lat,&lng,&pob)!=EOF){


    latr=deg2rad(lat);
    lngr=deg2rad(lng);
    cesfe2carte(latr,lngr,res2);

    p=(ploc+i);

    p->est=est;
    p->mun=mun;
    p->loc=loc;
    p->cconapo=eml2conapo(est,mun,loc);

    p->lat=lat;
    p->lng=lng;

    p->x=*(res2);
    p->y=*(res2+1);
    p->z=*(res2+2);

    p->pob=pob;

    for(j=0;j<CANTI_TIPO_REC;j++){
      p->dist[j]=M_PI;
    }

    p->id=i;
    i++;
  }

  printf("Localidades : %d\n",i);

  fclose(fh);

  free(res2);

}

/**
* Función que se encarga de cargar los recursos en la localidades
*/
void cargaArchivoRecs(char * archrecs){

  printf("Carga Recursos\n");

  int est;
  int mun;
  int loc;

  float lat,lng;
  float latr,lngr;

  char stipo[22];
  int tipo;
  int id;

  float *res2=(float *)malloc(3*sizeof(float));

  FILE *fh=fopen(archrecs,"r");

  PRecurso p;

  int i;

    for(i=0;i<CANTI_TIPO_REC;i++){
      cantixtipo[i]=0;
    }

    i=0;
  while(fscanf(fh,"%d %d %d %f %f %s %d %d",&est,&mun,&loc,&lat,&lng,stipo,&tipo,&id)!=EOF){


    latr=deg2rad(lat);
    lngr=deg2rad(lng);
    cesfe2carte(latr,lngr,res2);

    p=(prec+i);

    p->est=est;
    p->mun=mun;
    p->loc=loc;
    p->cconapo=eml2conapo(est,mun,loc);

    p->lat=lat;
    p->lng=lng;

    p->x=*(res2);
    p->y=*(res2+1);
    p->z=*(res2+2);

    strcpy(p->stipo,stipo);

    if((pdic+tipo)->nombre[0]=='\0'){
      strcpy((pdic+tipo)->nombre,stipo);
    }

    p->tipo=tipo;
    p->id=id;

    cantixtipo[tipo]++;

    p->uniq_id=i;
    i++;
  }

  printf("Recursos : %d\n",i);

  int total=0;
  for(i=0;i<CANTI_TIPO_REC;i++){
        printf("tipo:%d, %d\n",i,cantixtipo[i]);
        total+=cantixtipo[i];
      }
  //printf("Total: %d\n",total);

  free(res2);
}

/**
* Función que convierte las claves de Estado, Municipio y Localidad a la clave Conapo
*/
int eml2conapo(int e, int m, int l){
  return (e*1000+m)*10000+l;
}

/**
* Función que convierte los DEG a RAD
*/
float deg2rad(float x){
  return M_PI*x/180.00;
}

/**
* Función convierte coordenadas geográficas a coordenadas cartesianas espaciales
*/
void cesfe2carte(float lat, float lng, float *res){
  *(res)=sin(lng)*cos(lat);
  *(res+1)=cos(lng)*cos(lat);
  *(res+2)=sin(lat);
}



