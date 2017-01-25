#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#ifndef __DISTLOC_H__
#define __DISTLOC_H__


#define CANTI_TIPO_REC 23

//Definición de la estructura de Localidad
struct Localidad{
  int est;
  int mun;
  int loc;

  int id;
  int cconapo;
  float lat;
  float lng;

 //coordenas cartesianas
  float x;
  float y;
  float z;

  int pob;

  //distancias mínimas a recursos
  float dist[CANTI_TIPO_REC];

  //claves conapo a las localidades de los recursos
  int c[23];
};

typedef struct Localidad sLocalidad;
typedef struct Localidad* PLocalidad;


struct Recurso{
  int est;
  int mun;
  int loc;

  int cconapo;
  float lat;
  float lng;

  float x;
  float y;
  float z;

  char stipo[22];
  int tipo;

  int id;
  int uniq_id;
};

typedef struct Recurso sRecurso;
typedef struct Recurso* PRecurso;


struct Diccionario{
  char nombre[22];
};

typedef struct Diccionario sDiccionario;
typedef struct Diccionario* PDiccionario;


#endif
