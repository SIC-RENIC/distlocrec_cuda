/*
 * algoritmo_a.cu
 *
 *  Created on: 18/01/2017
 *      Author: alfonso
 */
#include "distlocrec.h"

void calculaDLRv2(void);
void alojaMemoria(void);
void liberaMemoria(void);

extern int cantiloc;
extern int cantirec;
extern PLocalidad ploc;
extern PRecurso prec;

//coordenadas de las localidades
double *hloc_x;
double *hloc_y;
double *hloc_z;

double *dloc_x;
double *dloc_y;
double *dloc_z;

//coordenadas de los recursos

double *hrec_x;
double *hrec_y;
double *hrec_z;


double *drec_x;
double *drec_y;
double *drec_z;


//valores resultantes del calculo

double *hdist_resultado;

double *ddist_resultado;

__global__ void calculadistLRv2(int nlocs, int nrecs, int ntipo, double* dloc_x,
		double* dloc_y, double* dloc_z, double* drec_x, double* drec_y,
		double* drec_z, double *ddist_resultado) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	double dist = 4;
	double daux;

	if (id < nlocs) {
		double x = *(dloc_x + id);
		double y = *(dloc_y + id);
		double z = *(dloc_z + id);

		for (int j = 0; j < nrecs; j++) {
			daux = *(drec_x + j) * x + *(drec_y + j) * y + *(drec_z + j) * z;
			if (daux < dist) {
				dist = daux;
			}
		}

		*(ddist_resultado + ((id - 1) * CANTI_TIPO_REC) + ntipo) = dist;

	}

}

/**
 *
 */
void calculaDLRv2(void) {
	alojaMemoria();

	liberaMemoria();
}

/**
 *
 */
void alojaMemoria(void) {
//Localidades
	hloc_x = (double*) malloc(sizeof(double) * cantiloc);
	hloc_y = (double*) malloc(sizeof(double) * cantiloc);
	hloc_z = (double*) malloc(sizeof(double) * cantiloc);

	for (int i = 0; i < cantiloc; i++) {
		Localidad p = *(ploc + i);
		*(hloc_x + i) = p.x;
		*(hloc_y + i) = p.y;
		*(hloc_z + i) = p.z;
	}

	cudaMalloc((void**) &(dloc_x), cantiloc * sizeof(double));
	cudaMalloc((void**) &(dloc_y), cantiloc * sizeof(double));
	cudaMalloc((void**) &(dloc_z), cantiloc * sizeof(double));

//Recursos
	hrec_x = (double*) malloc(sizeof(double) * cantirec);
	hrec_y = (double*) malloc(sizeof(double) * cantirec);
	hrec_z = (double*) malloc(sizeof(double) * cantirec);


	for (int i = 0; i < cantirec; i++) {
		Recurso p = *(prec + i);
		*(hrec_x + i) = p.x;
		*(hrec_y + i) = p.y;
		*(hrec_z + i) = p.z;

	}

	cudaMalloc((void**) &(drec_x), cantirec * sizeof(double));
	cudaMalloc((void**) &(drec_y), cantirec * sizeof(double));
	cudaMalloc((void**) &(drec_z), cantirec * sizeof(double));


//Resultados
	hdist_resultado = (double*) malloc(
			sizeof(double) * cantiloc * CANTI_TIPO_REC);

	cudaMalloc((void**) &(ddist_resultado),
			cantiloc * CANTI_TIPO_REC * sizeof(double));

}

/**
 *
 */
void liberaMemoria(void) {

	cudaFree(ddist_resultado);
	free(hdist_resultado);


	cudaFree(drec_z);
	cudaFree(drec_y);
	cudaFree(drec_x);


	free(hrec_z);
	free(hrec_y);
	free(hrec_x);

	cudaFree(dloc_z);
	cudaFree(dloc_y);
	cudaFree(dloc_x);

	free(hloc_z);
	free(hloc_y);
	free(hloc_x);
}

