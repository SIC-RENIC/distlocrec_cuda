/*
 * algoritmo_a.cu
 *
 *  Created on: 18/01/2017
 *      Author: alfonso
 */
#include "distlocrec.h"

void calculaDLRv2(double radio);
void alojaMemoria(void);
void liberaMemoria(void);
void imprimeResultado(double radio);

extern int cantiloc;
extern int cantirec;
extern PLocalidad ploc;
extern PRecurso prec;
extern PDiccionario pdic;

extern int cantixtipo[CANTI_TIPO_REC];

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

int *hidrec_resultado;
int *didrec_resultado;

/**
 *
 */
__global__ void calculadistLRv2(int nlocs, int nrecs, int ntipo,int offset, double* dloc_x,
		double* dloc_y, double* dloc_z, double* drec_x, double* drec_y,
		double* drec_z, double *ddist_resultado, int *didrec_resultado) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	double dist = 4;
	int idrec = 0;
	double daux;

	if (id < nlocs) {
		double x = *(dloc_x + id);
		double y = *(dloc_y + id);
		double z = *(dloc_z + id);

		for (int j = 0; j < nrecs; j++) {
			daux = *(drec_x + j) * x + *(drec_y + j) * y + *(drec_z + j) * z;
			daux = acos(daux);
			if (daux < dist) {
				dist = daux;
				idrec = j;
			}
		}

		*(ddist_resultado + (id * CANTI_TIPO_REC) + ntipo) = dist;
		*(didrec_resultado + (id * CANTI_TIPO_REC) + ntipo) = idrec+offset;
	}

}

/**
 *
 */
void calculaDLRv2(double radio) {

	int tema;

	int canti_hilos = 640;
	int canti_bloques = ceil(cantiloc / canti_hilos) + 1;

	alojaMemoria();
	cudaStream_t stream[CANTI_TIPO_REC];

	int offset = 0;

	cudaMemcpy(dloc_x, hloc_x, cantiloc * sizeof(double),
			cudaMemcpyHostToDevice);
	cudaMemcpy(dloc_y, hloc_y, cantiloc * sizeof(double),
			cudaMemcpyHostToDevice);
	cudaMemcpy(dloc_z, hloc_z, cantiloc * sizeof(double),
			cudaMemcpyHostToDevice);

	cudaMemcpy(drec_x, hrec_x, cantirec * sizeof(double),
			cudaMemcpyHostToDevice);
	cudaMemcpy(drec_y, hrec_y, cantirec * sizeof(double),
			cudaMemcpyHostToDevice);
	cudaMemcpy(drec_z, hrec_z, cantirec * sizeof(double),
			cudaMemcpyHostToDevice);

	for (tema = 0; tema < CANTI_TIPO_REC; tema++) {

		cudaStreamCreate(&stream[tema]);

		calculadistLRv2<<<canti_bloques, canti_hilos, 0, stream[tema]>>>(
				cantiloc, cantixtipo[tema], tema,offset, dloc_x, dloc_y, dloc_z,
				(drec_x + offset), (drec_y + offset), (drec_z + offset),
				ddist_resultado, didrec_resultado);

		offset += cantixtipo[tema];
	}

	for (tema = 0; tema < CANTI_TIPO_REC; tema++) {
		cudaStreamSynchronize(stream[tema]);
	}

	cudaMemcpy(hdist_resultado, ddist_resultado,
			cantiloc * CANTI_TIPO_REC * sizeof(double), cudaMemcpyDeviceToHost);

	cudaMemcpy(hidrec_resultado, didrec_resultado,
			cantiloc * CANTI_TIPO_REC * sizeof(int), cudaMemcpyDeviceToHost);

	imprimeResultado(radio);

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
	cudaMallocHost((void **) &(hdist_resultado),
			cantiloc * CANTI_TIPO_REC * sizeof(double));

	cudaMalloc((void**) &(ddist_resultado),
			cantiloc * CANTI_TIPO_REC * sizeof(double));

	cudaMallocHost((void **) &(hidrec_resultado),
			cantiloc * CANTI_TIPO_REC * sizeof(int));

	cudaMalloc((void**) &(didrec_resultado),
			cantiloc * CANTI_TIPO_REC * sizeof(int));

}

/**
 *
 */
void liberaMemoria(void) {

	cudaFree(ddist_resultado);
	cudaFreeHost(hdist_resultado);

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

/**
 *
 */
void imprimeResultado(double radio) {
	FILE * fh;

	fh = fopen("./salidav2.txt", "w");
	for (int i = 0; i < cantiloc; i++) {
		PLocalidad pl = (ploc + i);
		for (int tema = 0; tema < CANTI_TIPO_REC; tema++) {
			double distancia = *(hdist_resultado + (i * CANTI_TIPO_REC) + tema);
			int j = *(hidrec_resultado + (i * CANTI_TIPO_REC) + tema);
			PRecurso pr = (prec + j);

			fprintf(fh, "%d,%d,%d,%s,%d,%lf,%d,%d,%d,0\n", pl->est, pl->mun,
					pl->loc, (pdic + tema)->nombre, pl->pob, radio * distancia,
					pr->est, pr->mun, pr->loc);

		}
	}

	fclose(fh);
}
