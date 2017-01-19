#include "distlocrec.h"
#include "distlocrec_cuda.h"

extern int cantiloc;
extern int cantirec;
extern PLocalidad ploc;
extern PRecurso prec;

void calculaDLR(void);

void copiaLocalidad(void);
void liberaLocalidad(void);

void copiaRecursos(void);
void liberaRecursos(void);

void preparaMemoriaDest(void);
void liberaMemoriaDest(void);

void impactaResultado(void);


PSLoc h_ploc = NULL;
PSRec h_prec = NULL;

PSLoc d_ploc = NULL;
PSRec d_prec = NULL;

PSDest h_pdest = NULL;
PSDest d_pdest = NULL;

/**
 *
 */

__global__ void calculadistLR(int nlocs, int nrecs, PSLoc ploc, PSRec prec,
		PSDest pdest) {
	int j;
	struct SLOC loc;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	double dist[CANTI_TIPO_REC];
	int jmin[CANTI_TIPO_REC];
	for (j = 0; j < CANTI_TIPO_REC; j++) {
		dist[j] = 4;
	}

	double aux;

	if (i < nlocs) {
		loc = *(ploc + i);
		for (j = 0; j < nrecs; j++) {
			aux = loc.x * (prec + j)->x + loc.y * (prec + j)->y
					+ loc.z * (prec + j)->z;
			aux = acos(aux);
			if (dist[(prec + j)->tipo] > aux) {
				dist[(prec + j)->tipo] = aux;
				jmin[(prec + j)->tipo] = (prec + j)->id;
			}
		}

		for (j = 0; j < CANTI_TIPO_REC; j++) {
			(pdest + i)->jmin[j] = jmin[j];
			(pdest + i)->dist[j] = dist[j];
		}
	}

}

/**
 *
 */
void calculaDLR(void) {

	copiaLocalidad();
	copiaRecursos();


	int canti_hilos = 640;
	int canti_bloques = ceil(cantiloc / canti_hilos) + 1;

	printf("Bloques: %d Hilos: %d\n", canti_bloques, canti_hilos);

	cudaMalloc((void**) &(d_ploc), cantiloc * sizeof(struct SLOC));
	cudaMemcpy(d_ploc, h_ploc, cantiloc * sizeof(struct SLOC),
			cudaMemcpyHostToDevice);

	cudaMalloc((void**) &(d_prec), cantirec * sizeof(struct SREC));
	cudaMemcpy(d_prec, h_prec, cantirec * sizeof(struct SREC),
			cudaMemcpyHostToDevice);

	preparaMemoriaDest();

	calculadistLR<<<canti_bloques, canti_hilos>>>(cantiloc, cantirec, d_ploc,
			d_prec, d_pdest);

	cudaMemcpy(h_pdest, d_pdest, cantiloc * sizeof(struct SDEST),
			cudaMemcpyDeviceToHost);

	impactaResultado();

	liberaMemoriaDest();

	cudaFree(d_prec);
	cudaFree(d_ploc);

	liberaRecursos();
	liberaLocalidad();
}

/**
 *
 */
void copiaLocalidad(void) {
	int i;
	h_ploc = (PSLoc) malloc(sizeof(struct SLOC) * cantiloc);

	for (i = 0; i < cantiloc; i++) {
		(h_ploc + i)->x = (ploc + i)->x;
		(h_ploc + i)->y = (ploc + i)->y;
		(h_ploc + i)->z = (ploc + i)->z;
		(h_ploc + i)->id = (ploc + i)->id;
	}
}

/**
 *
 */
void liberaLocalidad(void) {
	free(h_ploc);
}

/**
 *
 */
void copiaRecursos(void) {
	int i;
	h_prec = (PSRec) malloc(sizeof(struct SREC) * cantirec);
	for (i = 0; i < cantirec; i++) {
		(h_prec + i)->x = (prec + i)->x;
		(h_prec + i)->y = (prec + i)->y;
		(h_prec + i)->z = (prec + i)->z;
		(h_prec + i)->id = (prec + i)->uniq_id;
		(h_prec + i)->tipo = (prec + i)->tipo;
	}
}

/**
 *
 */
void liberaRecursos(void) {
	free(h_prec);
}

/**
 *
 */
void preparaMemoriaDest(void) {

	h_pdest = (PSDest) malloc(sizeof(struct SDEST) * cantiloc);
	cudaMalloc((void**) &(d_pdest), cantiloc * sizeof(struct SDEST));
}

/**
 *
 */
void liberaMemoriaDest(void) {
	free(h_pdest);
	cudaFree(d_pdest);
	h_pdest = NULL;
	d_pdest = NULL;
}

/**
 *
 */
void impactaResultado(void) {

	int i, j, k;

	for (i = 0; i < cantiloc; i++) {
		for (j = 0; j < cantirec; j++) {
			for (k = 0; k < CANTI_TIPO_REC; k++) {
				if (k == (prec + j)->tipo
						&& (h_pdest + i)->jmin[k] == (prec + j)->uniq_id) {

					(ploc + i)->dist[(prec + j)->tipo] = (h_pdest + i)->dist[k];
					(ploc + i)->c[(prec + j)->tipo] = (prec + j)->cconapo;
				}
			}
		}
	}
}


