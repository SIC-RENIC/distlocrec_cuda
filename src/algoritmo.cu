#include "distlocrec.h"
#include "distlocrec_cuda.h"

extern int cantiloc;
extern int cantirec;
extern int cantixtipo[CANTI_TIPO_REC];

extern PLocalidad ploc;
extern PRecurso prec;

void calculaDLR(void);

void copiaLocalidad(void);
void liberaLocalidad(void);

void copiaRecursosxTema(void);
void liberaRecursosxTema(void);

void preparaMemoriaDestxTema(void);
void liberaMemoriaDestxTema(void);

void impactaResultado(void);

PSLoc h_ploc = NULL;
PSLoc d_ploc = NULL;

PSRec h_prec = NULL;
PSRec d_prec = NULL;

PSDest h_pdest = NULL;
PSDest d_pdest = NULL;

PSRec ah_prec[CANTI_TIPO_REC];
PSRec ad_prec[CANTI_TIPO_REC];
PSDest ah_pdest[CANTI_TIPO_REC];
PSDest ad_pdest[CANTI_TIPO_REC];

/**
 *
 */

__global__ void calculadistLR(int nlocs, int nrecs, PSLoc ploc, PSRec prec,
		PSDest pdest) {
	int j;
	struct SLOC loc;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	double dist = 4;
	int jmin;

	double aux;

	if (i < nlocs) {
		loc = *(ploc + i);
		for (j = 0; j < nrecs; j++) {
			aux = loc.x * (prec + j)->x + loc.y * (prec + j)->y
					+ loc.z * (prec + j)->z;
			aux = acos(aux);
			if (dist > aux) {
				dist = aux;
				jmin = (prec + j)->id;
			}
		}

		(pdest + i)->jmin = jmin;
		(pdest + i)->dist = dist;

	}

}

/**
 *
 */
void calculaDLR(void) {
	int tema;
	cudaStream_t stream[CANTI_TIPO_REC];



	copiaLocalidad();
	copiaRecursosxTema();
	preparaMemoriaDestxTema();

	int canti_hilos = 640;
	int canti_bloques = ceil(cantiloc / canti_hilos) + 1;

	printf("Bloques: %d Hilos: %d\n", canti_bloques, canti_hilos);


	for (tema = 0; tema < CANTI_TIPO_REC; tema++) {
			cudaStreamCreate(&stream[tema]);
		}

	for (tema = 0; tema < CANTI_TIPO_REC; tema++) {

		cudaMemcpyAsync(ad_pdest[tema], ah_pdest[tema],
				cantiloc * sizeof(struct SDEST), cudaMemcpyHostToDevice,
				stream[tema]);

		calculadistLR<<<canti_bloques, canti_hilos, 0, stream[tema]>>>(cantiloc,
				cantixtipo[tema], d_ploc, ad_prec[tema], ad_pdest[tema]);

		cudaMemcpyAsync(ah_pdest[tema], ad_pdest[tema],
				cantiloc * sizeof(struct SDEST), cudaMemcpyDeviceToHost,
				stream[tema]);

	}

	for (tema = 0; tema < CANTI_TIPO_REC; tema++) {
		cudaStreamSynchronize(stream[tema]);
	}

	impactaResultado();

	liberaMemoriaDestxTema();
	liberaRecursosxTema();
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

	cudaMalloc((void**) &(d_ploc), cantiloc * sizeof(struct SLOC));
	cudaMemcpy(d_ploc, h_ploc, cantiloc * sizeof(struct SLOC),
			cudaMemcpyHostToDevice);
}

/**
 *
 */
void liberaLocalidad(void) {
	cudaFree(d_ploc);
	free(h_ploc);
}

/**
 *
 */
void copiaRecursosxTema(void) {

	for (int tema = 0; tema < CANTI_TIPO_REC; tema++) {
		cudaMallocHost((void **) &ah_prec[tema],
				sizeof(struct SREC) * cantixtipo[tema]);
		int j = 0;
		for (int i = 0; i < cantirec; i++) {
			if ((prec + i)->tipo == tema) {
				(ah_prec[tema] + j)->x = (prec + i)->x;
				(ah_prec[tema] + j)->y = (prec + i)->y;
				(ah_prec[tema] + j)->z = (prec + i)->z;
				(ah_prec[tema] + j)->id = (prec + i)->uniq_id;
				(ah_prec[tema] + j)->tipo = (prec + i)->tipo;
				j++;
			}
		}
	}
}

/**
 *
 */
void liberaRecursosxTema(void) {
	for (int tema = 0; tema < CANTI_TIPO_REC; tema++) {
		cudaFreeHost(ah_prec[tema]);
	}
}

void preparaMemoriaDestxTema(void) {

	for (int tema = 0; tema < CANTI_TIPO_REC; tema++) {
		cudaMallocHost((void **) &ah_pdest[tema],
				sizeof(struct SDEST) * cantiloc);
		cudaMalloc((void**) &ad_pdest[tema], cantiloc * sizeof(struct SDEST));
	}

}

void liberaMemoriaDestxTema(void) {
	for (int tema = 0; tema < CANTI_TIPO_REC; tema++) {
		cudaFreeHost(ah_pdest[tema]);
		cudaFree(ad_pdest[tema]);
	}
}



/**
 *
 */
void impactaResultado(void) {

	for(int tema=0;tema<CANTI_TIPO_REC;tema++){
		for (int i = 0; i < cantiloc; i++) {
			for (int j = 0; j < cantirec; j++) {
				if((ah_pdest[tema]+i)->jmin==(prec + j)->uniq_id){
					(ploc + i)->dist[tema] = (ah_pdest[tema] + i)->dist;
					(ploc + i)->c[tema] = (prec + j)->cconapo;
				}
			}
		}
	}

}
