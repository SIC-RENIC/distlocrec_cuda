#include "distlocrec.h"
#include "distlocrec_cuda.h"

extern int cantiloc;
extern int cantirec;
extern PLocalidad ploc;
extern PRecurso prec;

extern int recxtipo[23];

void calculaDLR(void);

void copiaLocalidad(void);
void liberaLocalidad(void);

void copiaRecursoTema(PSRec h_prec,int tema);


void preparaMemoriaDest(void);
void liberaMemoriaDest(void);

void impactaResultado(void);

PSLoc h_ploc = NULL;


PSLoc d_ploc = NULL;
PSRec d_prec = NULL;

PSDest h_pdest = NULL;
PSDest d_pdest = NULL;


/**
 *
 */
__global__ void calculadistLR(int nlocs, int nrecs, PSLoc ploc, PSRec prec,
		PSDest pdest) {
	int j = 0;
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
	copiaLocalidad();

	int canti_hilos = 1024;
	int canti_bloques = ceil(cantiloc / canti_hilos);

	cudaMalloc((void**) &(d_ploc), cantiloc * sizeof(struct SLOC));
	cudaMemcpy(d_ploc, h_ploc, cantiloc * sizeof(struct SLOC),cudaMemcpyHostToDevice);

	preparaMemoriaDest();

	for (tema = 0; tema < 23; tema++) {

		PSRec h_prec = (PSRec) malloc(sizeof(struct SREC) * recxtipo[tema]);
		copiaRecursoTema(h_prec,tema);

		cudaMalloc((void**) &(d_prec), recxtipo[tema] * sizeof(struct SREC));
		cudaMemcpy(d_prec, h_prec,recxtipo[tema] * sizeof(struct SREC),cudaMemcpyHostToDevice);

		calculadistLR<<<canti_bloques,canti_hilos>>>(cantiloc,recxtipo[tema],d_ploc,d_prec,d_pdest);

		cudaMemcpy(h_pdest, d_pdest, cantiloc * sizeof(struct SDEST), cudaMemcpyDeviceToHost);

		impactaResultado();

		cudaFree(d_prec);

		free(h_prec);
		h_prec=NULL;
	}

	liberaMemoriaDest();

	cudaFree(d_ploc);

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
		(h_ploc + i)->id = (ploc + i)->cconapo;
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
void copiaRecursoTema(PSRec h_prec,int tema) {

	int i;
	int j=0;

	for (i = 0; i < cantirec; i++) {

		if ((prec + i)->tipo == tema) {
			(h_prec + j)->x = (prec + i)->x;
			(h_prec + j)->y = (prec + i)->y;
			(h_prec + j)->z = (prec + i)->z;
			(h_prec + j)->id = (prec + i)->uniq_id;
			j++;
		}

	}

}



/**
 *
 */
void preparaMemoriaDest(void){

	h_pdest = (PSDest) malloc(sizeof(struct SDEST) * cantiloc);
	cudaMalloc((void**) &(d_pdest), cantiloc * sizeof(struct SDEST));
}

/**
 *
 */
void liberaMemoriaDest(void){
	free(h_pdest);
	cudaFree(d_pdest);
	h_pdest=NULL;
	d_pdest=NULL;
}

/**
 *
 */
void impactaResultado(void){

	int i,j;

	for(i=0;i<cantiloc;i++){
		for(j=0;j<cantirec;j++){
			if((h_pdest+i)->jmin==(prec+j)->uniq_id){

				(ploc+i)->dist[(prec+j)->tipo]=(h_pdest+i)->dist;
				(ploc+i)->c[(prec+j)->tipo]=(prec+j)->cconapo;
			}
		}
	}
}
