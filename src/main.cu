/*
 * main.cu
 *
 *  Created on: 12/01/2017
 *      Author: alfonso
 */
#include "distlocrec.h"
#include <cuda.h>

extern void cargaArchivoLocs(char *);
extern void cargaArchivoRecs(char *);

extern void calculaDLRv2(float radio);

void inicializacionPDIC();

const float RT = 6371008.8;

PLocalidad ploc;
PRecurso prec;
PDiccionario pdic;

int cantiloc;
int cantirec;

/**
 *
 */
int main(int cargs, char ** args) {

	if (cargs < 5) {
		fprintf(stderr, "No estan completos los parámetros:\n");
		fprintf(stderr,
				"\ndistlocrec.exe CantiLocs CantiRecs ArchivoLoc ArchivoRec\n\n");
		fprintf(stderr, "\t CantiLocs:\tCantidad de localidades\n");
		fprintf(stderr, "\t CantiRecs:\tCantidad de recursos\n");
		fprintf(stderr, "\t ArchivoLoc:\tArchivo de localidades\n");
		fprintf(stderr, "\t ArchivoRec:\tArchivo de recursos\n");
		return 1;
	}

	cantiloc = atoi(*(args + 1));
	cantirec = atoi(*(args + 2));
	char * archlocs = *(args + 3);
	char * archrecs = *(args + 4);

	pdic = (PDiccionario) malloc(sizeof(sDiccionario) * CANTI_TIPO_REC);
	inicializacionPDIC();

	prec = (PRecurso) malloc(sizeof(sRecurso) * cantirec);
	ploc = (PLocalidad) malloc(sizeof(sLocalidad) * cantiloc);

	if (prec != NULL && ploc != NULL) {

		cargaArchivoLocs(archlocs);
		cargaArchivoRecs(archrecs);

		calculaDLRv2(RT);


	}

	free(ploc);
	free(prec);
	free(pdic);

	cudaDeviceReset();
	return 0;
}

/**
 *
 */
void inicializacionPDIC() {
	int ii;
	for (ii = 0; ii < CANTI_TIPO_REC; ii++) {
		(pdic + ii)->nombre[0] = '\0';
	}
}
