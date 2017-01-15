/*
 * distlocrec_cuda.h
 *
 *  Created on: 13/01/2017
 *      Author: alfonso
 */

#include <cuda.h>

#ifndef DISTLOCREC_CUDA_H_
#define DISTLOCREC_CUDA_H_

struct SLOC{
	double x;
	double y;
	double z;
	int id;
}SLoc;


typedef struct SLOC * PSLoc;


struct SREC{
	double x;
	double y;
	double z;
	int tipo;
	int id;
}SRec;


typedef struct SREC * PSRec;


struct SDEST{
	int jmin[CANTI_TIPO_REC];
	double dist[CANTI_TIPO_REC];
};

typedef struct SDEST *PSDest;

#endif /* DISLOCREC_CUDA_H_ */
