#ifndef RUNN_H
#define RUNN_H

// FLAGS:
// -MAT_USE_DOUBLE

#include <stddef.h>

// MAT
#ifdef MAT_USE_DOUBLE
	#define NN_DATA_TYPE double
#else
	#define NN_DATA_TYPE float
#endif

typedef struct Mat
{
	size_t rows;
	size_t cols;
	NN_DATA_TYPE *data;
} Mat;

#define MAT_ALLOC_FAILED(mat) (mat.data==NULL)

#define MATAT(mat,r,c) ((mat).data[r*(mat).cols+c])

// WARNING: Do not alloc twice
bool MatAlloc(Mat *mat, size_t rows, size_t cols);

void MatFree(Mat *mat);

void MatLog(Mat *mat);

void MatRandomize(Mat *mat, NN_DATA_TYPE from, NN_DATA_TYPE to);

void MatAssign(Mat *mat, NN_DATA_TYPE data[]);

// NET (NEURAL NETWORK)

// *ACTIV*

typedef struct NetActiv
{
	NN_DATA_TYPE (*func)(NN_DATA_TYPE);
	NN_DATA_TYPE (*deriv)(NN_DATA_TYPE);
} NetActiv;

// NULL
#define ACTIV_NULL (NetActiv){NULL,NULL}

// Linear
NN_DATA_TYPE ActivLinear(NN_DATA_TYPE val);
NN_DATA_TYPE ActivLinearDeriv(NN_DATA_TYPE val);
#define ACTIV_LINEAR (NetActiv){ActivLinear,ActivLinearDeriv}

// Sigmoid
NN_DATA_TYPE ActivSigmoid(NN_DATA_TYPE val);
NN_DATA_TYPE ActivSigmoidDeriv(NN_DATA_TYPE val);
#define ACTIV_SIGMOID (NetActiv){ActivSigmoid,ActivSigmoidDeriv}

// Tanh 
NN_DATA_TYPE ActivTanh(NN_DATA_TYPE val);
NN_DATA_TYPE ActivTanhDeriv(NN_DATA_TYPE val);
#define ACTIV_TANH (NetActiv){ActivTanh,ActivTanhDeriv}

// ReLU 
NN_DATA_TYPE ActivReLU(NN_DATA_TYPE val);
NN_DATA_TYPE ActivReLUDeriv(NN_DATA_TYPE val);
#define ACTIV_RELU (NetActiv){ActivReLU,ActivReLUDeriv}

// *LOSS*

NN_DATA_TYPE LossMeanSquaredError(Mat act, Mat exp);
void LossMeanSquaredErrorDeriv(Mat act, Mat exp, Mat *out);

// *LAYER*

typedef struct NetLayer
{
	size_t size;
	size_t nextSize;
	Mat weights;    // next x size
	Mat biases;     // next x 1
	Mat denseIn;    // size x 1
	Mat activIn;    // next x 1
	NetActiv activ;
} NetLayer;

// WARNING: Do not alloc twice
bool NetLayerAlloc(NetLayer *layer, size_t size, size_t nextSize, NetActiv activ);

void NetLayerFree(NetLayer *layer);

bool NetLayerForward(NetLayer *layer, NN_DATA_TYPE in[], NN_DATA_TYPE out[]);

// *NET*

typedef struct Net
{
	size_t    layerCount;
	NetLayer *layers;
} Net;

#endif /* RUNN_H */
