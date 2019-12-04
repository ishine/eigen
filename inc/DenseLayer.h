#pragma once
#include "Utility.h"

struct DenseLayer {
	/**
	 * 
	 */

	Matrix wDense;
	Vector bDense;

	Vector& operator()(const Vector &x, Vector &ret);
	Vector& operator()(Vector &x);

	Matrix& operator()(Matrix &x, Matrix &wDense);
	Matrix& operator()(Matrix &x);

	DenseLayer(BinaryReader &dis, bool bias = true);
};
