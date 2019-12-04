#include "Utility.h"
#include "matrix.h"

struct CRF {
	Vector bias;
	Matrix G;
	Matrix kernel;
	Vector left_boundary;
	Vector right_boundary;
	VectorActivator activation = nullptr;

	CRF(Matrix kernel, Matrix G, Vector bias, Vector left_boundary,
			Vector right_boundary);

	Matrix& viterbi_one_hot(const Matrix &X, Matrix &oneHot);

	vector<int>& call(const Matrix &X, vector<int> &best_paths);
	CRF(BinaryReader &dis);
};

