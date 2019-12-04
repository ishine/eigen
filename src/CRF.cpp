#include "CRF.h"

CRF::CRF(Matrix kernel, Matrix G, Vector bias, Vector left_boundary,
		Vector right_boundary) {
	this->bias = bias;

	this->G = G;

	this->kernel = kernel;

	this->left_boundary = left_boundary;

	this->right_boundary = right_boundary;
}

Matrix& CRF::viterbi_one_hot(const Matrix &X, Matrix &oneHot) {
	vector<int> label;
	label = call(X, label);
	int n = bias.cols();
	Matrix eye = Matrix::Identity(n, n);
	int m = label.size();
	oneHot.resize(m, n);
	for (int i = 0; i < m; ++i) {
		oneHot.row(i) = eye.row(label[i]);
	}
	return oneHot;
}

vector<int>& CRF::call(const Matrix &X, vector<int> &best_paths) {
	//add a row vector to a matrix
	Matrix x = X * kernel;
	x.rowwise() += bias;

	x.row(0) += left_boundary;

	int length = x.rows();
	x.row(length - 1) += right_boundary;

	int i = 0;
	Vector min_energy = x.row(i++);

	vector<vector<int>> argmin_tables(length);

	while (i < length) {
		Matrix energy = G;
		energy.rowwise() += min_energy;

		min_energy = min(energy, min_energy, argmin_tables[i - 1]);
		min_energy += x.row(i++);
	}

	int argmin;
	min_energy.minCoeff(&argmin);

	assert(i == length);

	best_paths.resize(length);
	best_paths[--i] = argmin;

	for (--i; i >= 0; --i) {
		argmin = argmin_tables[i][argmin];
		best_paths[i] = argmin;
	}
	return best_paths;
}

CRF::CRF(BinaryReader &dis) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	dis.read(kernel);
	dis.read(G);
	dis.read(bias);
	dis.read(left_boundary);
	dis.read(right_boundary);
}

