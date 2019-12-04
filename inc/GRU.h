/**
 * implimentation of Gated Recurrent Unit
 * 
 * @author Cosmos
 *
 */
#include "RNN.h"
#include "matrix.h"

struct GRU: RNN {

	VectorActivator sigmoid;
	VectorActivator tanh;
	VectorActivator softmax;

	Matrix Wxu;
	Matrix Whu;
	Vector bu;

	Matrix Wxr;
	Matrix Whr;
	Vector br;

	Matrix Wxh;
	Matrix Whh;
	Vector bh;

	Vector& call(const Matrix &x, Vector &h);
	Vector& call_reverse(const Matrix &x, Vector &h);
	Vector& call(const Matrix &x, Vector &h, vector<vector<double>> &arr);
	Vector& call_reverse(const Matrix &x, Vector &h,
			vector<vector<double>> &arr);

	Matrix& call_return_sequences(const Matrix &x, Matrix &ret);
	Matrix& call_return_sequences_reverse(const Matrix &x, Matrix &ret);

	Vector& activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h);
	Vector& activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h,
			vector<vector<double>> &arr);

	vector<vector<vector<double>>> &weight(vector<vector<vector<double>>> &arr);

	GRU(BinaryReader &dis);
};
