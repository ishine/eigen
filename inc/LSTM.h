#include "RNN.h"
#include "matrix.h"

struct LSTM: RNN {

	Matrix Wxi;
	Matrix Whi;
	Matrix Wci;
	Vector bi;

	Matrix Wxf;
	Matrix Whf;
	Matrix Wcf;
	Vector bf;

	Matrix Wxc;
	Matrix Whc;
	Vector bc;

	Matrix Wxo;
	Matrix Who;
	Matrix Wco;
	Vector bo;

	Matrix Why;
	Vector by;

	VectorActivator sigmoid;
	VectorActivator tanh;

	LSTM(Matrix Wxi, Matrix Wxf, Matrix Wxc, Matrix Wxo, Matrix Whi, Matrix Whf,
			Matrix Whc, Matrix Who, Vector bi, Vector bf, Vector bc, Vector bo);
	Vector& call(const Matrix &x, Vector &h);
	Vector& activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h,
			Vector &c);

	LSTM(BinaryReader &dis);
	Matrix& call_return_sequences(const Matrix &x, Matrix &arr);
	Matrix& call_return_sequences_reverse(const Matrix &x, Matrix &arr);
	Vector& call_reverse(const Matrix &x, Vector &h);
};
