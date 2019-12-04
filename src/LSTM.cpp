#include "LSTM.h"

LSTM::LSTM(Matrix Wxi, Matrix Wxf, Matrix Wxc, Matrix Wxo, Matrix Whi,
		Matrix Whf, Matrix Whc, Matrix Who, Vector bi, Vector bf, Vector bc,
		Vector bo) {
	this->Wxi = Wxi;
	this->Wxf = Wxf;
	this->Wxc = Wxc;
	this->Wxo = Wxo;

	this->Whi = Whi;
	this->Whf = Whf;
	this->Whc = Whc;
	this->Who = Who;

	this->bi = bi;
	this->bf = bf;
	this->bc = bc;
	this->bo = bo;

	this->sigmoid = ::hard_sigmoid;
	this->tanh = ::tanh;
}

Vector& LSTM::call(const Matrix &x, Vector &h) {
	Vector c;
	h = c = Vector::Zero(x.cols());

	for (int t = 0; t < x.rows(); ++t) {
		activate(x.row(t), h, c);
	}

	return h;
}

Vector& LSTM::activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h,
		Vector &c) {
	Vector i = x * Wxi + h * Whi + bi;
	Vector f = x * Wxf + h * Whf + bf;
	Vector _c = x * Wxc + h * Whc + bc;

	_c = sigmoid(f).cwiseProduct(c) + sigmoid(i).cwiseProduct(tanh(_c));

	Vector o = x * Wxo + h * Who + bo;

	c = _c;
	h = sigmoid(o).cwiseProduct(tanh(_c));
	return h;
}

LSTM::LSTM(BinaryReader &dis) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	dis.read(Wxi);
	dis.read(Wxf);
	dis.read(Wxc);
	dis.read(Wxo);

	dis.read(Whi);
	dis.read(Whf);
	dis.read(Whc);
	dis.read(Who);

	dis.read(bi);
	dis.read(bf);
	dis.read(bc);
	dis.read(bo);
	sigmoid = ::hard_sigmoid;
	tanh = ::tanh;
}

Matrix& LSTM::call_return_sequences(const Matrix &x, Matrix &arr) {
	arr.resize(x.rows(), x.cols());
	Vector h, c;
	h = c = Vector::Zero(x.cols());

	for (int t = 0, length = x.rows(); t < length; ++t) {
		arr.row(t) = activate(x.row(t), h, c);
	}

	return arr;
}

Matrix& LSTM::call_return_sequences_reverse(const Matrix &x, Matrix &arr) {
	arr.resize(x.rows(), x.cols());
	Vector h, c;
	h = c = Vector::Zero(x.cols());

	for (int t = x.rows() - 1; t >= 0; --t) {
		arr.row(t) = activate(x.row(t), h, c);
	}

	return arr;
}

Vector& LSTM::call_reverse(const Matrix &x, Vector &h) {
	Vector c;
	h = c = Vector::Zero(x.cols());

	for (int t = x.rows() - 1; t >= 0; --t) {
		activate(x.row(t), h, c);
	}

	return h;
}
