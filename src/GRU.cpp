#include "GRU.h"

Vector& GRU::call(const Matrix &x, Vector &h) {
	h = Vector::Zero(x.cols());
	for (int t = 0; t < x.rows(); ++t) {
		h = activate(x.row(t), h);
	}
	return h;
}

Vector& GRU::call_reverse(const Matrix &x, Vector &h) {
	h = Vector::Zero(x.cols());
	for (int t = x.rows() - 1; t >= 0; --t) {
		h = activate(x.row(t), h);
	}
	return h;
}

Vector& GRU::call(const Matrix &x, Vector &h, vector<vector<double>> &arr) {
	h = Vector::Zero(x.cols());
//	arr.push_back(convert2vector(this->Wxu, 0));
//	arr.push_back(convert2vector(this->Wxr, 0));
//	arr.push_back(convert2vector(this->Wxh, 0));
//	arr.push_back(convert2vector(this->Whu, 0));
//	arr.push_back(convert2vector(this->Whr, 0));
//	arr.push_back(convert2vector(this->Whh, 0));
//	arr.push_back(convert2vector(this->bu));
//	arr.push_back(convert2vector(this->br));
//	arr.push_back(convert2vector(this->bh));

	for (int t = 0; t < x.rows(); ++t) {
//		arr.push_back(convert2vector(h));
		h = activate(x.row(t), h);
	}
	return h;
}

Vector& GRU::call_reverse(const Matrix &x, Vector &h,
		vector<vector<double>> &arr) {
	h = Vector::Zero(x.cols());
//	arr.push_back(convert2vector(this->Wxu, 0));
//	arr.push_back(convert2vector(this->Wxr, 0));
//	arr.push_back(convert2vector(this->Wxh, 0));
//	arr.push_back(convert2vector(this->Whu, 0));
//	arr.push_back(convert2vector(this->Whr, 0));
//	arr.push_back(convert2vector(this->Whh, 0));
//
//	arr.push_back(convert2vector(this->bu));
//	arr.push_back(convert2vector(this->br));
//	arr.push_back(convert2vector(this->bh));

	for (int t = x.rows() - 1; t >= 0; --t) {
//		arr.push_back(convert2vector(h));
		h = activate(x.row(t), h);
	}

	return h;
}

Matrix& GRU::call_return_sequences(const Matrix &x, Matrix &ret) {
	Vector h = Vector::Zero(x.cols());
	return ret;
}

Matrix& GRU::call_return_sequences_reverse(const Matrix &x, Matrix &ret) {
	Vector h = Vector::Zero(x.cols());
	return ret;
}

Vector& GRU::activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h,
		vector<vector<double>> &arr) {
	Vector tmp = x * Wxr;
	arr.push_back(convert2vector(tmp)); //3

	tmp = h * Whr;
	arr.push_back(convert2vector(tmp)); //4

	arr.push_back(convert2vector(br)); //5

	Vector r = x * Wxr + h * Whr + br;
	arr.push_back(convert2vector(r));
	r = sigmoid(r);
//	arr.push_back(convert2vector(r));

	Vector u = x * Wxu + h * Whu + bu;
	u = sigmoid(u);
	arr.push_back(convert2vector(u));

	Vector gh = x * Wxh + r.cwiseProduct(h) * Whh + bh;
	gh = tanh(gh);
	arr.push_back(convert2vector(gh));

	h = (Vector::Ones(u.cols()) - u).cwiseProduct(gh) + u.cwiseProduct(h);
	return h;
}

Vector& GRU::activate(const Eigen::Block<const Matrix, 1, -1, 1> &x,
		Vector &h) {
	Vector r = x * Wxr + h * Whr + br;
	r = sigmoid(r);

	Vector u = x * Wxu + h * Whu + bu;
	u = sigmoid(u);

	Vector gh = x * Wxh + r.cwiseProduct(h) * Whh + bh;
	gh = tanh(gh);

	h = (Vector::Ones(u.cols()) - u).cwiseProduct(gh) + u.cwiseProduct(h);
	return h;
}

GRU::GRU(BinaryReader &dis) {
	dis.read(Wxu);
	dis.read(Wxr);
	dis.read(Wxh);

	dis.read(Whu);
	dis.read(Whr);
	dis.read(Whh);

	dis.read(bu);
	dis.read(br);
	dis.read(bh);

	this->sigmoid = ::hard_sigmoid;
	this->tanh = ::tanh;
	this->softmax = ::softmax;

}

vector<vector<vector<double>>> &GRU::weight(
		vector<vector<vector<double>>> &arr) {
	arr.push_back(convert2vector(this->Wxu));
	arr.push_back(convert2vector(this->Wxr));
	arr.push_back(convert2vector(this->Wxh));

	arr.push_back(convert2vector(this->Whu));
	arr.push_back(convert2vector(this->Whr));
	arr.push_back(convert2vector(this->Whh));

	return arr;
}
