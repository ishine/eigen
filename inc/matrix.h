#pragma once
#include "Utility.h"

Matrix& exp(Matrix &x);
Vector& exp(Vector &x);

Matrix& hard_sigmoid(Matrix &x);
Vector& hard_sigmoid(Vector &x);

Matrix& logistic(Matrix &x);
Vector& logistic(Vector &x);

Matrix& tanh(Matrix &x);
Vector& tanh(Vector &x);

Matrix& relu(Matrix &x);
Vector& relu(Vector &x);

Matrix& gelu(Matrix &x);
Vector& gelu(Vector &x);
Tensor& gelu(Tensor &x);
vector<Vector>& gelu(vector<Vector> &x);

Matrix& softmax(Matrix &x);
Vector& softmax(Vector &x);

Tensor& softmax(Tensor &x);
vector<Vector>& softmax(vector<Vector> &x);

Matrix& l2_normalize(Matrix &f);
Vector& l2_normalize(Vector &f);

Matrix& inverse(Matrix &x);
Vector& inverse(Vector &x);

vector<Vector>& extract(const Tensor &x, int index);
vector<Vector>& extract(const Tensor &x, int index, vector<Vector> &out);

typedef Vector& (*VectorActivator)(Vector &x);
typedef Matrix& (*MatrixActivator)(Matrix &x);
typedef Tensor& (*TensorActivator)(Tensor &x);

MatrixI& operator !=(MatrixI &x, int y);
MatrixI& operator ==(MatrixI &x, int y);
vector<VectorI>& operator !=(vector<VectorI> &x, int y);
vector<VectorI>& operator ==(vector<VectorI> &x, int y);
VectorI&operator !=(VectorI&x, int y);
VectorI&operator ==(VectorI&x, int y);

vector<Vector>& mean(const Tensor &x);
vector<double>& mean(const vector<Vector> &x);

Tensor& sqrt(Tensor &x);
vector<Vector>& sqrt(vector<Vector> &x);
vector<double>& sqrt(vector<double> &x);

Tensor& square(Tensor &x);
vector<Vector>& square(vector<Vector> &x);

template<typename _Tx, typename _Ty>
vector<_Tx>& operator +(vector<_Tx> &x, const _Ty &y) {
	return x += y;
}

template<typename _Tx, typename _Ty>
vector<_Tx>& operator +(const vector<_Tx> &x, const _Ty &y) {
	static vector<_Tx> out;
	out = x;
	return out += y;
}

template<typename _Tx, typename _Ty>
vector<_Tx>& operator -(vector<_Tx> &x, const _Ty &y) {
	return x -= y;
}

template<typename _Tx, typename _Ty>
vector<_Tx>& operator -(const vector<_Tx> &x, const _Ty &y) {
	static vector<_Tx> out;
	out = x;
	return out -= y;
}

template<typename _Tx, typename _Ty>
vector<_Tx>& operator *(vector<_Tx> &x, const _Ty &y) {
	return x *= y;
}

template<typename _Tx, typename _Ty>
vector<_Tx>& operator *(const vector<_Tx> &x, const _Ty &y) {
	static vector<_Tx> out;
	out = x;
	return out *= y;
}

template<typename _Tx, typename _Ty>
vector<_Tx>& operator /(vector<_Tx> &x, const _Ty &y) {
	return x /= y;
}

template<typename _Tx, typename _Ty>
vector<_Tx>& operator /(const vector<_Tx> &x, const _Ty &y) {
	static vector<_Tx> out;
	out = x;
	return out /= y;
}

Matrix& operator +=(Matrix &x, double y);
Tensor& operator +=(Tensor &x, double y);
Tensor& operator +=(Tensor &x, const Tensor &y);
vector<Vector>& operator +=(vector<Vector> &x, double y);
vector<double>& operator +=(vector<double> &x, double y);
Tensor& operator +=(Tensor &x, const Vector &y);
vector<Vector>& operator +=(vector<Vector> &x, const Vector &y);
vector<Vector>& operator +=(vector<Vector> &x, const vector<Vector> &y);

MatrixI& operator -(int x, MatrixI &y);
MatrixI& operator -=(MatrixI &x, int y);
Tensor& operator -=(Tensor &x, const vector<Vector> &y);
Tensor& operator -=(Tensor &x, const Tensor &y);
vector<Vector>& operator -=(vector<Vector> &x, const vector<double> &y);
vector<Vector>& operator -=(vector<Vector> &x, const vector<Vector> &y);
MatrixI& operator -=(MatrixI &x, int y);
vector<VectorI>& operator -=(vector<VectorI> &x, int y);

vector<Vector> &operator *(double x, const vector<VectorI> &y);
//Matrix &operator *(double x, const MatrixI &y);
Tensor& operator *=(Tensor &x, const Vector &y);
Tensor& operator *=(Tensor &x, const Matrix &y);
vector<Vector>& operator *=(vector<Vector> &x, const Matrix &y);
vector<VectorI>& operator *=(vector<VectorI> &x, int y);

Tensor& operator /=(Tensor &x, const Tensor &y);
Tensor& operator /=(Tensor &x, const vector<Vector> &y);
Tensor& operator /=(Tensor &x, double y);
vector<Vector>& operator /=(vector<Vector> &x, double y);
vector<Vector>& operator /=(vector<Vector> &x, const vector<double> &y);

Tensor& batch_dot(const Tensor &x, const Tensor &y, bool transpose = false);

vector<Vector>& batch_dot(vector<Vector> &x, const Tensor &y, bool transpose =
		false);
