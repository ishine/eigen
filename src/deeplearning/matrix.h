#pragma once
#include "utility.h"

Vector& min(const Matrix &x, Vector &m, vector<int> &argmin);
Vector& max(const Matrix &x, Vector &m, vector<int> &argmax);

Vector& min(const Matrix &x);
Vector& max(const Matrix &x);

Matrix& exp(Matrix &x);
Vector& exp(Vector &x);

Matrix& hard_sigmoid(Matrix &x);
Vector& hard_sigmoid(Vector &x);

Matrix& sigmoid(Matrix &x);
Vector& sigmoid(Vector &x);

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

enum class Activator : int {
	linear, softmax, l2_normalize, relu, gelu, hard_sigmoid, sigmoid, tanh
};

struct Activation {

	Activator act;
	template<typename _Ty>
	_Ty& operator ()(_Ty &x) {
		switch (act) {
		case Activator::linear:
			return x;
		case Activator::softmax:
			return softmax(x);
		case Activator::relu:
			return relu(x);
		case Activator::gelu:
			return gelu(x);
		case Activator::hard_sigmoid:
			return hard_sigmoid(x);
		case Activator::sigmoid:
			return sigmoid(x);
		case Activator::tanh:
			return tanh(x);
		default:
			return x;
		}
	}
};

MatrixI& operator !=(MatrixI &x, int y);
MatrixI& operator ==(MatrixI &x, int y);
vector<VectorI>& operator !=(vector<VectorI> &x, int y);
vector<VectorI>& operator ==(vector<VectorI> &x, int y);
VectorI& operator !=(VectorI &x, int y);
VectorI& operator ==(VectorI &x, int y);

vector<Vector>& mean(const Tensor &x);
Vector& mean(const Matrix &x);
vector<double>& mean(const vector<Vector> &x);

Tensor& sqrt(Tensor &x);
Matrix& sqrt(Matrix &x);
Vector& sqrt(Vector &x);
vector<Vector>& sqrt(vector<Vector> &x);
vector<double>& sqrt(vector<double> &x);

Tensor& square(Tensor &x);
Matrix& square(Matrix &x);
Vector& square(Vector &x);
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

Vector& operator +=(Vector &x, double y);
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
Vector& operator -=(Vector &x, double y);
Vector& operator -(Vector &x, double y);

vector<VectorI>& operator -=(vector<VectorI> &x, int y);

vector<Vector>& operator *(double x, const vector<VectorI> &y);
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

Tensor& batch_dot(Tensor &x, const Tensor &y, bool transpose = false);

vector<Vector>& batch_dot(vector<Vector> &x, const Tensor &y, bool transpose =
		false);

#include "../Eigen/src/Core/DenseBase.h"
using Eigen::DenseBase;
using Eigen::Index;

template<typename XprType>
struct CommaAssignment {

	template<typename OtherDerived>
	EIGEN_DEVICE_FUNC
	inline CommaAssignment(const XprType &xpr, DenseBase<OtherDerived> &other) :
			m_xpr(xpr), m_row(0), m_col(other.cols()), m_currentBlockRows(
					other.rows()) {
		other = m_xpr.block(0, 0, other.rows(), other.cols());
	}

	/* selects a matrix expression from the source matrix */
	template<typename OtherDerived>
	EIGEN_DEVICE_FUNC
	CommaAssignment& operator,(DenseBase<OtherDerived> &other) {
		if (m_col == m_xpr.cols()
				&& (other.cols() != 0 || other.rows() != m_currentBlockRows)) {
			m_row += m_currentBlockRows;
			m_col = 0;
			m_currentBlockRows = other.rows();
			eigen_assert(
					m_row + m_currentBlockRows <= m_xpr.rows()
							&& "Too many rows passed to comma initializer (operator<<)");
		}
		eigen_assert(
				(m_col + other.cols() <= m_xpr.cols())
						&& "Too many coefficients passed to comma initializer (operator<<)");
		eigen_assert(m_currentBlockRows == other.rows());
		other = m_xpr.template block<OtherDerived::RowsAtCompileTime,
				OtherDerived::ColsAtCompileTime>(m_row, m_col, other.rows(),
				other.cols());
		m_col += other.cols();
		return *this;
	}

	const XprType &m_xpr;           // source expression
	Index m_row;              // current row id
	Index m_col;              // current col id
	Index m_currentBlockRows; // current block height
};

template<typename Derived, typename OtherDerived>
CommaAssignment<Derived> operator>>(const DenseBase<Derived> &self,
		DenseBase<OtherDerived> &other) {
	return CommaAssignment<Derived>(*static_cast<const Derived*>(&self), other);
}

Matrix& add(Matrix &x, const Vector &y);
Matrix& sub(Matrix &x, const Vector &y);
Matrix& div(Matrix &x, const Vector &y);
Matrix& mul(Matrix &x, const Vector &y);
Vector& mul(Vector &x, const Vector &y);

Matrix& addt(Matrix &x, const Vector &y);
Matrix& subt(Matrix &x, const Vector &y);
Matrix& divt(Matrix &x, const Vector &y);
Matrix& mult(Matrix &x, const Vector &y);
