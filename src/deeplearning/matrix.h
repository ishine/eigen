#pragma once
#include "utility.h"

Vector& min(const Matrix &x, Vector &m, vector<int> &argmin);
Vector& max(const Matrix &x, Vector &m, vector<int> &argmax);

Vector min(const Matrix &x);
Vector max(const Matrix &x);
int max(const vector<int>&x, int&index);

Tensor& exp(Tensor &x);
Matrix& exp(Matrix &x);
Vector& exp(Vector &x);
vector<Vector>& exp(vector<Vector> &x);

Tensor& hard_sigmoid(Tensor &x);
Matrix& hard_sigmoid(Matrix &x);
Vector& hard_sigmoid(Vector &x);
vector<Vector>& hard_sigmoid(vector<Vector> &x);

Tensor& sigmoid(Tensor &x);
Matrix& sigmoid(Matrix &x);
Vector& sigmoid(Vector &x);
vector<Vector>& sigmoid(vector<Vector> &x);

Tensor& tanh(Tensor &x);
Matrix& tanh(Matrix &x);
Vector& tanh(Vector &x);
vector<Vector>& tanh(vector<Vector> &x);

//\text{ELU}(x) = \max\left(0,x\right) + \min\left(0, \alpha * \left(\exp\left(x\right) - 1\right)\right)
//https://www.codecogs.com/eqnedit.php
Tensor& elu(Tensor &x);
Matrix& elu(Matrix &x);
Vector& elu(Vector &x);
vector<Vector>& elu(vector<Vector> &x);

Tensor& relu(Tensor &x);
Matrix& relu(Matrix &x);
Vector& relu(Vector &x);
vector<Vector>& relu(vector<Vector> &x);

Matrix& gelu(Matrix &x);
Vector& gelu(Vector &x);
Tensor& gelu(Tensor &x);
vector<Vector>& gelu(vector<Vector> &x);

double logsumexp(Vector &x);

Matrix& log_softmax(Matrix &x);
Vector& log_softmax(Vector &x);
Tensor& log_softmax(Tensor &x);
vector<Vector>& log_softmax(vector<Vector> &x);

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
	linear, softmax, l2_normalize, relu, gelu, hard_sigmoid, sigmoid, tanh, elu,
};

struct Activation {

	Activator act;
	template<typename _Ty>
	_Ty& operator ()(_Ty &x) const {
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
		case Activator::elu:
			return elu(x);

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

vector<Vector> mean(const Tensor &x);
Vector mean(const Matrix &x);
vector<double> mean(const vector<Vector> &x);

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
vector<_Tx> operator +(const vector<_Tx> &x, const _Ty &y) {
	vector<_Tx> out;
	out = x;
	return out += y;
}

template<typename _Tx, typename _Ty>
vector<_Tx>& operator -(vector<_Tx> &x, const _Ty &y) {
	return x -= y;
}

template<typename _Tx, typename _Ty>
vector<_Tx> operator -(const vector<_Tx> &x, const _Ty &y) {
	vector<_Tx> out;
	out = x;
	return out -= y;
}

template<typename _Tx, typename _Ty>
vector<_Tx>& operator *(vector<_Tx> &x, const _Ty &y) {
	return x *= y;
}

template<typename _Tx, typename _Ty>
vector<_Tx> operator *(const vector<_Tx> &x, const _Ty &y) {
	vector<_Tx> out;
	out = x;
	return out *= y;
}

template<typename _Tx, typename _Ty>
vector<_Tx>& operator /(vector<_Tx> &x, const _Ty &y) {
	return x /= y;
}

template<typename _Tx, typename _Ty>
vector<_Tx> operator /(const vector<_Tx> &x, const _Ty &y) {
	vector<_Tx> out;
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
Tensor& operator +=(Tensor &x, const Matrix &y);
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

vector<Vector> operator *(double x, const vector<VectorI> &y);
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

//precondition: x, y are of the same shape!
Matrix dot(const Tensor &x, const Tensor &y);
//Tensor& dot(const Tensor &x, const Tensor &y, Tensor &z, int z_dimension);

Matrix broadcast(const Vector &x, int rows);

Matrix broadcast(const Eigen::Block<Matrix, 1, -1, 1> &x, int rows);

Matrix broadcast(const Eigen::Block<const Matrix, 1, -1, 1> &x, int rows);

template<int, int, int> Tensor transpose(const Tensor &x);

template<>
Tensor transpose<0, 2, 1>(const Tensor &x);

template<>
Tensor transpose<1, 0, 2>(const Tensor &x);

template<>
Tensor transpose<1, 2, 0>(const Tensor &x);

template<>
Tensor transpose<2, 0, 1>(const Tensor &x);

template<>
Tensor transpose<2, 1, 0>(const Tensor &x);

Tensor ndarray(int x_shape, int y_shape, int z_shape);
Matrix ndarray(int x_shape, int y_shape);
Vector ndarray(int x_shape);
