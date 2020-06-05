#include <math.h>
#include "matrix.h"
#include "../std/lagacy.h"

Vector& aggregate(const Matrix &x, Vector &v, vector<int> &arg,
		double (Matrix::ConstRowXpr::*aggregate)(int*) const) {
	int m = x.rows();
	v.resize(m);
	arg.resize(m);
	for (int i = 0; i < m; ++i) {
		v[i] = (x.row(i).*aggregate)(&arg[i]);
	}
	return v;
}

Vector& max(const Matrix &x, Vector &max, vector<int> &argmax) {
	return aggregate(x, max, argmax, &Matrix::ConstRowXpr::maxCoeff);
}

Vector& min(const Matrix &x, Vector &min, vector<int> &argmin) {
	return aggregate(x, min, argmin, &Matrix::ConstRowXpr::minCoeff);
}

int max(const vector<int> &x, int &index) {
	int max = std::numeric_limits<int>::min();
	for (int i = 0, size = x.size(); i < size; ++i) {
		if (x[i] > max) {
			max = x[i];
			index = i;
		}
	}
	return max;
}

Vector max(const Matrix &x) {
	Vector y;
	vector<int> argmax;
	return max(x, y, argmax);
}

Vector min(const Matrix &x) {
	Vector y;
	vector<int> argmax;
	return min(x, y, argmax);
}
//eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
//http://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
//https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#abstract

//double hard_sigmoid(double x) {
//	if (x < -2.5)
//		return 0;
//	if (x > 2.5)
//		return 1;
//	return 0.2 * x + 0.5;
//}
//

double elu(double x, double alpha) {
	if (x >= 0)
		return x;
	return alpha * (exp(x) - 1);
}

double elu(double x) {
	if (x >= 0)
		return x;
	return exp(x) - 1;
}

double relu(double x) {
	return x < 0 ? 0 : x;
}

double inverse(double x) {
	return 1 / x;
}

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

Vector& function(Vector &x, double (*fptr)(double)) {
	int cols = x.cols();

	for (int j = 0; j < cols; ++j) {
		x[j] = fptr(x[j]);
	}
	return x;
}

Matrix& function(Matrix &x, double (*fptr)(double)) {
	int rows = x.rows();
	int cols = x.cols();

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			x(i, j) = fptr(x(i, j));
		}
	}
	return x;
}

vector<Vector>& function(vector<Vector> &x, double (*fptr)(double)) {
	int batch_size = x.size();

	for (int k = 0; k < batch_size; ++k) {
		function(x[k], fptr);
	}
	return x;
}

Tensor& function(Tensor &x, double (*fptr)(double)) {
	int batch_size = x.size();

	for (int k = 0; k < batch_size; ++k) {
		function(x[k], fptr);
	}
	return x;
}

Matrix& sigmoid(Matrix &x) {
	return function(x, sigmoid);
}

Tensor& sigmoid(Tensor &x) {
	return function(x, sigmoid);
}

Vector& sigmoid(Vector &x) {
	return function(x, sigmoid);
}

vector<Vector>& sigmoid(vector<Vector> &x) {
	return function(x, sigmoid);
}

Matrix& hard_sigmoid(Matrix &x) {
	return function(x, hard_sigmoid);
}

Tensor& hard_sigmoid(Tensor &x) {
	return function(x, hard_sigmoid);
}

Vector& hard_sigmoid(Vector &x) {
	return function(x, hard_sigmoid);
}

vector<Vector>& hard_sigmoid(vector<Vector> &x) {
	return function(x, hard_sigmoid);
}

Matrix& tanh(Matrix &x) {
	return function(x, std::tanh);
}

Tensor& tanh(Tensor &x) {
	return function(x, std::tanh);
}

Vector& tanh(Vector &x) {
	return function(x, std::tanh);
}

vector<Vector>& tanh(vector<Vector> &x) {
	return function(x, std::tanh);
}

Tensor& exp(Tensor &x) {
	return function(x, std::exp);
}

Matrix& exp(Matrix &x) {
	return function(x, std::exp);
}

Vector& exp(Vector &x) {
	return function(x, std::exp);
}

vector<Vector>& exp(vector<Vector> &x) {
	return function(x, std::exp);
}

Matrix& relu(Matrix &x) {
	return function(x, relu);
//	return x.cwiseProduct((x.array() > 0.0).matrix().cast<double>());
}

Tensor& relu(Tensor &x) {
	return function(x, relu);
//	return x.cwiseProduct((x.array() > 0.0).matrix().cast<double>());
}

Vector& relu(Vector &x) {
	return function(x, relu);
//	return x.cwiseProduct((x.array() > 0.0).matrix().cast<double>());
}

vector<Vector>& relu(vector<Vector> &x) {
	return function(x, relu);
//	return x.cwiseProduct((x.array() > 0.0).matrix().cast<double>());
}

Matrix& elu(Matrix &x) {
	return function(x, elu);
//	return x.cwiseProduct((x.array() > 0.0).matrix().cast<double>());
}

Tensor& elu(Tensor &x) {
	return function(x, elu);
}

Vector& elu(Vector &x) {
	return function(x, elu);
//	return x.cwiseProduct((x.array() > 0.0).matrix().cast<double>());
}

vector<Vector>& elu(vector<Vector> &x) {
	return function(x, elu);
//	return x.cwiseProduct((x.array() > 0.0).matrix().cast<double>());
}

vector<Vector>& gelu(vector<Vector> &x) {
	return function(x, gelu);
}

Matrix& gelu(Matrix &x) {
	return function(x, gelu);
}

Vector& gelu(Vector &x) {
	return function(x, gelu);
}

Tensor& gelu(Tensor &x) {
	return function(x, gelu);
}

//Matrix& log_softmax(Matrix &x) {
//	__cout(__PRETTY_FUNCTION__)
////	__cout(x);
//	print_shape(x);
//
//	for (int i = 0, size = x.rows(); i < size; ++i) {
//		cout << "processing " << i << endl;
//		auto row = x.row(i);
//		double lambda = row.maxCoeff();
//
//		cout << "lambda " << lambda << endl;
//		row.array() -= lambda;
//
//		cout << "row.array().exp().sum() = " << row.array().exp().sum() << endl;
//		row.array() -= log(row.array().exp().sum());
//	}
//
//	return x;
//}

Matrix& log_softmax(Matrix &x) {
	__cout(__PRETTY_FUNCTION__)
//	__cout(x);
//	print_shape(x);

	for (int i = 0, size = x.rows(); i < size; ++i) {
//		cout << "processing " << i << endl;
		Vector row = x.row(i);

		row -= row.maxCoeff();
		row -= log(exp(row).sum());

		x.row(i) = row;
	}

	return x;
}

Tensor& log_softmax(Tensor &x) {
	__cout(__PRETTY_FUNCTION__)
	for (int i = 0, size = x.size(); i < size; ++i) {
		log_softmax(x[i]);
	}
	return x;
}

double logsumexp(Vector &x) {
	double lambda = x.maxCoeff();
	x -= lambda;
	return lambda + log(exp(x).sum());
}

Vector& log_softmax(Vector &x) {
	double lambda = x.maxCoeff();
	x -= lambda;
	return x - log(exp(x).sum());
}

Matrix& softmax(Matrix &x) {
	int rows = x.rows();

	x = exp(subt(x, max(x)));
	for (int i = 0; i < rows; ++i) {
		x.row(i) /= x.row(i).sum();
	}
	return x;
}

Tensor& softmax(Tensor &x) {
	int batch_size = x.size();

	for (int k = 0; k < batch_size; ++k) {
		softmax(x[k]);
	}
	return x;
}

vector<Vector>& softmax(vector<Vector> &x) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		softmax(x[k]);
	}
	return x;
}

Vector& softmax(Vector &x) {
	x = exp(x -= x.maxCoeff());
	x /= x.sum();
	return x;
}

Matrix& l2_normalize(Matrix &x) {
	int rows = x.rows();

	for (int i = 0; i < rows; ++i) {
		x.row(i) /= x.row(i).norm();
	}
	return x;
}

Vector& l2_normalize(Vector &x) {
	x /= x.norm();
	return x;
}

Matrix& inverse(Matrix &x) {
	return function(x, inverse);
}

Vector& inverse(Vector &x) {
	return function(x, inverse);
}

MatrixI& operator !=(MatrixI &x, int y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] = x[k] != y;
	}

	return x;
}

VectorI& operator !=(VectorI &x, int y) {
	int cols = x.size();
	for (int j = 0; j < cols; ++j) {
		x[j] = x[j] != y;
	}

	return x;
}

MatrixI& operator ==(MatrixI &x, int y) {
	int rows = x.size();
	int cols = x[0].size();

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			x[i][j] = x[i][j] == y ? 1 : 0;
		}
	}
	return x;
}

vector<Vector> mean(const Tensor &x) {
	vector<Vector> mean;
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		mean[k] = x[k].rowwise().mean();
	}
	return mean;
}

Vector mean(const Matrix &x) {
	Vector mean;
	mean = x.rowwise().mean();

	return mean;
}

vector<double> mean(const vector<Vector> &x) {
	vector<double> mean;
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		mean[k] = x[k].mean();
	}
	return mean;
}

Tensor& operator -(Tensor &x, const vector<Vector> &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] -= y[k];
	}
	return x;
}

Tensor& square(Tensor &x) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] = x[k].array().square();
	}
	return x;
}

Matrix& square(Matrix &x) {
	x = x.array().square();

	return x;
}

Vector& square(Vector &x) {
	x = x.array().square();
	return x;
}

vector<Vector>& square(vector<Vector> &x) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] = x[k].array().square();
	}
	return x;
}

vector<Vector>& sqrt(vector<Vector> &x) {
	return function(x, sqrt);
}

Matrix& sqrt(Matrix &x) {
	return function(x, sqrt);
}

Vector& sqrt(Vector &x) {
	return function(x, sqrt);
}

vector<double>& sqrt(vector<double> &x) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] = sqrt(x[k]);
	}
	return x;
}

Tensor& operator +=(Tensor &x, const Vector &y) {
	const auto &y_array = y.array();
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		int rows = x[k].rows();
		for (int j = 0; j < rows; ++j) {
			x[k].row(j).array() += y_array;
		}
	}
	return x;
}

Tensor& operator +=(Tensor &x, const Matrix &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] += y;
	}
	return x;
}

Tensor& operator +=(Tensor &x, const Tensor &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] += y[k];
	}
	return x;
}

vector<Vector>& operator +=(vector<Vector> &x, double y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k].array() += y;
	}
	return x;
}

vector<double>& operator +=(vector<double> &x, double y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] += y;
	}
	return x;

}

vector<Vector>& operator +=(vector<Vector> &x, const Vector &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] += y;
	}
	return x;
}

vector<Vector>& operator +=(vector<Vector> &x, const vector<Vector> &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] += y[k];
	}
	return x;
}

Matrix& operator +=(Matrix &x, double y) {
	x.array() += y;
	return x;
}

Vector& operator +=(Vector &x, double y) {
	x.array() += y;
	return x;
}

MatrixI& operator -(int x, MatrixI &y) {
	int rows = y.size();
	int cols = y[0].size();
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			y[i][j] = x - y[i][j];
		}
	}

	return y;
}

MatrixI& operator -=(MatrixI &x, int y) {

	int rows = x.size();
	for (int i = 0; i < rows; ++i) {
		x[i] -= y;
	}

	return x;
}

VectorI& operator -=(VectorI &x, int y) {
	for (int &t : x) {
		t -= y;
	}

	return x;
}


Vector& operator -=(Vector &x, double y) {
	x.array() -= y;
	return x;
}

Vector& operator -(Vector &x, double y) {
	return x -= y;
}

Tensor& operator -=(Tensor &x, const vector<Vector> &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		int rows = x[k].rows();
		for (int j = 0; j < rows; ++j) {
			x[k].row(j) -= y[k];
		}
	}
	return x;
}

Tensor& operator -=(Tensor &x, const Tensor &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] -= y[k];
	}
	return x;
}

vector<Vector>& operator -=(vector<Vector> &x, const vector<Vector> &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] -= y[k];
	}
	return x;
}

vector<Vector>& operator -=(vector<Vector> &x, const vector<double> &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k].array() -= y[k];
	}
	return x;
}

//vector<Vector> operator *(double x, const MatrixI &y) {
//	vector<Vector> out;
//	int batch_size = y.size();
//	out.resize(batch_size);
//	for (int k = 0; k < batch_size; ++k) {
//		out[k] = y[k] * x;
//	}
//	return out;
//}

Tensor& operator *=(Tensor &x, const Vector &y) {
	const auto &y_array = y.array();
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		int rows = x[k].rows();
		for (int j = 0; j < rows; ++j) {
			x[k].row(j).array() *= y_array;
		}
	}
	return x;
}

Tensor& operator *=(Tensor &x, const Matrix &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] *= y;
	}
	return x;
}

vector<Vector>& operator *=(vector<Vector> &x, const Matrix &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		int rows = x[k].rows();
		for (int j = 0; j < rows; ++j) {
			x[k] *= y;
		}
	}
	return x;
}

MatrixI& operator *=(MatrixI &x, int y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] *= y;
	}
	return x;
}

VectorI& operator *=(VectorI &x, int y) {
	for (int & t : x) {
		t *= y;
	}
	return x;
}

Tensor& operator /=(Tensor &x, const vector<Vector> &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		int rows = x[k].rows();
		for (int j = 0; j < rows; ++j) {
			x[k].row(j).array() /= y[k].array();
		}
	}
	return x;
}

Tensor& operator /=(Tensor &x, double y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] /= y;
	}
	return x;
}

vector<Vector>& operator /=(vector<Vector> &x, double y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] /= y;
	}
	return x;
}

vector<Vector>& operator /=(vector<Vector> &x, const vector<double> &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] /= y[k];
	}
	return x;
}

Tensor& batch_dot(Tensor &x, const Tensor &y, bool transpose) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		if (transpose)
			x[k] *= y[k].transpose();
		else
			x[k] *= y[k];
	}
	return x;
}

vector<Vector>& batch_dot(vector<Vector> &x, const Tensor &y, bool transpose) {
	int batch_size = x.size();

	for (int k = 0; k < batch_size; ++k) {
		if (transpose)
			x[k] *= y[k].transpose();
		else
			x[k] *= y[k];
	}
	return x;
}

vector<Vector>& extract(const Tensor &x, int index) {
	vector<Vector> out;
	return extract(x, index, out);
}

vector<Vector>& extract(const Tensor &x, int index, vector<Vector> &out) {
	int batch_size = x.size();
	out.resize(batch_size);

	for (int k = 0; k < batch_size; ++k) {
		out[k] = x[k].row(index);
	}
	return out;
}

Matrix& subt(Matrix &x, const Vector &y) {
	for (int i = 0; i < x.rows(); ++i) {
		x.row(i).array() -= y(i);
	}
	return x;
}

Matrix& mult(Matrix &x, const Vector &y) {
	for (int i = 0; i < x.rows(); ++i) {
		x.row(i).array() -= y(i);
	}
	return x;
}

Matrix& divt(Matrix &x, const Vector &y) {
	for (int i = 0; i < x.rows(); ++i) {
		x.row(i).array() /= y(i);
	}
	return x;
}

Matrix& addt(Matrix &x, const Vector &y) {
	for (int i = 0; i < x.rows(); ++i) {
		x.row(i).array() += y(i);
	}
	return x;
}

Matrix& sub(Matrix &x, const Vector &y) {
	x.rowwise() -= y;
	return x;
}

Matrix& mul(Matrix &x, const Vector &y) {
	x.array().rowwise() *= y.array();
	return x;
}

Vector& mul(Vector &x, const Vector &y) {
	x.array() *= y.array();
	return x;
}

Matrix& div(Matrix &x, const Vector &y) {
	x.array().rowwise() /= y.array();
	return x;
}

Matrix& add(Matrix &x, const Vector &y) {
	x.rowwise() += y;
	return x;
}

Matrix dot(const Tensor &x, const Tensor &y) {
	Matrix z;
	int n = x.size();
	int m = x[0].rows();
	z.resize(n, m);

	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j) {
			z(i, j) = x[i].row(j) * y[i].row(j).transpose();
		}

	return z;
}

Matrix broadcast(const Vector &x, int rows) {
	Matrix ret;
	ret.resize(rows, x.cols());
	for (int i = 0; i < rows; ++i) {
		ret.row(i) = x;
	}
	return ret;
}

Matrix broadcast(const Eigen::Block<Matrix, 1, -1, 1> &x, int rows) {
	Matrix ret;
	ret.resize(rows, x.cols());
	for (int i = 0; i < rows; ++i) {
		ret.row(i) = x;
	}
	return ret;
}

Matrix broadcast(const Eigen::Block<const Matrix, 1, -1, 1> &x, int rows) {
	Matrix ret;
	ret.resize(rows, x.cols());
	for (int i = 0; i < rows; ++i) {
		ret.row(i) = x;
	}
	return ret;
}

template<>
Tensor transpose<0, 2, 1>(const Tensor &x) {
	int n = x.size();
	Tensor y(n);
	for (int i = 0; i < n; ++i)
		y[i] = x[i].transpose();
	return y;
}

template<>
Tensor transpose<2, 0, 1>(const Tensor &x) {
	__cout(__PRETTY_FUNCTION__)
	int n = x.size();
	int m = x[0].rows();
	int z_dimension = x[0].cols();

//	__cout(n)
//	__cout(m)
//	__cout(z_dimension)
	Tensor y = ndarray(z_dimension, n, m);
	for (int z = 0; z < z_dimension; ++z) {
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				y[z](i, j) = x[i](j, z);
			}
		}
	}
	return y;
}

template<>
Tensor transpose<2, 1, 0>(const Tensor &x) {
	int n = x.size();
	int m = x[0].rows();
	int z_dimension = x[0].cols();
	Tensor y = ndarray(z_dimension, m, n);
	for (int z = 0; z < z_dimension; ++z) {
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				y[z](j, i) = x[i](j, z);
			}
		}
	}
	return y;
}

Tensor ndarray(int x_shape, int y_shape, int z_shape) {
	Tensor t(x_shape);
	for (int i = 0; i < x_shape; ++i) {
		t[i].resize(y_shape, z_shape);
	}
	return t;
}

Matrix ndarray(int x_shape, int y_shape) {
	Matrix t;
	t.resize(x_shape, y_shape);
	return t;
}

Vector ndarray(int x_shape) {
	Vector t;
	t.resize(x_shape);
	return t;
}

MatrixI Zero(int m, int n) {
	MatrixI x(m);
	for (auto &v : x) {
		v.resize(n);
	}
	return x;
}
