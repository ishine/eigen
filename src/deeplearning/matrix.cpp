#include <math.h>
#include "matrix.h"
#include "lagacy.h"

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

Vector& max(const Matrix &x) {
	static Vector y;
	static vector<int> argmax;
	return max(x, y, argmax);
}

Vector& min(const Matrix &x) {
	static Vector y;
	static vector<int> argmax;
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
//double relu(double x) {
//	if (x > 0) {
//		return x;
//	}
//	return 0;
//}

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

Vector& sigmoid(Vector &x) {
	return function(x, sigmoid);
}

Matrix& hard_sigmoid(Matrix &x) {
	return function(x, hard_sigmoid);
}

Vector& hard_sigmoid(Vector &x) {
	return function(x, hard_sigmoid);
}

Matrix& tanh(Matrix &x) {
	return function(x, std::tanh);
}

Vector& tanh(Vector &x) {
	return function(x, std::tanh);
}

Matrix& exp(Matrix &x) {
	return function(x, std::exp);
}

Vector& exp(Vector &x) {
	return function(x, std::exp);
}

Matrix& relu(Matrix &x) {
	return function(x, relu);
//	return x.cwiseProduct((x.array() > 0.0).matrix().cast<double>());
}

Vector& relu(Vector &x) {
	return function(x, relu);
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
	int rows = x.rows();
	int cols = x.cols();

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			x(i, j) = x(i, j) == y ? 0 : 1;
		}
	}
	return x;
}

vector<VectorI>& operator !=(vector<VectorI> &x, int y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] = x[k] != y;
	}

	return x;
}

VectorI& operator !=(VectorI &x, int y) {
	int cols = x.cols();
	for (int j = 0; j < cols; ++j) {
		x[j] = x[j] != y;
	}

	return x;
}

MatrixI& operator ==(MatrixI &x, int y) {
	int rows = x.rows();
	int cols = x.cols();

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			x(i, j) = x(i, j) == y ? 1 : 0;
		}
	}
	return x;
}

vector<Vector>& mean(const Tensor &x) {
	static vector<Vector> mean;
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		mean[k] = x[k].rowwise().mean();
	}
	return mean;
}

Vector& mean(const Matrix &x) {
	static Vector mean;
	mean = x.rowwise().mean();

	return mean;
}

vector<double>& mean(const vector<Vector> &x) {
	static vector<double> mean;
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
	y.array() -= x - y.array();
	return y;
}

MatrixI& operator -=(MatrixI &x, int y) {
	x.array() -= y;
	return x;
}

Vector& operator -=(Vector &x, double y) {
	x.array() -= y;
	return x;
}

Vector& operator -(Vector &x, double y) {
	return x -= y;
}

vector<VectorI>& operator -=(vector<VectorI> &x, int y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k].array() -= y;
	}
	return x;
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

vector<Vector>& operator *(double x, const vector<VectorI> &y) {
	static vector<Vector> out;
	int batch_size = y.size();
	out.resize(batch_size);
	for (int k = 0; k < batch_size; ++k) {
		out[k] = x * y[k].cast<double>();
	}
	return out;
}

Matrix& operator *(double x, const MatrixI &y) {
	static Matrix out;
	out = x * y.cast<double>();
	return out;
}

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

vector<VectorI>& operator *=(vector<VectorI> &x, int y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] *= y;
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
	static vector<Vector> out;
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
