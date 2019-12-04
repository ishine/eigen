#include <math.h>
#include <matrix.h>
#include "lagacy.h"

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

double logistic(double x) {
	return 1 / (1 + exp(-x));
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

Vector& function(Vector &x, double (*fptr)(double)) {
	int cols = x.cols();

	for (int j = 0; j < cols; ++j) {
		x[j] = fptr(x[j]);
	}
	return x;
}

Matrix& hard_sigmoid(Matrix &x) {
	return function(x, hard_sigmoid);
}

Vector& hard_sigmoid(Vector &x) {
	return function(x, hard_sigmoid);
}

Matrix& logistic(Matrix &x) {
	return function(x, logistic);
}

Vector& logistic(Vector &x) {
	return function(x, logistic);
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

Matrix& softmax(Matrix &x) {
	int rows = x.rows();

	x = exp(x);
	for (int i = 0; i < rows; ++i) {
		x.row(i) /= x.row(i).sum();
	}
	return x;
}

Vector& softmax(Vector &x) {
	x = exp(x);
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

MatrixI& not_equal(MatrixI &x, int y) {
	int rows = x.rows();
	int cols = x.cols();

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			x(i, j) = x(i, j) == y ? 0 : 1;
		}
	}
	return x;
}

MatrixI& equal(MatrixI &x, int y) {
	int rows = x.rows();
	int cols = x.cols();

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			x(i, j) = x(i, j) == y ? 1 : 0;
		}
	}
	return x;
}

MatrixI& operator -=(MatrixI &x, int y) {
	int rows = x.rows();
	int cols = x.cols();

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			x(i, j) -= y;
		}
	}
	return x;

}

MatrixI& operator -(int x, MatrixI &y) {
	int rows = y.rows();
	int cols = y.cols();

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			y(i, j) = x - y(i, j);
		}
	}
	return y;

}

vector<Vector>& mean(const vector<Matrix> &x) {
	static vector<Vector> mean;
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		mean[k] = x[k].rowwise().mean();
	}
	return mean;
}

vector<Matrix>& operator -(vector<Matrix> &x, const vector<Vector> &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] -= y[k];
	}
	return x;
}

vector<Matrix>& square(vector<Matrix> &x) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] = x[k].array().square();
	}
	return x;
}

vector<Vector>& sqrt(vector<Vector> &x) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k] = x[k].cwiseSqrt();
	}
	return x;
}
vector<Vector>& operator +(vector<Vector> &x, double y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		x[k].array() += y;
	}
	return x;

}

vector<Matrix>& operator /(vector<Matrix> &x, const vector<Vector> &y) {
	int batch_size = x.size();
	for (int k = 0; k < batch_size; ++k) {
		int rows = x[k].rows();
		for (int j = 0; j < rows; ++j) {
			x[k].row(j).array() /= y[k].array();
		}

	}
	return x;
}

vector<Matrix>& operator *(vector<Matrix> &x, const Vector &y) {
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

vector<Matrix>& operator +(vector<Matrix> &x, const Vector &y) {
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
