#pragma once
#include "../std/utility.h"
#include <omp.h>
//#define EIGEN_HAS_OPENMP
//#define EIGEN_DONT_PARALLELIZE
//#define EIGEN_USE_MKL_ALL
#include "../Eigen/Dense"

using floatx = double;

typedef Eigen::Matrix<double, 1, -1, 1> Vector;
typedef Eigen::Matrix<double, -1, -1, 1> Matrix;
typedef vector<Matrix> Tensor;

typedef Eigen::Matrix<int, 1, -1, 1> VectorI;
typedef Eigen::Matrix<int, -1, -1, 1> MatrixI;
typedef vector<MatrixI> TensorI;
#include "../hdf5/H5Cpp.h"
//https://portal.hdfgroup.org/display/support/HDF5+1.10.5
struct HDF5Reader {
	H5::H5File hdf5;
	vector<string> layer_names;
	int layer_index;
	H5::Group group;
	int weight_index;
	vector<string> weight_names;

	HDF5Reader& operator >>(std::pair<vector<int>, vector<double>>&);
	HDF5Reader(const string &s_FilePath);

	HDF5Reader& operator >>(Vector &arr);
	HDF5Reader& operator >>(Matrix &arr);
	HDF5Reader& operator >>(Tensor &arr);
};

string& modelsDirectory();
string& nerModelsDirectory();
string& serviceBinary();

vector<double> convert2vector(const Matrix &m, int row_index);
vector<double> convert2vector(const Vector &m);
vector<vector<double>> convert2vector(const Matrix &m);

VectorI string2id(const String &s, const dict<char16_t, int> &dict);
VectorI string2id(const vector<String> &s, const dict<String, int> &dict);
vector<VectorI> string2id(const vector<String> &s, const dict<char16_t, int> &dict);

vector<VectorI> string2ids(const vector<String> &s,
		const dict<char16_t, int> &dict);

//forward declaration to prevent runtime linking error.
extern string workingDirectory;
extern int cpu_count;
