#pragma once
#include "../../../std/src/std/utility.h"
#include <omp.h>
//#define EIGEN_HAS_OPENMP
//#define EIGEN_DONT_PARALLELIZE
//#define EIGEN_USE_MKL_ALL
#include "../Eigen/Dense"

using floatx = double;

using Vector = Eigen::Matrix<double, 1, -1, 1>;
using Matrix = Eigen::Matrix<double, -1, -1, 1>;
using Tensor = vector<Matrix>;

using VectorI = vector<int>;
using MatrixI = vector<VectorI>;
using TensorI = vector<MatrixI>;
#include "../../../hdf5/src/hdf5/H5Cpp.h"
//https://portal.hdfgroup.org/display/support/HDF5+1.10.5
struct KerasReader {
	H5::H5File hdf5;
	vector<string> layer_names;
	int layer_index;
	H5::Group group;
	int weight_index;
	vector<string> weight_names;

	KerasReader& operator >>(std::pair<vector<int>, vector<double>>&);
	KerasReader(const string &s_FilePath);

	Vector read_vector();
	Matrix read_matrix();
	Tensor read_tensor();

	KerasReader& operator >>(Vector &arr);
	KerasReader& operator >>(Matrix &arr);
	KerasReader& operator >>(Tensor &arr);
};

struct TorchModule {
	TorchModule * const parent;
//	H5::Group group;
	vector<string> Parameter, Module;
	vector<std::pair<vector<int>, vector<double>>> tuple;
	vector<TorchModule*> children;
	size_t indexParameter, indexModule;

	string path;
	~TorchModule();

	TorchModule(const string &s_FilePath);
	TorchModule(const H5::Group &group, TorchModule *parent,
			const string &path);
};

struct TorchReader {
	TorchModule *self;

	TorchReader(const string &s_FilePath);
	TorchReader(TorchModule *self);

	double read_double();
	Vector read_vector();
	Matrix read_matrix();
	Tensor read_tensor();

	TorchReader& operator >>(std::pair<vector<int>, vector<double>>&);

	TorchReader& operator >>(double &a);

	TorchReader& operator >>(Vector &arr);
	TorchReader& operator >>(Matrix &arr);
	TorchReader& operator >>(Tensor &arr);
	~TorchReader();
};

//string& modelsDirectory();
string& nerModelsDirectory();
string& serviceBinary();

vector<double> convert2vector(const Matrix &m, int row_index);
vector<double> convert2vector(const Vector &m);
vector<vector<double>> convert2vector(const Matrix &m);

VectorI string2id(const String &s, const dict<char16_t, int> &dict);
VectorI string2id(const vector<String> &s, const dict<String, int> &dict);

VectorI string2id(const vector<string> &s, const dict<string, int> &dict);
vector<VectorI> string2id(const vector<String> &s,
		const dict<char16_t, int> &dict);

vector<VectorI> string2ids(const vector<String> &s,
		const dict<char16_t, int> &dict);

//forward declaration to prevent runtime linking error.
extern string workingDirectory;
string& modelsDirectory();

#ifdef _DEBUG
#define print_shape(matrix) {std::cout << #matrix << ".shape = (" << matrix.rows() << ", " << matrix.cols() << ")" << std::endl;}
#define print_tensor(matrix) {std::cout << #matrix << ".shape = (" << matrix.size() << ", " << matrix[0].rows() << ", " << matrix[0].cols() << ")" << std::endl;}
#else
#define print_shape(matrix)
#define print_tensor(matrix)
#endif



