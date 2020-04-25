#include "utility.h"
#include <string>

#include<fstream>

using namespace std;

string workingDirectory = "../";

string& modelsDirectory() {
	static string modelsDirectory = workingDirectory + "models/";
	return modelsDirectory;
}

string& nerModelsDirectory() {
	static string nerModelsDirectory = modelsDirectory() + "cn/ner/";
	return nerModelsDirectory;

}

string& serviceModelsDirectory() {
	static string serviceModelsDirectory = modelsDirectory() + "cn/gru_data/";
	return serviceModelsDirectory;
}

vector<string> openAttribute(const H5::Group &group, const char *name);

HDF5Reader::HDF5Reader(const string &s_FilePath) :
		hdf5(s_FilePath, H5F_ACC_RDONLY), layer_names(
				openAttribute(hdf5, "layer_names")), layer_index(0), group(
				hdf5.openGroup(layer_names[layer_index])), weight_index(-1) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;

//	cout << "weight_index = " << weight_index << endl;

//	this->s_FilePath = s_FilePath;
}

HDF5Reader& HDF5Reader::operator >>(Vector &arr) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
	std::pair<vector<int>, vector<double>> tuple;
	*this >> tuple;
	auto &shape = tuple.first;
	auto &weight = tuple.second;
	assert(shape.size() == 1);

	int dimension = shape[0];
//	cout << "x = " << dimension << endl;

	arr.resize(dimension);
	assert(arr.cols() == dimension);
	for (int i = 0; i < dimension; ++i) {
		arr[i] = weight[i];
	}
	return *this;
}

//https://bitbucket.hdfgroup.org/projects/HDFFV/repos/hdf5/browse
//https://blog.csdn.net/renyhui/article/details/77735314
//http://web.mit.edu/fwtools_v3.1.0/www/H5.intro.html#Intro-TOC
//http://web.mit.edu/fwtools_v3.1.0/www/cpplus_RM/files.html
//void HDF5Reader::read_hdf5() {
//	vector<int> content;
//	int x;
//	dis.seekg(0, std::ios::end);
//	int size = dis.tellg();
//	cout << "size = " << size << endl;
//	dis.seekg(0, std::ios::beg);
//
//	for (int i = 0; i < size / 4; ++i) {
//		this->dis.read((char*) &x, 4);
//		content.push_back(x);
//	}
//
//	int _size = dis.tellg();
//	assert(_size == size);
//	cout << "finish reading " << endl;
//}

//void* HDF5Reader::read(void *x, int size) {
//	char *arr = (char*) x;
//	this->dis.read(arr, size);
//	for (int i = 0, length = size / 2; i < length; ++i) {
//		std::swap(arr[i], arr[size - 1 - i]);
//	}
//	return x;
//}
//
//int HDF5Reader::read(int &x) {
//	this->read(&x, sizeof(int));
//	return x;
//}

//double HDF5Reader::read(double &x) {
//	this->read(&x, sizeof(double));
//	return x;
//}
//
//float HDF5Reader::read(float &x) {
//	this->read(&x, sizeof(float));
//	return x;
//}
//
//word HDF5Reader::read(word &x) {
//	this->read(&x, sizeof(word));
//	return x;
//}
//

HDF5Reader& HDF5Reader::operator >>(Matrix &arr) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
	std::pair<vector<int>, vector<double>> tuple;
	*this >> tuple;
	auto &shape = tuple.first;
	auto &weight = tuple.second;

//	cout << "shape.size() = " << shape.size() << endl;

	assert(shape.size() == 2);

	int dimension0 = shape[0];
	int dimension1 = shape[1];
//	cout << "x = " << dimension0 << ", " << "y = " << dimension1 << endl;

	arr.resize(dimension0, dimension1);

	int index = 0;
	for (int i0 = 0; i0 < dimension0; ++i0) {
		for (int i1 = 0; i1 < dimension1; ++i1) {
			arr(i0, i1) = weight[index++];
		}
	}
	return *this;
}

//unordered_map<word, int>& HDF5Reader::read(unordered_map<word, int> &char2id) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
//	int length;
//	read(length);
//
//	cout << "length = " << length << endl;
////	print_primitive_type_size();
//
//	assert(length >= 0);
//
//	for (int i = 0; i < length; ++i) {
//		word key;
//		read(key);
//		read(char2id[key]);
//
////		cout << "char2id[" << key << "] = " << char2id[key] << endl;
//	}
//	return char2id;
//}

HDF5Reader& HDF5Reader::operator >>(
		std::pair<vector<int>, vector<double>> &tuple) {
	std::pair<vector<int>, vector<double>>& read_keras_model(
			const H5::H5File &file, const H5::Group &group,
			const string &weight_name,
			std::pair<vector<int>, vector<double>> &tuple);

//	cout << "layer_names = " << layer_names << endl;
//	cout << "weight_index = " << weight_index << endl;

	while (true) {
//		cout << "weight_names = " << weight_names << endl;
		if (++weight_index < (int) weight_names.size()) {
			break;
		}

		++layer_index;
		assert(layer_index < (int ) layer_names.size());

//		cout << "layer_names[" << layer_index << "] = "
//				<< layer_names[layer_index] << endl;

		group = hdf5.openGroup(layer_names[layer_index]);

		this->weight_names = openAttribute(group, "weight_names");
		weight_index = -1;
	}
	read_keras_model(hdf5, group, weight_names[weight_index], tuple);
	return *this;
}

HDF5Reader& HDF5Reader::operator >>(Tensor &arr) {
	std::pair<vector<int>, vector<double>> tuple;
	*this >> tuple;
	auto &shape = tuple.first;
	auto &weight = tuple.second;
	assert(shape.size() == 3);

	int dimension0 = shape[0];
	int dimension1 = shape[1];
	int dimension2 = shape[2];

//	printf("d0 = %d, d1 = %d, d2 = %d\n", dimension0, dimension1, dimension2);
	arr.resize(dimension0);

	int index = 0;
	for (int i0 = 0; i0 < dimension0; ++i0) {
		arr[i0].resize(dimension1, dimension2);
		for (int i1 = 0; i1 < dimension1; ++i1) {
			for (int i2 = 0; i2 < dimension2; ++i2) {
				arr[i0](i1, i2) = weight[index++];
			}
		}
	}

	return *this;
}

vector<double> convert2vector(const Matrix &m, int row_index) {
	auto start = m.data() + row_index * m.cols();

	vector<double> v(start, start + m.cols());
	return v;
}

vector<vector<double>> convert2vector(const Matrix &m) {
	vector<vector<double>> v(m.rows(), vector<double>(m.cols(), 0.0));
	for (int i = 0; i < m.rows(); ++i) {
		for (int j = 0; j < m.cols(); ++j) {
			v[i][j] = m(i, j);
		}
	}
	return v;
}

vector<double> convert2vector(const Vector &m) {
	auto start = m.data();

	vector<double> v(start, start + m.cols());
	return v;
}

VectorI string2id(const String &s, const ::dict<char16_t, int> &dict) {
	VectorI v;
	v.resize(s.size());

	for (size_t i = 0; i < s.size(); ++i) {
		auto iter = dict.find(s[i]);
		v(i) = iter == dict.end() ? 1 : iter->second;
	}
	return v;
}

vector<VectorI> string2id(const vector<String> &s,
		const ::dict<char16_t, int> &dict) {
	vector<VectorI> v(s.size());

	for (size_t i = 0; i < s.size(); ++i) {
		v[i] = string2id(s[i], dict);
	}
	return v;
}

VectorI string2id(const vector<String> &s, const ::dict<String, int> &dict) {
	VectorI v;
	v.resize(s.size());

	for (size_t i = 0; i < s.size(); ++i) {
		auto iter = dict.find(s[i]);
		v(i) = iter == dict.end() ? 1 : iter->second;
	}
	return v;
}

vector<VectorI> string2ids(const vector<String> &s,
		const unordered_map<char16_t, int> &dict) {
	vector<VectorI> v;
	int batch_size = s.size();
	v.reserve(batch_size);

	for (int k = 0; k < batch_size; ++k) {
		v[k] = string2id(s[k], dict);
	}
	return v;
}

/*
 int test_matmul() {
 const int m = 80;
 const int k = 4;
 const int n = 4;
 int Matrix1[m][k] = { };
 int Matrix2[k][n] = { };
 int Matrix[m][n] = { };
 clock_t start, end;
 cout << "Matrix1:\n";
 int i, j;
 for (i = 0; i < m; i++) {
 for (j = 0; j < k; j++) {
 Matrix1[i][j] = i + j;
 cout << Matrix1[i][j] << '\t';
 }
 cout << endl;
 }
 cout << "Matrix2:\n";
 for (i = 0; i < k; i++) {
 for (j = 0; j < n; j++) {
 Matrix2[i][j] = 2 * i - j;
 cout << Matrix2[i][j] << '\t';
 }
 cout << endl;
 }

 //	omp_set_num_threads(3);
 int pnum = omp_get_num_procs();
 cout << "Thread_pnum =" << pnum << endl;
 int l;
 start = clock();
 //开始计时
 #pragma omp parallel shared(Matrix1, Matrix2, Matrix) private(j, l)
 {
 #pragma omp for schedule(dynamic)
 for (i = 0; i < m; i++) {
 cout << "Thread_num:" << omp_get_thread_num() << '\n';
 for (j = 0; j < n; j++) {
 for (l = 0; l < k; l++) {
 Matrix[i][j] += Matrix1[i][l] * Matrix2[l][j];
 }
 }
 }
 }
 end = clock();
 cout << "Matrix multiply time:" << (end - start) << endl;

 //	cout << "The result is:\n";
 //	for (i = 0; i < m; i++) {
 //		for (j = 0; j < n; j++) {
 //			cout << Matrix[i][j] << '\t';
 //		}
 //		cout << endl;
 //	}
 return 0;
 }
 int test() {

 int i;

 printf("*Hello World! Thread: %d\n", omp_get_thread_num());

 #pragma omp parallel for
 for (i = 0; i < 32; i += 3) {
 if (i % 2)
 printf("Hello World!  Thread: %d, odd %d, \n", omp_get_thread_num(),
 i);
 else
 printf("Hello World!  Thread: %d, even %d, \n",
 omp_get_thread_num(), i);
 }

 return 0;

 }

 */
/**
 ctrl + tab  switch between .h and .cpp
 shift + alt + t
 shift + alt + m
 ctrl  + alt + s
 ctrl + alt + h
 ctrl + o
 Ctrl + Shift + G
 Ctrl + Shift + Minus
 Ctrl + Shift + Plus
 */

struct Object {
	Object() {
		x = y = z = 0;
		kinder = new Object(2, 2, 2);
		cout << "in " << __PRETTY_FUNCTION__ << endl;
	}

	Object(int x, int y, int z) :
			x(x), y(y), z(z) {
		cout << "in " << __PRETTY_FUNCTION__ << endl;
	}

	~Object() {
		cout << "in " << __PRETTY_FUNCTION__ << endl;
	}

	void reset() {
		x = y = z = 1;
		kinder = nullptr;
	}

	int x, y, z;
	object<Object> kinder;
	friend std::ostream& operator <<(std::ostream &cout, const Object &p) {
		cout << "x = " << p.x << ",\t";
		cout << "y = " << p.y << ",\t";
		cout << "z = " << p.z << endl;
		return cout;
	}
};

Object return_tmp() {
	Object obj;
	obj.reset();
	return obj;
}

#include <omp.h>
#include <iostream>
int cpu_count = []() -> int {
//	http://eigen.tuxfamily.org/dox/TopicMultiThreading.html
		int cpu_count = omp_get_max_threads();
		Eigen::setNbThreads(cpu_count);
		Eigen::initParallel();
		cout << "Eigen::initParallel() is called!" << endl;
		cout << "cpu_count = " << cpu_count << endl;
		return cpu_count;
	}();

#include <chrono>
//gcc -mavx -mfma
void test_speed() {
	const int dim = 100;
	std::chrono::time_point<std::chrono::system_clock> start, end;

	int n;
	n = Eigen::nbThreads();
	cout << n << "\n";

	Matrix m1(dim, dim);
	Matrix m2(dim, dim);
	Matrix m_res(dim, dim);
	m1.setRandom(dim, dim);
	m2.setRandom(dim, dim);

	start = std::chrono::system_clock::now();

	for (int i = 0; i < 100000; ++i) {
		m_res = m1 * m2;
	}

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;

	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

//http://eigen.tuxfamily.org/dox/
//https://blog.csdn.net/zong596568821xp/article/details/81134406
void test_eigen() {
	Matrix A = Matrix::Random(3000, 3000);  // 随机初始化矩阵
	Matrix B = Matrix::Random(3000, 3000);

	double start = clock();
	Matrix C = A * B;    // 乘法好简洁
	double endd = clock();
	double thisTime = (double) (endd - start) / CLOCKS_PER_SEC;

	cout << "time cost for 3000 * 3000 matrix multiplication: = " << thisTime
			<< endl;
}
