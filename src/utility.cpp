#include <utility.h>
string& workingDirectory() {
#ifdef _WIN32
	static string workingDirectory = "D:/360/solution/";
#else
	static string workingDirectory = "/home/v-rogenhxh/deeplearning/";
#endif
	return workingDirectory;
}

//string get_workingDirectory() {
//
//	int index = workingDirectory.find_last_of("/\\");
//
//	workingDirectory = workingDirectory.substr(0, index);
//
//	workingDirectory += "/../";
//
//	cout << "workingDirectory = " << workingDirectory << endl;
//	return workingDirectory;
//}

string& modelsDirectory() {
	static string modelsDirectory = workingDirectory() + "models/";
	return modelsDirectory;
}

string& cnModelsDirectory() {
	static string cnModelsDirectory = modelsDirectory() + "cn/";
	return cnModelsDirectory;
}

string& nerModelsDirectory() {
	static string nerModelsDirectory = cnModelsDirectory() + "ner/";
	return nerModelsDirectory;

}

string& serviceModelsDirectory() {
	static string serviceModelsDirectory = cnModelsDirectory() + "gru_data/";
	return serviceModelsDirectory;
}

string& serviceBinary() {
	static string serviceBinary = serviceModelsDirectory() + "service.bin";
	return serviceBinary;
}

string nerBinary(const string &service) {
	return nerModelsDirectory() + service + ".bin";
}

BinaryReader::BinaryReader(const string &s_FilePath) :
		dis(s_FilePath.c_str(), ios::in | std::ios_base::binary) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	this->s_FilePath = s_FilePath;
}

Vector& BinaryReader::read(Vector &arr) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	int dimension;
	read(dimension);
	cout << "x = " << dimension << endl;
	arr.resize(dimension);
	assert(arr.cols() == dimension);
	for (int i = 0; i < dimension; ++i) {
		read(arr[i]);
	}
	return arr;
}

void* BinaryReader::read(void *x, int size) {
	char *arr = (char*) x;
	this->dis.read(arr, size);
	for (int i = 0, length = size / 2; i < length; ++i) {
		std::swap(arr[i], arr[size - 1 - i]);
	}
	return x;
}

int BinaryReader::read(int &x) {
	this->read(&x, sizeof(int));
	return x;
}

double BinaryReader::read(double &x) {
	this->read(&x, sizeof(double));
	return x;
}

float BinaryReader::read(float &x) {
	this->read(&x, sizeof(float));
	return x;
}

word BinaryReader::read(word &x) {
	this->read(&x, sizeof(word));
	return x;
}

Matrix& BinaryReader::read(Matrix &arr) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	int dimension0;
	read(dimension0);
	int dimension1;
	read(dimension1);
	cout << "x = " << dimension0 << ", " << "y = " << dimension1 << endl;
	arr.resize(dimension0, dimension1);
	for (int i0 = 0; i0 < dimension0; ++i0) {
		for (int i1 = 0; i1 < dimension1; ++i1) {
			read(arr(i0, i1));
		}
	}
	return arr;
}

unordered_map<word, int>& BinaryReader::read(
		unordered_map<word, int> &char2id) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	int length;
	read(length);

	cout << "length = " << length << endl;
//	print_primitive_type_size();

	assert(length >= 0);

	for (int i = 0; i < length; ++i) {
		word key;
		read(key);
		read(char2id[key]);

//		cout << "char2id[" << key << "] = " << char2id[key] << endl;
	}
	return char2id;
}

Tensor& BinaryReader::read(Tensor &arr) {
	int dimension0;
	read(dimension0);
	int dimension1;
	read(dimension1);
	int dimension2;
	read(dimension2);
	printf("d0 = %d, d1 = %d, d2 = %d\n", dimension0, dimension1, dimension2);
	arr.resize(dimension0);

	for (int i0 = 0; i0 < dimension0; ++i0) {
		arr[i0].resize(dimension1, dimension2);
		for (int i1 = 0; i1 < dimension1; ++i1) {
			for (int i2 = 0; i2 < dimension2; ++i2) {
				read(arr[i0](i1, i2));
			}
		}
	}
//			System.out.println(Utility.toString(arr[0][0]));
	return arr;
}

void print_primitive_type_size() {
	cout << "sizeof(char) = " << sizeof(char) << endl;
	cout << "sizeof(wchar_t) = " << sizeof(wchar_t) << endl;
	cout << "sizeof(short) = " << sizeof(short) << endl;
	cout << "sizeof(int) = " << sizeof(int) << endl;
	cout << "sizeof(long) = " << sizeof(long) << endl;
	cout << "sizeof(long long) = " << sizeof(long long) << endl;

	cout << "sizeof(unsigned char) = " << sizeof(unsigned char) << endl;
	cout << "sizeof(unsigned wchar_t) = " << sizeof(unsigned wchar_t) << endl;
	cout << "sizeof(unsigned short) = " << sizeof(unsigned short) << endl;
	cout << "sizeof(unsigned int) = " << sizeof(unsigned int) << endl;
	cout << "sizeof(unsigned long) = " << sizeof(unsigned long) << endl;
	cout << "sizeof(unsigned long long) = " << sizeof(unsigned long long)
			<< endl;

	cout << "sizeof(float) = " << sizeof(float) << endl;
	cout << "sizeof(double) = " << sizeof(double) << endl;
	cout << "sizeof(byte) = " << sizeof(byte) << endl;
	cout << "sizeof(word) = " << sizeof(word) << endl;
	cout << "sizeof(dword) = " << sizeof(dword) << endl;
	cout << "sizeof(qword) = " << sizeof(qword) << endl;
	cout << "sizeof(void*) = " << sizeof(void*) << endl;
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

void BinaryReader::close() {
	int size = dis.tellg();
	dis.seekg(0, std::ios::end);
	int _size = dis.tellg();
	assert(_size == size);
	;

	dis.close();
}

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

//std::ostream& operator <<(std::ostream &cout, const String &v) {
//
//	MultiByteToWideChar()
//	if (!v.empty()) {
//
//		for (size_t i = 0; i < v.size(); ++i) {
//			sizeof word
//			wctomb()
//			cout << ", " << v[i];
//		}
//	}
//
//	cout << ']';
//	return cout;
//}

//Matrix &operator +(Matrix &x, const Vector &m) {
//	x.rowwise() += m;
//	return x;
//}
//
//Matrix &operator +(const Vector &m, Matrix &x) {
//	x.rowwise() += m;
//	return x;
//}
