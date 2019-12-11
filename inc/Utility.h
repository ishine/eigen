#pragma once
#include <iostream>

typedef unsigned long long qword;
typedef unsigned char byte;
typedef unsigned short word;
typedef unsigned int dword;
static_assert(sizeof(void *) == 8, "only 64-bit code generation is supported!");
static_assert(sizeof(byte) == 1, "sizeof(byte) must be 1");
static_assert(sizeof(word) == 2, "sizeof(word) must be 2");
static_assert(sizeof(dword) == 4, "sizeof(dword) must be 4");
static_assert(sizeof(qword) == 8, "sizeof(qword) must be 8");

template<typename _Ty, int b = 2>
struct color_ptr {
	static const qword MASK = (1 << b) - 1;
	typedef color_ptr _Myt;
	typedef _Ty value_type;
	friend std::ostream& operator <<(std::ostream &cout, const _Myt &p) {
		if (!p)
			cout << "nullptr";
		else
			; // cout << *p;
		return cout << ",\t" << std::boolalpha << p.color;
	}

	//postcondition: return a valid pointer
	operator value_type*() {
		return reptr();
	}

	operator const value_type*() const {
		return (const value_type*) reptr();
	}

	color_ptr() {
	}

	explicit color_ptr(value_type *ptr, byte color = 0) {
		this->reptr(ptr);
		this->color = color;
	}

	//_Myt &operator = (value_type *ptr) {
	//	this->ptr = ptr;
	//	return *this;
	//}

	bool operator <(const _Myt &y) const {
		return reptr() < y.reptr();
	}

	bool operator ==(const _Myt &y) const {
		return value == y.value;
	}

	bool operator ==(const value_type *ptr) const {
		return this->reptr() == reptr();
	}

	bool operator ==(value_type *ptr) const {
		return this->reptr() == reptr();
	}

	bool operator !() const {
		return !reptr();
	}

	const value_type* operator ->() const {
		return operator const value_type*();
	}

	value_type* operator ->() {
		return operator value_type*();
	}

	value_type* reptr() const {
		return (value_type*) (value & ~MASK);
	}

	void reptr(value_type *ptr) {
		(value &= MASK) |= (qword) ptr;
		//		return ptr;
	}

	union {
		mutable byte color :b;
		mutable qword value;
	};
};

template<typename _Ty>
struct object: color_ptr<_Ty> {
	typedef object _Myt;
	typedef _Ty element_type;
//	size_t hashCode() const {
//		return ::hashCode(reptr());
//	}

	object(const _Myt &y) :
			color_ptr<_Ty>(y) {
		y.color = true;
	}

	_Myt& operator =(const _Myt &y) {
		_Myt tmp = *this;
		return *::new (this) _Myt(y);
	}

	object(_Ty *ptr = 0) :
			color_ptr<_Ty>(ptr) {
	}

	_Myt& operator =(_Ty *y) {
		this->~object();
		return *::new (this) _Myt(y);
	}

	bool operator <(const _Myt &y) const {
		return *this->reptr() < *y.reptr();
	}

	template<typename T> _Myt& operator +=(T &y) {
		this->reptr(&(*this->reptr() + y));
		return *this;
	}
	template<typename T> _Myt& operator -=(T &y) {
		this->reptr(&(*this->reptr() - y));
		return *this;
	}
	template<typename T> _Myt& operator *=(T &y) {
		this->reptr(&(*this->reptr() * y));
		return *this;
	}
	template<typename T> _Myt& operator /=(T &y) {
		this->reptr(&(*this->reptr() / y));
		return *this;
	}
	template<typename T> _Myt& operator %=(T &y) {
		this->reptr(&(*this->reptr() % y));
		return *this;
	}
	template<typename T> _Myt& operator &=(T &y) {
		this->reptr(&(*this->reptr() & y));
		return *this;
	}
	template<typename T> _Myt& operator |=(T &y) {
		this->reptr(&(*this->reptr() | y));
		return *this;
	}
	template<typename T> _Myt& operator ^=(T &y) {
		this->reptr(&(*this->reptr() ^ y));
		return *this;
	}

	~object() {
		auto ptr = this->reptr();
		if (this->color || !ptr)
			return;
		delete ptr;
	}

	template<typename T>
	T* instanceof() {
		return dynamic_cast<T*>(this->reptr());
	}

	template<typename T>
	const T* instanceof() const {
		return dynamic_cast<const T*>(this->reptr());
	}
};

#include <vector>
using std::vector;


#include <Eigen/Dense>
//${MINGW_HOME}\lib\gcc\x86_64-w64-mingw32\8.1.0\include\c++
typedef Eigen::Matrix<double, 1, -1, 1> Vector;
typedef Eigen::Matrix<double, -1, -1, 1> Matrix;
typedef vector<Matrix> Tensor;

typedef Eigen::Matrix<int, 1, -1, 1> VectorI;
typedef Eigen::Matrix<int, -1, -1, 1> MatrixI;
typedef vector<MatrixI> TensorI;

#include <fstream>
using std::ifstream;
using std::ofstream;
using std::ios;

#include <unordered_map>
using std::unordered_map;

#include <string>
using std::string;
typedef std::basic_string<word> String;

#include <iostream>
using std::cout;
//using std::wcout;
using std::cerr;
using std::endl;

struct BinaryReader {
	ifstream dis;
	string s_FilePath;
	BinaryReader(const string &s_FilePath);

	int read(int &x);
	void* read(void *x, int size);
	word read(word &x);
	float read(float &x);
	double read(double &x);
	long long read(long long &x);
	Vector& read(Vector &arr);
	Matrix& read(Matrix &arr);
	vector<Matrix>& read(vector<Matrix> &arr);

	unordered_map<word, int>& read(unordered_map<word, int> &word2id);

	vector<vector<vector<double>>>& read(vector<vector<vector<double>>>&);

	void close();
};

string& workingDirectory();
string& modelsDirectory();
string& cnModelsDirectory();
string& nerModelsDirectory();
string& serviceBinary();
string nerBinary(const string &service);

void print_primitive_type_size();

#include <assert.h>

vector<double> convert2vector(const Matrix &m, int row_index);
vector<double> convert2vector(const Vector &m);
vector<vector<double>> convert2vector(const Matrix &m);

template<typename _Ty>
std::ostream& operator <<(std::ostream &cout, const vector<_Ty> &v) {
	cout << '[';
	if (!v.empty()) {
		cout << v[0];
		for (size_t i = 1; i < v.size(); ++i) {
			cout << ", " << v[i];
		}
	}

	cout << ']';
	return cout;
}

Vector& min(const Matrix &x, Vector &m, vector<int> &argmin);
Vector& max(const Matrix &x, Vector &m, vector<int> &argmax);

template<typename _Ty>
_Ty gcd(_Ty x, _Ty y) {
	if (!y)
		return x;
	return gcd(y, x % y);
}
