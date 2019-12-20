#pragma once
#include <iostream>
#include <fstream>
#include <vector>
using std::vector;
#include <iterator>
#include <regex>

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

#include <Eigen/Dense>

using floatx = double;

typedef Eigen::Matrix<double, 1, -1, 1> Vector;
typedef Eigen::Matrix<double, -1, -1, 1> Matrix;
typedef vector<Matrix> Tensor;

typedef Eigen::Matrix<int, 1, -1, 1> VectorI;
typedef Eigen::Matrix<int, -1, -1, 1> MatrixI;
typedef vector<MatrixI> TensorI;

using std::ifstream;
using std::ofstream;
using std::ios;

#include <unordered_map>
using std::unordered_map;

#include <string>
using std::string;

typedef std::u16string String;
//namespace std {
////this compiler setting doesnot solve the problem: g++ -fshort-wchar
///// std::hash specialization for wstring.
//template<>
//struct hash<String> : public __hash_base<size_t, String> {
//	size_t operator()(const String& __s) const noexcept
//	{
//		return std::_Hash_impl::hash(__s.data(), __s.length() * sizeof(word));
//	}
//};
//
//template<>
//struct __is_fast_hash<hash<String>> : std::false_type {
//};
//}

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

	BinaryReader &operator >> (int &x);
	BinaryReader &operator >> (Vector &arr);
	BinaryReader &operator >> (Matrix &arr);
	void read_hdf5();
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

template<typename _Ty>
_Ty gcd(_Ty x, _Ty y) {
	if (!y)
		return x;
	return gcd(y, x % y);
}

struct Text {
	Text(const string &file);
	ifstream file;

	Text& operator >>(word &v);
//	Text& operator >>(string &v);
	Text& operator >>(String &v);
	Text& operator >>(unordered_map<String, int> &word2id);
	operator bool();
	static char str[];
	static word utf2unicode(const char *pText);

	static const char* unicode2utf(word wc, char *pText = 0);
	static string& unicode2utf(const String &wstr);
	static string& unicode2gbk(const String &wstr);
	static char get_bits(char ch, int start, int size, int shift = 0);
	static size_t get_utf8_char_len(char byte);
	static void test_utf_unicode_conversion();
};

VectorI& string2id(const String &s, const unordered_map<String, int> &dict);
vector<VectorI>& string2id(const vector<String> &s,
		const unordered_map<String, int> &dict);

std::ostream& operator <<(std::ostream &cout, const String &v);

vector<String> &split(const String &in);

#include "wchar.h"

template<class T>
std::basic_string<T>& strip(std::basic_string<T> &s) {
	if (s.empty()) {
		return s;
	}

	typename std::basic_string<T>::iterator c;

	// Erase whitespace before the string

	for (c = s.begin(); c != s.end() && iswspace(*c++);)
		;
	s.erase(s.begin(), --c);

	// Erase whitespace after the string

	for (c = s.end(); c != s.begin() && iswspace(*--c);)
		;
	s.erase(++c, s.end());
	return s;
}

template<class T>
std::basic_string<T>& lstrip(std::basic_string<T> &s) {
	if (s.empty()) {
		return s;
	}

	typename std::basic_string<T>::iterator c;

	// Erase whitespace before the string

	for (c = s.begin(); c != s.end() && iswspace(*c++);)
		;
	s.erase(s.begin(), --c);

	return s;
}

template<class T>
std::basic_string<T>& tolower(std::basic_string<T> &s) {
	if (s.empty()) {
		return s;
	}
	for (T &ch : s) {
		ch = std::towlower(ch);
	}
	return s;
}

template<class _Ty>
vector<_Ty>& operator <<(vector<_Ty> &out, const vector<_Ty> &in) {
	out.insert(out.end(), in.begin(), in.end());
	return out;
}

template<class _Ty>
vector<_Ty>& operator <<(vector<_Ty> &out, const _Ty &in) {
	out.insert(out.end(), in);
	return out;
}

template<class _Ty>
vector<std::basic_string<_Ty>>& operator <<(vector<std::basic_string<_Ty>> &out, const _Ty *in) {
	return out << std::basic_string<_Ty>(in);
}

template<class _Ty>
bool operator !(const vector<_Ty> &x) {
	return x.empty();
}

template<class T>
bool operator !(const std::basic_string<T> &s) {
	return s.empty();
}
