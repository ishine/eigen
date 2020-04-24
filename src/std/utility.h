#pragma once
//gcc -Werror=return-local-addr
#pragma GCC diagnostic error "-Wreturn-local-addr"

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

	bool operator ==(const _Myt &y) const {
		return *this->reptr() == *y.reptr();
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

using std::ifstream;
using std::ofstream;
using std::ios;

#include <unordered_map>
template<typename KEY, typename VALUE>
using dict = std::unordered_map<KEY, VALUE>;

#include <string>
using std::string;

typedef std::u16string String;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <assert.h>

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
	struct iterator {
		iterator(Text *text, bool eof);
		Text *text;
		bool eof = true;

		iterator& operator++();
		bool operator!=(iterator &other);
		String& operator*();
	};

	String line;
	int unicode2jchar(int unicode);

	iterator begin();
	iterator end();

	Text(const string &file);
	ifstream file;

	Text& operator >>(int &unicode);
	Text& operator >>(String &v);
	Text& operator >>(vector<String> &v);
	Text& operator >>(dict<String, int> &word2id);
	Text& operator >>(dict<char16_t, int> &word2id);
	dict<String, int> read_vocab();
	dict<char16_t, int> read_char_vocab();
	String toString();
	operator bool();
	static char str[];
	static int utf2unicode(const char *pText);

	static const char* unicode2utf(word wc, char *pText = 0);
	static string unicode2utf(const String &wstr);
	static string unicode2gbk(const String &wstr);
	static char get_bits(char ch, int start, int size, int shift = 0);
	static char get_bits(char ch, int start, int size, char _ch);
	static char get_bits(char ch, int start, int size, char _ch, int _size);
	static size_t get_utf8_char_len(char byte);
	static void test_utf_unicode_conversion();
};

std::ostream& operator <<(std::ostream &cout, const String &v);

vector<String> split(const String &in);

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

char16_t tolower(char16_t ch);

char16_t toupper(char16_t ch);

template<class T>
std::basic_string<T>& tolower(std::basic_string<T> &s) {
	if (s.empty()) {
		return s;
	}
	for (T &ch : s) {
		ch = tolower(ch);
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
vector<std::basic_string<_Ty>>& operator <<(vector<std::basic_string<_Ty>> &out,
		const _Ty *in) {
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

namespace std {
int strlen(const String &value);

String toString(int);

template<typename _Ty>
vector<_Ty> sample(vector<_Ty> v, int size) {
	int v_size = v.size();
	for (int i = 0; i < size; ++i) {
		int j = i + rand() % (v_size - i);
		std::swap(v[i], v[j]);
	}
	v.resize(size);
	return v;
}
}

#include <queue>
#include <forward_list>
