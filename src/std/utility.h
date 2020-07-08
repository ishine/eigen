#pragma once
//gcc -Werror=return-local-addr
#pragma GCC diagnostic error "-Wreturn-local-addr"
#pragma GCC diagnostic error "-Wreturn-type"

#include <iostream>
#include <fstream>
#include <vector>
using std::vector;
#include <iterator>
#include <regex>

using qword = unsigned long long;
using byte = unsigned char;
using word = unsigned short;
using dword = unsigned int;
static_assert(sizeof(void *) == 8, "only 64-bit code generation is supported!");
static_assert(sizeof(byte) == 1, "sizeof(byte) must be 1");
static_assert(sizeof(word) == 2, "sizeof(word) must be 2");
static_assert(sizeof(dword) == 4, "sizeof(dword) must be 4");
static_assert(sizeof(qword) == 8, "sizeof(qword) must be 8");

template<typename _Ty, int b = 2>
struct color_ptr {
	static const qword MASK = (1 << b) - 1;
	using _Myt = color_ptr;
	using value_type = _Ty;
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
		value = 0;
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
	using _Myt = object;
	using element_type = _Ty;
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

using String = std::u16string;

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

#include <set>
template<typename _Ty>
std::ostream& operator <<(std::ostream &cout, const std::set<_Ty> &v) {
	cout << '{';
	bool initial = true;
	for (const auto &e : v) {
		if (initial) {
			cout << e;
			initial = false;
		} else
			cout << ", " << e;
	}

	cout << '}';
	return cout;
}

#include <unordered_set>
template<typename _Ty>
std::ostream& operator <<(std::ostream &cout,
		const std::unordered_set<_Ty> &v) {
	cout << '{';
	bool initial = true;
	for (const auto &e : v) {
		if (initial) {
			cout << e;
			initial = false;
		} else
			cout << ", " << e;
	}

	cout << '}';
	return cout;
}

template<typename _Key, typename _Ty>
std::ostream& operator <<(std::ostream &cout, const dict<_Key, _Ty> &map) {
	cout << '{';
	bool initial = true;
	for (const auto &p : map) {
		if (initial) {
			initial = false;
		} else
			cout << ", ";
		cout << p.first << " : " << p.second;
	}

	cout << '}';
	return cout;
}

template<typename _Key, typename _Ty>
std::ostream& operator <<(std::ostream &cout, const std::map<_Key, _Ty> &map) {
	cout << '{';
	bool initial = true;
	for (const auto &p : map) {
		if (initial) {
			initial = false;
		} else
			cout << ", ";
		cout << p.first << " : " << p.second;
	}

	cout << '}';
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

	vector<String> readlines();
	Text& operator >>(int &unicode);
	Text& operator >>(String &v);
	Text& operator >>(string &v);

	Text& operator >>(vector<String> &v);
	Text& operator >>(dict<String, int> &word2id);
	Text& operator >>(dict<char16_t, int> &word2id);
	dict<String, int> read_vocab(int index = 2);
	dict<string, int> read_vocab_cstr();

	dict<String, int>& read_vocab(dict<String, int> &word2id, int index = 2);
	dict<string, int>& read_vocab(dict<string, int> &word2id, int index = 2);

	dict<char16_t, int> read_vocab_char();
	String toString();
	operator bool();

	static int utf2unicode(const char *pText);

	static const char* unicode2utf(word wc, char *pText);
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

template<class _CharT>
bool startsWith(const std::basic_string<_CharT> &str,
		const std::basic_string<_CharT> &start) {
	auto startlen = start.size();
	return str.size() >= startlen && str.substr(0, startlen) == start;
}

#include "wchar.h"

template<class _CharT>
std::basic_string<_CharT>& strip(std::basic_string<_CharT> &s) {
	if (s.empty()) {
		return s;
	}

	typename std::basic_string<_CharT>::iterator c;

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

template<class _CharT>
std::basic_string<_CharT>& lstrip(std::basic_string<_CharT> &s) {
	if (s.empty()) {
		return s;
	}

	typename std::basic_string<_CharT>::iterator c;

	// Erase whitespace before the string

	for (c = s.begin(); c != s.end() && iswspace(*c++);)
		;
	s.erase(s.begin(), --c);

	return s;
}

char16_t tolower(char16_t ch);

char16_t toupper(char16_t ch);

String& tolower(String &s);

string& tolower(string &s);

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

template<class _CharT>
bool operator !(const std::basic_string<_CharT> &s) {
	return s.empty();
}

namespace std {
int strlen(const String &value);

String toString(int);
String toString(const string &s);

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

template<typename _Ty>
std::set<_Ty> as_set(std::initializer_list<_Ty> list) {
	std::set<_Ty> s;
	for (const auto &x : list) {
		s.insert(x);
	}
	return s;
}

template<typename _Ty>
std::vector<_Ty> list(const std::set<_Ty> &c) {
	std::vector<_Ty> v;
	for (const auto &x : c) {
		v.push_back(x);
	}
	return v;
}

template<typename T>
int indexOf(const vector<T> &elementData, const T &o, int index = 0) {
	for (int i = index, size = elementData.size(); i < size; ++i)
		if (o == elementData[i])
			return i;
	return -1;
}

template<typename T>
vector<T> copierSauf(const vector<T> &arr, size_t i) {
	vector<T> arrNew(arr.size() - 1);
	int index = 0;
	for (size_t j = 0; j < arr.size(); ++j) {
		if (i != j)
			arrNew[index++] = arr[j];
	}
	return arrNew;
}

template<typename T>
bool contains(const vector<T> &elementData, const T &o) {
	return indexOf(elementData, o) >= 0;
}

const double oo = std::numeric_limits<double>::infinity();

#define __log(symbol) {std::cout << #symbol << " = \n" << (symbol) << std::endl;}

#ifdef _DEBUG
#define __cout(symbol) __log(symbol)
#define assert_gt(x, y) if (x > y){}else{std::cout << #x << " > " << #y << std::endl; __cout(x);__cout(y);}
#define assert_lt(x, y) if (x < y){}else{std::cout << #x << " < " << #y << std::endl; __cout(x);__cout(y);}
#define assert_le(x, y) if (x <= y){}else{std::cout << #x << " <= " << #y << std::endl; __cout(x);__cout(y);}
#define assert_ge(x, y) if (x >= y){}else{std::cout << #x << " >= " << #y << std::endl; __cout(x);__cout(y);}
#define assert_eq(x, y) if (x == y){}else{std::cout << #x << " == " << #y << std::endl; __cout(x);__cout(y);}
#define assert_neq(x, y) if (x != y){}else{std::cout << #x << " != " << #y << std::endl; __cout(x);__cout(y);}
#define assert_true(expr) if (expr){}else{std::cout << #expr << " is true " << std::endl; __cout(expr);}
#define assert_false(expr) if (!expr){}else{std::cout << #expr << " is false " << std::endl; __cout(expr);}
#define assert_or(x, y) if (x || y){}else{std::cout << #x << " || " << #y << std::endl; __cout(x);__cout(y);}
#define assert_and(x, y) if (x && y){}else{std::cout << #x << " && " << #y << std::endl; __cout(x);__cout(y);}
#else
#define __cout(symbol)
#define assert_gt(x, y)
#define assert_lt(x, y)
#define assert_le(x, y)
#define assert_ge(x, y)
#define assert_eq(x, y)
#define assert_neq(x, y)
#define assert_true(expr)
#define assert_false(expr)
#define assert_or(x, y)
#define assert_and(x, y)
#endif

struct Timer {
	Timer();
	clock_t start;
	void report(const char *message);
};

// implements a maximum priority queue;
template<typename _Ty, typename _Pr = std::less<_Ty>>
struct priority_queue: std::vector<_Ty> {
	priority_queue() {
	}
	priority_queue(std::vector<_Ty> &c) :
			std::vector<_Ty>(c) {
		// make non-trivial [_First, _Last) into a heap, using _Pred
		int _Top = this->size() >> 1;
		while (0 < _Top--) {
			int _Hole = _Top;
			adjust_heap(_Hole, _Pred);
		}
	}

	void insert(const _Ty &_Val) {	// push operation;
		int _Hole = this->size();
		this->push_back(_Val);
		push_heap(_Hole, 0, _Val, _Pred);
	}

	//precondition : size > 0 && _Where < size;
	_Ty erase(int _Where = 0) {
		int size = this->size();
		assert(_Where < size);
//		if (_Where >= size()) {
//			throw std::exception("_Where < size in " __FUNCTION__, __LINE__);
//		}
		const _Ty &_Val = (*this)[_Where];
		const _Ty &end = this->back();
		--size;
		this->pop_back();
		if (_Where != size) {
			(*this)[_Where] = end;
			adjust_heap(_Where, _Pred);
		}

		return _Val;
	}

	_Pr _Pred;	// the comparator functor
protected:

	void mov(int &_Hole, int _Idx) {
		(*this)[_Hole] = (*this)[_Idx];
		_Hole = _Idx;
	}

	// precondition: ptr[_Hole] is the element to be adjusted;
	void adjust_heap(int &_Hole, _Pr &_Pred) { // percolate _Hole to _Bottom, then push _Val, using _Pred
		_Ty _Val = (*this)[_Hole];
		int _Top = _Hole;
		int _Idx = _Hole;
		int size = this->size();
		while ((++_Idx <<= 1) < size) { // move _Hole down to larger kinder
			if (_Pred((*this)[_Idx], (*this)[_Idx - 1]))
				--_Idx;
			mov(_Hole, _Idx);
		}
		if (_Idx == size) // only kinder at bottom, move _Hole down to it
			mov(_Hole, --_Idx); // a possible bug here, _Top and _Hole might still be the same, thus overwriting ptr[_Top] which must be used as an argument in the push_heap function call;
		push_heap(_Hole, _Top, _Val, _Pred);
	}

	void push_heap(int &_Hole, int _Top, const _Ty &_Val, _Pr &_Pred) { // percolate _Hole to _Top or where _Val belongs
		auto _Idx = _Hole;
		while (_Top < _Idx && _Pred((*this)[--_Idx >>= 1], _Val)) // move _Hole up to parent
			mov(_Hole, _Idx);
		(*this)[_Hole] = _Val;    // drop _Val into final hole
	}
};

//template <typename _Ty, typename _Pr>
//_Pr priority_queue<_Ty, _Pr>::_Pred;

// implements a maximum priority queue whose elements are unique keys, ie, no duplicate items;
template<typename _Ty, typename _Pr = std::less<_Ty>>
struct priority_dict: std::vector<_Ty> {
	priority_dict() {
	}
	priority_dict(const _Pr &_Pred) :
			_Pred(_Pred) {
	}

	priority_dict(const std::set<_Ty> &c) :
			std::vector<_Ty>(c.begin(), c.end()) {
		// make non-trivial [_First, _Last) into a heap, using _Pred
		int size = this->size();
		for (int i = 0; i < size; ++i) {
			this->map[(*this)[i]] = i;
		}

		int _Top = size >> 1;
		while (0 < _Top--) {
			int _Hole = _Top;
			adjust_heap(_Hole, _Pred);
		}
	}

	void insert(const _Ty &_Val) {    // push operation;
		if (map.count(_Val))
			return;
		int _Hole = this->size();
		this->push_back(_Val);
		map[_Val] = _Hole;
		push_heap(_Hole, 0, _Val, _Pred);
//		validity_check();
	}

	void erase(const _Ty &x) {
		if (!map.count(x))
			return;
		int index = map[x];
		map.erase(x);
		erase_indexed(index);
//		validity_check();
	}

	_Ty pop() {
		int index = 0;
		_Ty x = (*this)[index];
		map.erase(x);
		erase_indexed(index);
//		validity_check();
		return x;
	}

	//precondition : size > 0 && _Where < size;
	void erase_indexed(int _Where = 0) {
		int size = this->size();
		assert(_Where < size);

//		_Ty _Val = (*this)[_Where];
		const _Ty &end = this->back();
		--size;

		if (_Where != size) {
			(*this)[_Where] = end;
			map[end] = _Where;
			this->pop_back();

			adjust_heap(_Where, _Pred);
		} else
			this->pop_back();
//		return _Val;
	}

	dict<_Ty, int> map;
	_Pr _Pred;	// the comparator functor
protected:
	void validity_check() {
		assert(map.size() == this->size());
		for (const auto &p : map) {
			if ((*this)[p.second] != p.first) {
				throw std::exception("(*this)[p.second] != p.first");
			}
		}
	}

	void mov(int &_Hole, int _Idx) {
		(*this)[_Hole] = (*this)[_Idx];
		map[(*this)[_Idx]] = _Hole;
		_Hole = _Idx;
	}

	// precondition: ptr[_Hole] is the element to be adjusted;
	void adjust_heap(int &_Hole, _Pr &_Pred) { // percolate _Hole to _Bottom, then push _Val, using _Pred
		_Ty _Val = (*this)[_Hole];
		int _Top = _Hole;
		int _Idx = _Hole;
		int size = this->size();
		while ((++_Idx <<= 1) < size) { // move _Hole down to larger kinder
			if (_Pred((*this)[_Idx], (*this)[_Idx - 1]))
				--_Idx;
			mov(_Hole, _Idx);
		}
		if (_Idx == size) // only kinder at bottom, move _Hole down to it
			mov(_Hole, --_Idx); // a possible bug here, _Top and _Hole might still be the same, thus overwriting ptr[_Top] which must be used as an argument in the push_heap function call;
		push_heap(_Hole, _Top, _Val, _Pred);
	}

	void push_heap(int &_Hole, int _Top, const _Ty &_Val, _Pr &_Pred) { // percolate _Hole to _Top or where _Val belongs
		auto _Idx = _Hole;
		while (_Top < _Idx && _Pred((*this)[--_Idx >>= 1], _Val)) // move _Hole up to parent
			mov(_Hole, _Idx);
		(*this)[_Hole] = _Val;    // drop _Val into final hole
		map[_Val] = _Hole;
	}
};
//template <typename _Ty, typename _Pr>
//_Pr priority_queue<_Ty, _Pr>::_Pred;
void seed_rand();
int nextInt(int max);

template<typename _Ty>
struct SubList {
	_Ty *_begin, *_end;
	struct iterator {
		_Ty *ptr;

		iterator& operator++() {
			++ptr;
			return *this;
		}

		bool operator!=(const iterator &other) const {
			return ptr != other.ptr;
		}

		_Ty& operator*() {
			return *ptr;
		}
	};

	struct const_iterator {
		_Ty *ptr;

		const_iterator& operator++();
		bool operator!=(const const_iterator &other) const;
		const _Ty& operator*();
	};

	iterator begin() {
		return {_begin};
	}
	iterator end() {
		return {_end};
	}
	const_iterator begin() const;
	const_iterator end() const;
};

template<typename _Ty>
SubList<_Ty> subList(vector<_Ty> &list, int start, int end) {
	return SubList<_Ty> { list.data() + start, list.data() + end };
}

template<typename _Ty>
vector<_Ty> copyOfRange(const vector<_Ty> &list, int start, int end) {
	vector<_Ty> arr(end - start);
	int index = 0;
	for (int i = start; i < end; ++i) {
		arr[index++] = list[i];
	}
	return arr;

}

template<typename _Ty>
vector<_Ty> copier(const vector<_Ty> &a, const vector<_Ty> &b) {
	vector<_Ty> c(a.size() + b.size());
	int index = 0;
	for (auto &e : a) {
		c[index++] = e;
	}
	for (auto &e : b) {
		c[index++] = e;
	}
	return c;
}

template<typename Key, typename Value>
bool operator ==(const std::map<Key, Value*> &lhs,
		const std::map<Key, Value*> &rhs) {
	if (lhs.size() != rhs.size())
		return false;
	auto q = rhs.begin();
	for (auto p : lhs) {
		if (p.first != q->first)
			return false;
		if (*p.second != *q->second)
			return false;
		++q;
	}
	return true;
}
