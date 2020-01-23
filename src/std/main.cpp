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

//http://eigen.tuxfamily.org/dox/
//https://blog.csdn.net/zong596568821xp/article/details/81134406
#include <time.h>
#include <string>
#include <iostream>
using namespace std;

int create_hdf5_file();
//to setup hdf5 project, use
//eigen.exe H5Tpkg.c

#include <stdio.h>

#include <omp.h>//调用Openmp库函数

#include "../deeplearning/utility.h"
#include "../deeplearning/classification.h"
#include "../deeplearning/bert.h"
#include "../deeplearning/lagacy.h"

void test_eigen() {
	Matrix A = Matrix::Ones(2560, 2560);
	Matrix B = Matrix::Ones(2560, 2560);
	auto start = clock();
	Matrix C = A * B;
	auto end = clock();
	cout << "time cost = " << (end - start) << endl;
}

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

#include "../ahocorasick/public.h"
int main(int argc, char **argv) {
	{
		Object old = return_tmp();
		Object tmp = old;
		cout << tmp;
	}
//	test();
//	test_matmul();
//	test_eigen();
//	return 0;
//	reader.read_hdf5();
//	Text::test_utf_unicode_conversion();

	auto &phatic = Classifier::phatic_classifier();
	auto &qatype = Classifier::qatype_classifier();
	auto &paraphrase = Paraphrase::instance();

	cout << "phatic = " << phatic.predict(u"请问您在哪个城市,请提供您的有效联系方式") << endl;

	cout << "qatype = " << qatype.predict(u"how are you today?") << endl;

	cout << "score = " << paraphrase(u"你们公司有些什么业务", u"你们公司业务有哪些") << endl;

	cout << "score = " << paraphrase(u"周末你去哪里玩", u"周末你去哪里玩") << endl;

	cout << "zero = " << zero << endl;
	cout << "one = " << one << endl;
	cout << "one_fifth = " << one_fifth << endl;
	cout << "half = " << half << endl;

	cout << "gcd_long(10, 46) = " << gcd_long(10, 46) << endl;
	cout << "gcd_qword(10, 46) = " << gcd_qword(10, 46) << endl;
	cout << "gcd_int(10, 46) = " << gcd_int(10, 46) << endl;
	cout << "gcd_dword(10, 46) = " << gcd_dword(10, 46) << endl;

	cout << "relu(10.1) = " << relu(10.1) << endl;
	cout << "relu(0.0) = " << relu(0.0) << endl;
	cout << "relu(-10.1) = " << relu(-10.1) << endl;
	cout << "hard_sigmoid(-10.1) = " << hard_sigmoid(-10.1) << endl;
	cout << "hard_sigmoid(10.1) = " << hard_sigmoid(10.1) << endl;
	cout << "hard_sigmoid(2.5) = " << hard_sigmoid(2.5) << endl;
	cout << "hard_sigmoid(-2.5) = " << hard_sigmoid(-2.5) << endl;
	cout << "hard_sigmoid(0) = " << hard_sigmoid(0) << endl;

	cout << "sum8args(1, 2, 3, 4, 5, 6, 7, 8) = "
			<< sum8args(1, 2, 3, 4, 5, 6, 7, 8) << endl;

//	try {
//		ahocorasick::initialize("../corpus/ahocorasick/en/dictionary.txt", 100);
//		ahocorasick::test();
//	} catch (std::exception &e) {
//		cout << e.what() << endl;
//	} catch (...) {
//		cout << "unknown error!" << endl;
//	}
//
	void print_primitive_type_size();
	print_primitive_type_size();
	return 0;
}
