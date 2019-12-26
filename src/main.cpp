//http://eigen.tuxfamily.org/dox/
//https://blog.csdn.net/zong596568821xp/article/details/81134406

#include <time.h>
#include <utility.h>
#include "classification.h"
#include "bert.h"
#include <string>
#include <iostream>
using namespace std;
#include "lagacy.h"
int create_hdf5_file();
//to setup hdf5 project, use
//eigen.exe H5Tpkg.c
int main(int argc, char **argv) {
//	reader.read_hdf5();
//	Text::test_utf_unicode_conversion();

	auto &phatic = Classifier::phatic_classifier();
	auto &qatype = Classifier::qatype_classifier();
	auto &paraphrase = Paraphrase::instance();

	cout << "phatic = " << phatic.predict(u"请问您在哪个城市,请提供您的有效联系方式") << endl;

	cout << "qatype = " << qatype.predict(u"你很高吗?") << endl;

	cout << "score = " << paraphrase(u"你们公司有些什么业务", u"你们公司业务有哪些") << endl;

	cout << "score = " << paraphrase(u"周末你去哪里玩", u"周末你去哪里玩") << endl;

//	cout << "zero = " << zero << endl;
//	cout << "one = " << one << endl;
//	cout << "one_fifth = " << one_fifth << endl;
//	cout << "half = " << half << endl;

	cout << "gcd_long(10, 46) = " << gcd_long(10, 46) << endl;
	cout << "gcd_qword(10, 46) = " << gcd_qword(10, 46) << endl;
	cout << "gcd_int(10, 46) = " << gcd_int(10, 46) << endl;
	cout << "gcd_dword(10, 46) = " << gcd_dword(10, 46) << endl;

	cout << "relu(10.1) = " << relu(10.1)<< endl;
	cout << "relu(0.0) = " << relu(0.0)<< endl;
	cout << "relu(-10.1) = " << relu(-10.1)<< endl;
	cout << "hard_sigmoid(-10.1) = " << hard_sigmoid(-10.1)<< endl;
	cout << "hard_sigmoid(10.1) = " << hard_sigmoid(10.1)<< endl;
	cout << "hard_sigmoid(2.5) = " << hard_sigmoid(2.5)<< endl;
	cout << "hard_sigmoid(-2.5) = " << hard_sigmoid(-2.5)<< endl;
	cout << "hard_sigmoid(0) = " << hard_sigmoid(0)<< endl;

	cout << "sum8args(1, 2, 3, 4, 5, 6, 7, 8) = " << sum8args(1, 2, 3, 4, 5, 6, 7, 8) << endl;

	return 0;
}

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
