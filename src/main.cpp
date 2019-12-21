//http://eigen.tuxfamily.org/dox/
//https://blog.csdn.net/zong596568821xp/article/details/81134406

#include <time.h>
#include <utility.h>
#include "classification.h"
#include "bert.h"
#include <string>
#include <iostream>
using namespace std;

int create_hdf5_file();

int main(int argc, char *argv[]) {

//	reader.read_hdf5();
//	Text::test_utf_unicode_conversion();
	auto &phatic = Classifier::phatic_classifier();
	cout << phatic.predict(u"请问您在哪个城市,请提供您的有效联系方式")[1] << endl;

	auto &qatype = Classifier::qatype_classifier();
	cout << qatype.predict(u"你很高吗?")[1] << endl;

	auto &paraphrase = Paraphrase::instance();

	cout << "score = " << paraphrase(u"你们公司有些什么业务", u"你们公司业务有哪些") << endl;

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
