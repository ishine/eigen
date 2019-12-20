//http://eigen.tuxfamily.org/dox/
//https://blog.csdn.net/zong596568821xp/article/details/81134406

#include <time.h>
#include <utility.h>
#include "classification.h"
#include "bert.h"
#include <string>
#include <iostream>
using namespace std;
int main(int argc, char *argv[]) {
	BinaryReader reader((cnModelsDirectory() + "bert/paraphrase/test.h5"));
	reader.read_hdf5();
//	Text::test_utf_unicode_conversion();
//	auto &phatic = Classifier::phatic_classifier();
//	auto y = phatic.predict(u"请问您在哪个城市,请提供您的有效联系方式");
//	cout << y << endl;

//	auto &qatype = Classifier::qatype_classifier();
//	auto &y = qatype.predict(u"你很高吗?")[1];
//	cout << y << endl;
//	return 0;
	auto &paraphrase = Paraphrase::instance();

	double score = paraphrase(u"你们公司有些什么业务", u"你们公司业务有哪些");
	cout << "score = " << score << endl;

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
