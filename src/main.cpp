#include <time.h>
#include <string>
#include <iostream>
using namespace std;

#include <stdio.h>

#include "std/lagacy.h"

#include "deeplearning/utility.h"
#include "deeplearning/classification.h"
#include "deeplearning/bert.h"

#include "ahocorasick/CWSTagger.h"
#include "deeplearning/SyntaxParser.h"
#include "ahocorasick/KeyGenerator.h"
#include "deeplearning/keywordExpansion.h"

int main(int argc, char **argv) {
	cout << "argc = " << argc << endl;
	for (int i = 0; i < argc; ++i) {
		cout << argv[i] << endl;
	}

	if (1 < argc) {
		workingDirectory = argv[1];
		append_file_separator(workingDirectory);
		cout << "workingDirectory = " << workingDirectory << endl;

		weightsDirectory() = workingDirectory + "models/";
		cout << "modelsDirectory = " << weightsDirectory() << endl;
	}

	if (2 < argc) {
		testingDirectory = argv[2];
		testingDirectory += '/';
		cout << "testingDirectory = " << testingDirectory << endl;
	}

	//	KeyGenerator::test();
//	void testLoop();
//	testLoop();

	auto &cwsTagger = CWSTagger::instance();

	cout << "segments = " << cwsTagger.segment(u"结婚的和尚未结婚的确实在场地上散步。") << endl;

	cout << "segments = " << cwsTagger.segment(u"JAVAEE技术") << endl;

	auto &lexiconSP = PretrainingAlbertEnglish::instance();
	cout << lexiconSP("vector") << endl;

	void test_sentencepiece_keras(const string &s = "");
	test_sentencepiece_keras();
//	auto &phatic = Classifier::phatic_classifier();
//	auto &qatype = Classifier::qatype_classifier();
	auto &keyword_cn = ClassifierChar::instance();
	auto &keyword_en = ClassifierWord::instance();
//	auto &paraphrase = Pairwise::paraphrase();
	auto &lexicon = PretrainingAlbertChinese::instance();

	cout << "lexicon = " << lexicon(u"承运和挡板") << endl;

	cout << "keyword = " << keyword_cn.predict(u"如图所示") << endl;

	cout << "keyword = " << keyword_en.predict("Pairwise Algorithm") << endl;

	auto &syntaxParser = SyntaxParser::instance();

	{
		vector<String> seg = { u"我们", u"研究", u"所有", u"东西", u"。"};
	vector<String> pos = {u"PN", u"VT", u"JJ", u"NN", u"PU"};
	vector<String> dep;
	auto heads = syntaxParser.predict(seg, pos, dep);
	cout << "seg = " << seg << endl;
	cout << "pos = " << pos << endl;
	cout << "dep = " << dep << endl;
	cout << "heads = " << heads << endl;
}
	{
		vector<String> seg = { u"你", u"说", u",", u"这", u"比", u"山", u"还", u"高", u"比", u"海", u"还", u"深", u"的", u"情谊", u",", u"我们", u"怎么", u"能", u"忘怀", u"?", u"仿写", u"句子"};
	vector<String> pos = {u"PN", u"VT", u"PU", u"DT", u"P", u"NN", u"AD", u"VA", u"P", u"NN", u"AD", u"VA", u"DE", u"NN", u"PU", u"PN", u"AD", u"MD", u"VT", u"PU", u"VT", u"NN"};
	vector<String> dep;
	auto heads = syntaxParser.predict(seg, pos, dep);
	cout << "seg = " << seg << endl;
	cout << "pos = " << pos << endl;
	cout << "dep = " << dep << endl;
	cout << "heads = " << heads << endl;
}
//	cout << "phatic = " << phatic.predict(u"请问您在哪个城市,请提供您的有效联系方式") << endl;

//	cout << "qatype = " << qatype.predict(u"how are you today?") << endl;

//	cout << "paraphrase score = " << paraphrase(u"你们公司有些什么业务", u"你们公司业务有哪些") << endl;
//	cout << "paraphrase score = " << paraphrase(u"周末你去哪里玩", u"今天他去哪里玩？") << endl;

	cout << "zero = " << zero << endl;
	cout << "one = " << one << endl;
	cout << "one_fifth = " << one_fifth << endl;
	cout << "half = " << half << endl;

//	cout << "gcd_long(10, 46) = " << gcd_long(10, 46) << endl;
//	cout << "gcd_qword(10, 46) = " << gcd_qword(10, 46) << endl;
//	cout << "gcd_int(10, 46) = " << gcd_int(10, 46) << endl;
//	cout << "gcd_dword(10, 46) = " << gcd_dword(10, 46) << endl;

	cout << "relu(10.1) = " << relu(10.1) << endl;
	cout << "relu(0.0) = " << relu(0.0) << endl;
	cout << "relu(-10.1) = " << relu(-10.1) << endl;
	cout << "hard_sigmoid(-10.1) = " << hard_sigmoid(-10.1) << endl;
	cout << "hard_sigmoid(10.1) = " << hard_sigmoid(10.1) << endl;
	cout << "hard_sigmoid(2.5) = " << hard_sigmoid(2.5) << endl;
	cout << "hard_sigmoid(-2.5) = " << hard_sigmoid(-2.5) << endl;
	cout << "hard_sigmoid(0) = " << hard_sigmoid(0) << endl;

//	for (int n = 100; n < 10000; ++n) {
//		printf("pi_test(%d) = %f\n", n, pi_test(n));
//		assert(pi_test(n) > 3);
//	}
//	void test_eigen();
//	test_eigen();
	return 0;
}
//todo:
//https://blog.csdn.net/AMDS123/article/details/77284751?utm_source=blogxgwz9?utm_medium=distribute.pc_relevant.none-task-blog-baidujs-1
//openBlas
