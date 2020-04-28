#include <time.h>
#include <string>
#include <iostream>
using namespace std;

#include <stdio.h>

#include "../deeplearning/utility.h"
#include "../deeplearning/classification.h"
#include "../deeplearning/bert.h"
#include "../deeplearning/lagacy.h"
#include "../deeplearning/CWSTagger.h"
#include "../deeplearning/SyntaxParser.h"
#include "../ahocorasick/public.h"

int main(int argc, char **argv) {
	cout << "argc = " << argc << endl;
	for (int i = 0; i < argc; ++i) {
		cout << argv[i] << endl;
	}

	if (1 < argc) {
		workingDirectory = argv[1];
	}

//	auto &phatic = Classifier::phatic_classifier();
//	auto &qatype = Classifier::qatype_classifier();
	auto &keyword_cn = ClassifierChar::keyword_cn_classifier();
	auto &keyword_en = ClassifierWord::keyword_en_classifier();
//	auto &paraphrase = Paraphrase::instance();

	auto &cwsTagger = CWSTagger::instance();

	cout << "segments = " << cwsTagger.predict(u"(1) 圖示所揭露之虛線之部分，為本案不主張之部分。") << endl;

	cout << "keyword = " << keyword_cn.predict(u"，ooooo") << endl;

	cout << "keyword = " << keyword_en.predict(u"Poly盲them") << endl;

	auto &syntaxParser = SyntaxParser::instance();

	vector<String> seg = { u"我们", u"研究", u"所有", u"东西" };
	vector<String> pos = { u"PN", u"VT", u"JJ", u"NN" };
	vector<String> dep;
	auto heads = syntaxParser.predict(seg, pos, dep);
	cout << "seg = " << seg << endl;
	cout << "pos = " << pos << endl;
	cout << "dep = " << dep << endl;
	cout << "heads = " << heads << endl;

//	cout << "phatic = " << phatic.predict(u"请问您在哪个城市,请提供您的有效联系方式") << endl;

//	cout << "qatype = " << qatype.predict(u"how are you today?") << endl;

//	cout << "score = " << paraphrase(u"你们公司有些什么业务", u"你们公司业务有哪些") << endl;

//	cout << "score = " << paraphrase(u"周末你去哪里玩", u"今天他去哪里玩？") << endl;

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

	cout << "cpu_count() = " << cpu_count << endl;
//	double pi_test(int n);

//	for (int n = 100; n < 10000; ++n) {
//		printf("pi_test(%d) = %f\n", n, pi_test(n));
//		assert(pi_test(n) > 3);
//	}
//	void test_eigen();
//	test_eigen();
	return 0;
}
//https://www.cnblogs.com/listenscience/p/11509164.html

//https://academy.zhihuiya.com/#/user/learning/index
//15821495341
//https://ks.wjx.top/jq/69262900.aspx
