#include <time.h>
#include <string>
#include <iostream>
using namespace std;

#include <stdio.h>

//#include "../../std/src/std/lagacy.h"

#include "deeplearning/utility.h"

//#include "deeplearning/classification.h"
//#include "deeplearning/bert.h"

#include "deeplearning/CWSTagger.h"
//#include "deeplearning/SyntaxParser.h"

int main(int argc, char **argv) {
	cout << "argc = " << argc << endl;
	for (int i = 0; i < argc; ++i) {
		cout << argv[i] << endl;
	}

	if (1 < argc) {
		workingDirectory = argv[1];
		workingDirectory += '/';
		cout << "workingDirectory = " << workingDirectory << endl;
		modelsDirectory() = workingDirectory + "models/";
//
		cout << "modelsDirectory = " << modelsDirectory() << endl;
//
//		PairwiseVectorChar::model_path = modelsDirectory()
//				+ "cn/lexicon/model.h5";
//		PairwiseVectorChar::config_path = modelsDirectory()
//				+ "cn/lexicon/config.json";
//		PairwiseVectorChar::vocab_path = modelsDirectory()
//				+ "cn/bert/vocab.txt";
//
//		en_vocab_path = modelsDirectory()
//				+ "en/bert/albert_base/30k-clean.model";
//
//		PairwiseVectorSP::config_path = modelsDirectory()
//				+ "en/lexicon/config.json";
//		PairwiseVectorSP::model_path = modelsDirectory()
//				+ "en/lexicon/model.h5";
//
//		ClassifierChar::model_path = modelsDirectory() + "cn/keyword/model.h5";
//		ClassifierChar::vocab_path = modelsDirectory() + "cn/keyword/vocab.txt";
//
//		ClassifierWord::model_path = modelsDirectory() + "en/keyword/model.h5";
//		ClassifierWord::vocab_path = modelsDirectory() + "en/keyword/vocab.txt";
//
		CWSTagger::model_path = modelsDirectory() + "cn/cws/model.h5";
		CWSTagger::vocab_path = modelsDirectory() + "cn/cws/vocab.txt";
	}

//	auto &lexiconSP = PairwiseVectorSP::instance();
//	cout << lexiconSP("abd", "deflkj") << endl;
//
////	auto &phatic = Classifier::phatic_classifier();
////	auto &qatype = Classifier::qatype_classifier();
//	auto &keyword_cn = ClassifierChar::instance();
//	auto &keyword_en = ClassifierWord::instance();
////	auto &paraphrase = Pairwise::paraphrase();
//	auto &lexicon = PairwiseVectorChar::instance();
//
//	cout << "lexicon = " << lexicon(u"承运", u"挡板") << endl;
	auto &cwsTagger = CWSTagger::instance();
//
	cout << "segments = " << cwsTagger.predict(u"(1) 圖示所揭露之虛線之部分，為本案不主張之部分。") << endl;
//
//	cout << "keyword = " << keyword_cn.predict(u"如图所示") << endl;
//
//	cout << "keyword = " << keyword_en.predict("Pairwise Algorithm") << endl;
//
//	auto &syntaxParser = SyntaxParser::instance();
//
//	{
//		vector<String> seg = { u"我们", u"研究", u"所有", u"东西", u"。"};
//	vector<String> pos = {u"PN", u"VT", u"JJ", u"NN", u"PU"};
//	vector<String> dep;
//	auto heads = syntaxParser.predict(seg, pos, dep);
//	cout << "seg = " << seg << endl;
//	cout << "pos = " << pos << endl;
//	cout << "dep = " << dep << endl;
//	cout << "heads = " << heads << endl;
//}
//	{
//		vector<String> seg = { u"你", u"说", u",", u"这", u"比", u"山", u"还", u"高", u"比", u"海", u"还", u"深", u"的", u"情谊", u",", u"我们", u"怎么", u"能", u"忘怀", u"?", u"仿写", u"句子"};
//	vector<String> pos = {u"PN", u"VT", u"PU", u"DT", u"P", u"NN", u"AD", u"VA", u"P", u"NN", u"AD", u"VA", u"DE", u"NN", u"PU", u"PN", u"AD", u"MD", u"VT", u"PU", u"VT", u"NN"};
//	vector<String> dep;
//	auto heads = syntaxParser.predict(seg, pos, dep);
//	cout << "seg = " << seg << endl;
//	cout << "pos = " << pos << endl;
//	cout << "dep = " << dep << endl;
//	cout << "heads = " << heads << endl;
//}
//
////	cout << "phatic = " << phatic.predict(u"请问您在哪个城市,请提供您的有效联系方式") << endl;
//
////	cout << "qatype = " << qatype.predict(u"how are you today?") << endl;
//
////	cout << "paraphrase score = " << paraphrase(u"你们公司有些什么业务", u"你们公司业务有哪些") << endl;
////	cout << "paraphrase score = " << paraphrase(u"周末你去哪里玩", u"今天他去哪里玩？") << endl;
//
//	cout << "lexicon score = " << lexicon(u"业务", u"公司业务") << endl;
//	cout << "lexicon score = " << lexicon(u"今晚", u"今天") << endl;
//
	cout << "test finished successfully!" << endl;
	return 0;
}
