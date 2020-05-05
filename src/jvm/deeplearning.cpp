//"${JAVA_HOME}/include"
//"${JAVA_HOME}/include/win32"
#include <stdio.h>
#include <cstring>

#include "../std/utility.h"
#include "../deeplearning/bert.h"
#include "../deeplearning/NERTagger.h"
#include "../deeplearning/lagacy.h"
#include "../deeplearning/classification.h"
#include "../deeplearning/CWSTagger.h"
#include "../deeplearning/POSTagger.h"
#include "../deeplearning/SyntaxParser.h"

#include "java.h"
extern "C" {
void JNICALL Java_com_util_Native_initializeH5Model(JNIEnv *env, jobject obj,
		jstring pwd) {
	workingDirectory = CString(env, pwd);
	switch (workingDirectory.back()) {
	case '/':
	case '\\':
		break;
	default:
		workingDirectory += '/';
	}
	cout << "initialize workingDirectory = " << workingDirectory << endl;

//	test_eigen();

	CWSTagger::instance();
	POSTagger::instance();
	SyntaxParser::instance();
	ClassifierChar::keyword_cn_classifier();
	ClassifierWord::keyword_en_classifier();
}

jint JNICALL Java_com_util_Native_sum8args(JNIEnv *env, jobject obj, jint rcx,
		jint rdx, jint r8, jint r9, jint fifthArg, jint sixthArg,
		jint seventhArg, jint eighthArg) {
	return sum8args(rcx, rdx, r8, r9, fifthArg, sixthArg, seventhArg, eighthArg);
}

jdouble JNICALL Java_com_util_Native_relu(JNIEnv *env, jobject obj,
		jdouble rcx) {
	return relu(rcx);
}

jint JNICALL Java_com_util_Native_gcdint(JNIEnv *env, jobject obj, jint rcx,
		jint rdx) {
	return gcd_int(rcx, rdx);
}

jlong JNICALL Java_com_util_Native_gcdlong(JNIEnv *env, jobject obj, jlong rcx,
		jlong rdx) {
	return gcd_long(rcx, rdx);
}

jint JNICALL Java_com_util_Native_gcdinttemplate(JNIEnv *env, jobject obj,
		jint rcx, jint rdx) {
	return gcd(rcx, rdx);
}

jlong JNICALL Java_com_util_Native_gcdlongtemplate(JNIEnv *env, jobject obj,
		jlong rcx, jlong rdx) {
	return gcd(rcx, rdx);
}

jobjectArray JNICALL Java_com_util_Native_NER(JNIEnv *env, jobject obj,
		jstring _service, jstring _text, jintArray _code) {
	__cout(__PRETTY_FUNCTION__)
	string service = CString(env, _service);
	String text = JString(env, _text);
	JArray<int> code(env, _code);
	cout << "code from java = " << code << endl;

	VectorI arr = code;
	cout << "converted to C++ = " << arr << endl;

	vector<vector<vector<double>>> debug;
	NERTaggerDict::_predict(service, text, arr, debug);
	return Object(env, debug);
}

jdouble JNICALL Java_com_util_Native_qatype(JNIEnv *env, jobject obj,
		jstring str) {
	__cout(__PRETTY_FUNCTION__)
	String s = JString(env, str);
	return Classifier::qatype_classifier().predict(s)[1];
}

jdouble JNICALL Java_com_util_Native_phatic(JNIEnv *env, jobject obj,
		jstring str) {
	__cout(__PRETTY_FUNCTION__)
	String s = JString(env, str);
	return Classifier::phatic_classifier().predict(s)[1];
}

jobjectArray JNICALL Java_com_util_Native_segmentCN(JNIEnv *env, jobject obj,
		jstring text) {
//	__cout(__PRETTY_FUNCTION__)
	String s = JString(env, text);
	return Object(env, CWSTagger::instance().predict(s));
}

//inputs: String x, String y;
//ouputs: double ;

jstring JNICALL Java_com_util_Native_hyponymCN(JNIEnv *env, jobject _,
		jstring jhypernym, jstring jhyponym) {
//	__cout(__PRETTY_FUNCTION__)
	String hypernym = JString(env, jhypernym);
	String hyponym = JString(env, jhyponym);
	auto probability = PairwiseVector::hyponymCN()(hypernym, hyponym);
	if (probability >= 0.9)
		return Object(env, u"hyponym");
	if (probability >= 0.8)
		return Object(env, u"synonym");
	if (probability >= 0.5)
		return Object(env, u"related");
	if (probability >= 0.1)
		return Object(env, u"unrelated");
	return Object(env, u"hypernym");
}

jstring JNICALL Java_com_util_Native_hyponymEN(JNIEnv *env, jobject _,
jstring jhypernym, jstring jhyponym) {
//	__cout(__PRETTY_FUNCTION__)
			String hypernym = JString(env, jhypernym);
			String hyponym = JString(env, jhyponym);
			auto probability = PairwiseVector::hyponymEN()(hypernym, hyponym);
			if (probability >= 0.9)
			return Object(env, u"hyponym");
			if (probability >= 0.8)
			return Object(env, u"synonym");
			if (probability >= 0.5)
			return Object(env, u"related");
			if (probability >= 0.1)
			return Object(env, u"unrelated");
			return Object(env, u"hypernym");
		}

		jobjectArray JNICALL Java_com_util_Native_hyponymCNs(JNIEnv *env, jobject _,jobjectArray jtext) {
//	__cout(__PRETTY_FUNCTION__)
			return Object(env, PairwiseVector::hyponymCN()(JArray<String>(env, jtext)));
		}

		jobjectArray JNICALL Java_com_util_Native_hyponymENs(JNIEnv *env, jobject _,jobjectArray jtext) {
//	__cout(__PRETTY_FUNCTION__)
			return Object(env, PairwiseVector::hyponymEN()(JArray<String>(env, jtext)));
		}

//inputs: String [] keywords;
//ouputs: int [] heads;

		jintArray JNICALL Java_com_util_Native_hyponymStructureCN(JNIEnv *env, jobject _, jobjectArray keywords, jintArray frequency) {
			__cout(__PRETTY_FUNCTION__);
			JArray<int> jArray(env, frequency);
			jArray = hyponymStructureCN(JArray<String>(env, keywords), jArray);
			return frequency;
		}

		jintArray JNICALL Java_com_util_Native_hyponymStructureEN(JNIEnv *env, jobject _, jobjectArray seg, jintArray frequency) {
			__cout(__PRETTY_FUNCTION__)
			return Object(env, hyponymStructureEN(JArray<String>(env, seg), JArray<int>(env, frequency)));
		}

		//inputs: String [] text;
//ouputs: String [][] segment;

		jintArray JNICALL Java_com_util_Native_depCN(JNIEnv *env, jobject _, jobjectArray seg, jobjectArray pos, jobjectArray dep) {
			__cout(__PRETTY_FUNCTION__)
			vector<String> depCPP;
			auto ret = Object(env,
			SyntaxParser::instance().predict(JArray<String>(env, seg),
					JArray<String>(env, pos), depCPP));

			JArray<String> depJava(env, dep);
			for (int i = 0, size = depCPP.size(); i < size; ++i) {
				depJava[i] = depCPP[i];
			}

			return ret;
		}

//inputs: String [] text;
//ouputs: String [][] segment;

		jobjectArray JNICALL Java_com_util_Native_posCN(JNIEnv *env, jobject _,
		jobjectArray text) {
//	__cout(__PRETTY_FUNCTION__)
			return Object(env, POSTagger::instance().predict(JArray<String>(env, text)));
		}

//inputs: String [] text;
//ouputs: String [][] segment;

		jobjectArray JNICALL Java_com_util_Native_segmentCNs(JNIEnv *env, jobject _,
		jobjectArray text) {
			__cout(__PRETTY_FUNCTION__)
			return Object(env, CWSTagger::instance().predict(JArray<String>(env, text)));
		}
//inputs: String [][] text;
//ouputs: String [][][] segment;
		jobjectArray JNICALL Java_com_util_Native_segmentCNss(JNIEnv *env, jobject _,
		jobjectArray text) {
//	__cout(__PRETTY_FUNCTION__)
			return Object(env,
			CWSTagger::instance().predict(JArray<vector<String>>(env, text)));
		}

		void JNICALL Java_com_util_Native_reinitializeCWSTagger(JNIEnv *env,
		jobject obj) {
			__cout(__PRETTY_FUNCTION__)
			CWSTagger::instantiate();
		}

		void JNICALL Java_com_util_Native_reinitializeLexiconCN(JNIEnv *env,
		jobject obj) {
			__cout(__PRETTY_FUNCTION__)
			PairwiseVector::instantiateHyponymCN();
		}

		void JNICALL Java_com_util_Native_reinitializeLexiconEN(JNIEnv *env,
		jobject obj) {
			__cout(__PRETTY_FUNCTION__)
			PairwiseVector::instantiateHyponymEN();
		}

		void JNICALL Java_com_util_Native_reinitializeKeywordCN(JNIEnv *env,
		jobject obj) {
			__cout(__PRETTY_FUNCTION__)
			ClassifierChar::instantiate_keyword_cn_classifier();
		}

		void JNICALL Java_com_util_Native_reinitializeKeywordEN(JNIEnv *env,
		jobject obj) {
			__cout(__PRETTY_FUNCTION__)
			ClassifierWord::instantiate_keyword_en_classifier();
		}

		jint JNICALL Java_com_util_Native_keywordCN(JNIEnv *env, jobject obj,
		jstring str) {
//	__cout(__PRETTY_FUNCTION__)
			String s = JString(env, str);
			int index;
			return ClassifierChar::keyword_cn_classifier().predict(s, index);
		}

		jdouble JNICALL Java_com_util_Native_keywordCNDouble(JNIEnv *env, jobject obj,
		jstring str) {
//	__cout(__PRETTY_FUNCTION__)
			String s = JString(env, str);
			return ClassifierChar::keyword_cn_classifier().predict_debug(s)[1];
		}

		jdouble JNICALL Java_com_util_Native_keywordENDouble(JNIEnv *env, jobject obj,
		jstring str) {
//	__cout(__PRETTY_FUNCTION__)
			String s = JString(env, str);
			return ClassifierWord::keyword_en_classifier().predict_debug(s)[1];
		}

		jintArray JNICALL Java_com_util_Native_keywordCNs(JNIEnv *env, jobject _,
		jobjectArray str) {
//	__cout(__PRETTY_FUNCTION__)
			vector<String> ss = JArray<String>(env, str);
			vector<int> index;
			return Object(env,
			ClassifierChar::keyword_cn_classifier().predict(ss, index));
		}

		jint JNICALL Java_com_util_Native_keywordEN(JNIEnv *env, jobject obj,
		jstring str) {
//	__cout(__PRETTY_FUNCTION__)
			String s = JString(env, str);
			int index;
			return ClassifierWord::keyword_en_classifier().predict(s, index);
		}

		jintArray JNICALL Java_com_util_Native_keywordENs(JNIEnv *env, jobject _,
		jobjectArray str) {
//	__cout(__PRETTY_FUNCTION__)
			vector<String> ss = JArray<String>(env, str);
			vector<int> index;
			return Object(env,
			ClassifierWord::keyword_en_classifier().predict(ss, index));
		}

		jdouble JNICALL Java_com_util_Native_similarity(JNIEnv *env, jobject obj,
		jstring x, jstring y) {
			__cout(__PRETTY_FUNCTION__)
			String s1 = JString(env, x);
			String s2 = JString(env, y);

			cout << "s1 = " << s1 << endl;
			cout << "s2 = " << s2 << endl;

			return Pairwise::paraphrase()(s1, s2);
		}

		jintArray JNICALL Java_com_util_Native_ner(JNIEnv *env, jobject obj,
		jstring _service, jstring _text, jintArray _code) {
//	__cout(__PRETTY_FUNCTION__)
			string service = CString(env, _service);
			String text = JString(env, _text);
			JArray<int> code(env, _code);

			{
				JArray<byte> code(env, 0);
			}
			{
				JArray<bool> code(env, 0);
			}
			{
				JArray<short> code(env, 0);
			}
			{
				JArray<long> code(env, 0);
			}
			{
				JArray<float> code(env, 0);
			}
			{
				JArray<double> code(env, 0);
			}
			{
				JArray<String> code(env, (jobjectArray) 0);
			}
//	cout << "code from java = " << code << endl;

			VectorI arr = code;
//	cout << "converted to C++ = " << arr << endl;

			NERTaggerDict::predict(service, text, arr);
			return Object(env, arr);
		}
	}
