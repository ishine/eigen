//"${JAVA_HOME}/include"
//"${JAVA_HOME}/include/win32"
#include <stdio.h>
#include <cstring>

#include "../std/utility.h"
#include "../std/lagacy.h"

#include "../deeplearning/bert.h"
#include "../deeplearning/NERTagger.h"

#include "../deeplearning/classification.h"
#include "../deeplearning/CWSTagger.h"
#include "../deeplearning/POSTagger.h"
#include "../deeplearning/SyntaxParser.h"

#include "java.h"
extern "C" {
void JNICALL Java_org_dll_Native_initializeH5Model(JNIEnv *env, jobject obj,
		jstring pwd) {
	workingDirectory = CString(env, pwd);
	switch (workingDirectory.back()) {
	case '/':
	case '\\':
		break;
	default:
		workingDirectory += '/';
	}

	if (workingDirectory[0] == '~') {
		workingDirectory = getenv("HOME") + workingDirectory.substr(1);
	}

	cout << "after initializing workingDirectory = " << workingDirectory
			<< endl;
}

jintArray JNICALL Java_org_dll_Native_tokens2idsEN(JNIEnv *env, jobject _,
		jobjectArray text) {
	__cout(__PRETTY_FUNCTION__)
	vector<string> ctext = JArray<string>(env, text);
//	__log(ctext)
	return Object(env, en_tokenizer().PieceToId(ctext));

}

jintArray JNICALL Java_org_dll_Native_token2idEN(JNIEnv *env, jobject _,
		jstring text) {
	static const string comma(1, ',');
	std::vector<std::string> pieces;
	en_tokenizer().Encode((string) CString(env, text), &pieces);

	std::vector<int> ids;
	for (auto &s : pieces) {
		if (s.size() > 1 && s.back() == ',') {
			s.pop_back();
			ids << en_tokenizer().PieceToId(s);
			ids << en_tokenizer().PieceToId(comma);
		} else {
			ids << en_tokenizer().PieceToId(s);
		}
	}

	return Object(env, ids);
}

jobjectArray JNICALL Java_org_dll_Native_tokenizeEN(JNIEnv *env, jobject _,
		jstring text) {
	__cout(__PRETTY_FUNCTION__)
	vector<string> pieces;
	en_tokenizer().Encode((string) CString(env, text), &pieces);

//	__log(pieces)
	return Object(env, pieces);
}

jint JNICALL Java_org_dll_Native_sum8args(JNIEnv *env, jobject obj, jint rcx,
		jint rdx, jint r8, jint r9, jint fifthArg, jint sixthArg,
		jint seventhArg, jint eighthArg) {
	return sum8args(rcx, rdx, r8, r9, fifthArg, sixthArg, seventhArg, eighthArg);
}

jdouble JNICALL Java_org_dll_Native_relu(JNIEnv *env, jobject obj,
		jdouble rcx) {
	return relu(rcx);
}

jint JNICALL Java_org_dll_Native_gcdint(JNIEnv *env, jobject obj, jint rcx,
		jint rdx) {
	return gcd_int(rcx, rdx);
}

jlong JNICALL Java_org_dll_Native_gcdlong(JNIEnv *env, jobject obj, jlong rcx,
		jlong rdx) {
	return gcd_long(rcx, rdx);
}

jint JNICALL Java_org_dll_Native_gcdinttemplate(JNIEnv *env, jobject obj,
		jint rcx, jint rdx) {
	return gcd(rcx, rdx);
}

jlong JNICALL Java_org_dll_Native_gcdlongtemplate(JNIEnv *env, jobject obj,
		jlong rcx, jlong rdx) {
	return gcd(rcx, rdx);
}

jobjectArray JNICALL Java_org_dll_Native_NER(JNIEnv *env, jobject obj,
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

jdouble JNICALL Java_org_dll_Native_qatype(JNIEnv *env, jobject obj,
		jstring str) {
	__cout(__PRETTY_FUNCTION__)
	String s = JString(env, str);
	return Classifier::qatype_classifier().predict(s)[1];
}

jdouble JNICALL Java_org_dll_Native_phatic(JNIEnv *env, jobject obj,
		jstring str) {
	__cout(__PRETTY_FUNCTION__)
	String s = JString(env, str);
	return Classifier::phatic_classifier().predict(s)[1];
}

jobjectArray JNICALL Java_org_dll_Native_segmentCN(JNIEnv *env, jobject obj,
		jstring text) {
//	__cout(__PRETTY_FUNCTION__)
	String s = JString(env, text);
	return Object(env, CWSTagger::instance().predict(s));
}

//inputs: String x, String y;
//ouputs: String ;

jdouble JNICALL Java_org_dll_Native_relevance(JNIEnv *env, jobject _, jint lang,
		jstring jx, jstring jy) {
	__cout(__PRETTY_FUNCTION__)
	String text = JString(env, jx);
	String derivant = JString(env, jy);

	__cout(text)
	__cout(derivant)
	Vector probability =
			lang ? PairwiseVectorChar::instance()(text, derivant) : PairwiseVectorSP::instance()(
							text, derivant);

	return probability(0);
}

jobjectArray JNICALL Java_org_dll_Native_lexiconMutualScoreWithEmbedding(
		JNIEnv *env, jobject _, jint lang, jobjectArray jdoubleArrayArray) {
	__cout(__PRETTY_FUNCTION__)
	auto &model =
			lang ? (PairwiseVector&) PairwiseVectorChar::instance() : (PairwiseVector&) PairwiseVectorSP::instance();
	return Object(env, model(JArray<Vector>(env, jdoubleArrayArray)));
}

jobjectArray JNICALL Java_org_dll_Native_lexiconMutualScoreCNs(JNIEnv *env,
		jobject _, jobjectArray jtext) {
	__cout(__PRETTY_FUNCTION__)
	return Object(env,
			PairwiseVectorChar::instance()(JArray<String>(env, jtext)));
}

jobjectArray JNICALL Java_org_dll_Native_lexiconMutualScoreENs(JNIEnv *env,
		jobject _, jobjectArray jtext) {
//	__cout(__PRETTY_FUNCTION__)
	return Object(env, PairwiseVectorSP::instance()(JArray<string>(env, jtext)));
}

//inputs: String [] keywords;
//ouputs: int [] heads;

jintArray JNICALL Java_org_dll_Native_lexiconStructureCN(JNIEnv *env, jobject _,
		jobjectArray keywords, jintArray frequency) {
	__cout(__PRETTY_FUNCTION__);
	JArray<int> jArray(env, frequency);
	jArray = lexiconStructure(JArray<String>(env, keywords), jArray);
	return frequency;
}

jintArray JNICALL Java_org_dll_Native_lexiconStructureWithEmbedding(JNIEnv *env,
		jobject _, jint lang, jobjectArray jEmbedding,
		jobjectArray jHierarchicalMatrix, jintArray frequency) {
	__cout(__PRETTY_FUNCTION__);
	JArray<int> jArray(env, frequency);
	JArray<vector<double>> jDoubleDoubleArray(env, jEmbedding);
	vector<vector<double>> cDoubleDoubleVector = jDoubleDoubleArray;
	vector<vector<double>> cHierarchicalMatrix = JArray<vector<double>>(env,
			jHierarchicalMatrix);

	jArray = lexiconStructure(lang, cDoubleDoubleVector, cHierarchicalMatrix,
			jArray);
	return frequency;
}

jobjectArray JNICALL Java_org_dll_Native_lexiconEmbedding(JNIEnv *env,
		jobject _, jint lang, jobjectArray keywords) {
	vector<String> text = JArray<String>(env, keywords);
	int size = text.size();
	vector<vector<double>> matrix(size);
	if (lang) {
		auto &model = PairwiseVectorChar::instance();
#pragma omp parallel for
		for (int i = 0; i < size; ++i) {
			Vector embedding = model(text[i]);
			auto begin = embedding.data();
			matrix[i].assign(begin, begin + embedding.size());
		}

	} else {
		__cout(__PRETTY_FUNCTION__);
		auto &model = PairwiseVectorSP::instance();
#pragma omp parallel for
		for (int i = 0; i < size; ++i) {
			Vector embedding = model(text[i]);
			auto begin = embedding.data();
			matrix[i].assign(begin, begin + embedding.size());
		}

	}
	return Object(env, matrix);
}

jintArray JNICALL Java_org_dll_Native_lexiconStructureEN(JNIEnv *env, jobject _,
		jobjectArray seg, jintArray frequency) {
	__cout(__PRETTY_FUNCTION__)
	return Object(env,
			lexiconStructure(JArray<string>(env, seg),
					JArray<int>(env, frequency)));
}

//inputs: String [] text;
//ouputs: String [][] segment;

jintArray JNICALL Java_org_dll_Native_depCN(JNIEnv *env, jobject _,
		jobjectArray seg, jobjectArray pos, jobjectArray dep) {
	__cout(__PRETTY_FUNCTION__)
	vector<String> depCPP;
	auto ret = Object(env,
			SyntaxParser::instance().predict(JArray<String>(env, seg),
					JArray<String>(env, pos), depCPP));

	JArray<String> depJava(env, dep);
	depJava = depCPP;
	return ret;
}

//inputs: String [] text;
//ouputs: String [][] segment;

jobjectArray JNICALL Java_org_dll_Native_posCN(JNIEnv *env, jobject _,
		jobjectArray text) {
//	__cout(__PRETTY_FUNCTION__)
	return Object(env, POSTagger::instance().predict(JArray<String>(env, text)));
}

//inputs: String [] text;
//ouputs: String [][] segment;

jobjectArray JNICALL Java_org_dll_Native_segmentCNs(JNIEnv *env, jobject _,
		jobjectArray text) {
	__cout(__PRETTY_FUNCTION__)
	return Object(env, CWSTagger::instance().predict(JArray<String>(env, text)));
}
//inputs: String [][] text;
//ouputs: String [][][] segment;
jobjectArray JNICALL Java_org_dll_Native_segmentCNss(JNIEnv *env, jobject _,
		jobjectArray text) {
//	__cout(__PRETTY_FUNCTION__)
	return Object(env,
			CWSTagger::instance().predict(JArray<vector<String>>(env, text)));
}

void JNICALL Java_org_dll_Native_initializeCWSTagger(JNIEnv *env, jobject obj,
		jstring modelPath, jstring vocabPath) {
	__cout(__PRETTY_FUNCTION__)
	CWSTagger::model_path = CString(env, modelPath);
	CWSTagger::vocab_path = CString(env, vocabPath);
	CWSTagger::instance();
}

void JNICALL Java_org_dll_Native_initializeLexiconCN(JNIEnv *env, jobject obj,
		jstring configPath, jstring modelPath, jstring vocabPath) {
	__cout(__PRETTY_FUNCTION__)
	PairwiseVectorChar::config_path = CString(env, configPath);
	PairwiseVectorChar::model_path = CString(env, modelPath);
	PairwiseVectorChar::vocab_path = CString(env, vocabPath);
	PairwiseVectorChar::instance();
}

void JNICALL Java_org_dll_Native_initializeLexiconEN(JNIEnv *env, jobject obj,
		jstring configPath, jstring modelPath, jstring vocabPath) {
	__cout(__PRETTY_FUNCTION__)
	PairwiseVectorSP::config_path = CString(env, configPath);
	PairwiseVectorSP::model_path = CString(env, modelPath);
//	PairwiseVectorSP::vocab_model_path = CString(env, vocabModelPath);
	en_vocab_path = CString(env, vocabPath);
	PairwiseVectorSP::instance();
}

void JNICALL Java_org_dll_Native_initializeKeywordCN(JNIEnv *env, jobject obj,
		jstring modelPath, jstring vocabPath) {
	__cout(__PRETTY_FUNCTION__)
	ClassifierChar::model_path = CString(env, modelPath);
	ClassifierChar::vocab_path = CString(env, vocabPath);

	ClassifierChar::instance();
}

void JNICALL Java_org_dll_Native_initializeKeywordEN(JNIEnv *env, jobject obj,
		jstring modelPath, jstring vocabPath) {
	__cout(__PRETTY_FUNCTION__)
	ClassifierWord::model_path = CString(env, modelPath);
	ClassifierWord::vocab_path = CString(env, vocabPath);

	ClassifierWord::instance();
}

jint JNICALL Java_org_dll_Native_keyword(JNIEnv *env, jobject obj, jint lang,
		jstring str) {
	__cout(__PRETTY_FUNCTION__)
	String s = JString(env, str);
	int index;
	if (lang)
		return ClassifierChar::instance().predict(s, index);
	else
		return ClassifierWord::instance().predict(s, index);
}

jdouble JNICALL Java_org_dll_Native_keywordCNDouble(JNIEnv *env, jobject obj,
		jstring str) {
	__cout(__PRETTY_FUNCTION__)
	String s = JString(env, str);
	return ClassifierChar::instance().predict_debug(s)[1];
}

jdouble JNICALL Java_org_dll_Native_keywordENDouble(JNIEnv *env, jobject _,
		jstring str) {
	__cout(__PRETTY_FUNCTION__)
	String s = JString(env, str);
	return ClassifierWord::instance().predict_debug(s)[1];
}

jintArray JNICALL Java_org_dll_Native_keywordCNs(JNIEnv *env, jobject _,
		jobjectArray str) {
	__cout(__PRETTY_FUNCTION__)
	vector<String> ss = JArray<String>(env, str);
	vector<int> index;
	return Object(env, ClassifierChar::instance().predict(ss, index));
}

jintArray JNICALL Java_org_dll_Native_keywordENs(JNIEnv *env, jobject _,
		jobjectArray str) {
	__cout(__PRETTY_FUNCTION__)
	vector<String> ss = JArray<String>(env, str);

	vector<int> index;
	return Object(env, ClassifierWord::instance().predict(ss, index));
}

jdouble JNICALL Java_org_dll_Native_similarity(JNIEnv *env, jobject obj,
		jstring x, jstring y) {
	__cout(__PRETTY_FUNCTION__)
	String s1 = JString(env, x);
	String s2 = JString(env, y);

	cout << "s1 = " << s1 << endl;
	cout << "s2 = " << s2 << endl;

	return Pairwise::paraphrase()(s1, s2);
}

jintArray JNICALL Java_org_dll_Native_ner(JNIEnv *env, jobject obj,
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
