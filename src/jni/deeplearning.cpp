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
#include <jni.h>
#include "../jni/java.h"

extern "C" {
void JNICALL Java_org_dll_Native_initializeWorkingDirectory(JNIEnv *env,
		jobject obj, jstring pwd) {
	__debug(__PRETTY_FUNCTION__);
	workingDirectory = CString(env, pwd);

	append_file_separator(workingDirectory);
	if (workingDirectory[0] == '~') {
		workingDirectory = getenv("HOME") + workingDirectory.substr(1);
	}

	weightsDirectory() = workingDirectory + "weights/";
	cout << "after initializing workingDirectory = " << workingDirectory
			<< endl;

	cout << "after initializing weightsDirectory = " << weightsDirectory()
			<< endl;
}

jintArray JNICALL Java_org_dll_Native_tokens2idsEN(JNIEnv *env, jobject _,
		jobjectArray text) {
	__debug(__PRETTY_FUNCTION__)
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
			ids.push_back(en_tokenizer().PieceToId(s));
			ids.push_back(en_tokenizer().PieceToId(comma));
		} else {
			ids.push_back(en_tokenizer().PieceToId(s));
		}
	}

	return Object(env, ids);
}

jobjectArray JNICALL Java_org_dll_Native_tokenizeEN(JNIEnv *env, jobject _,
		jstring text) {
	__debug(__PRETTY_FUNCTION__)
	vector<string> pieces;
	en_tokenizer().Encode((string) CString(env, text), &pieces);

	vector<string> _pieces;
	for (auto &s : pieces) {
		if (s.size() > 1 && s.back() == ',') {
			s.pop_back();
			_pieces.push_back(s);
			_pieces.push_back(",");
		} else {
			_pieces.push_back(s);
		}
	}

//	__log(pieces)
	return Object(env, _pieces);
}

//jint JNICALL Java_org_dll_Native_sum8args(JNIEnv *env, jobject obj, jint rcx,
//		jint rdx, jint r8, jint r9, jint fifthArg, jint sixthArg,
//		jint seventhArg, jint eighthArg) {
//	return sum8args(rcx, rdx, r8, r9, fifthArg, sixthArg, seventhArg, eighthArg);
//}

jdouble JNICALL Java_org_dll_Native_relu(JNIEnv *env, jobject obj,
		jdouble rcx) {
	return relu(rcx);
}

//jint JNICALL Java_org_dll_Native_gcdint(JNIEnv *env, jobject obj, jint rcx,
//		jint rdx) {
//	return gcd_int(rcx, rdx);
//}

//jlong JNICALL Java_org_dll_Native_gcdlong(JNIEnv *env, jobject obj, jlong rcx,
//		jlong rdx) {
//	return gcd_long(rcx, rdx);
//}

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
	__debug(__PRETTY_FUNCTION__)
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
	__debug(__PRETTY_FUNCTION__);
	String s = JString(env, str);
	return Classifier::qatype_classifier().predict(s)[1];
}

jdouble JNICALL Java_org_dll_Native_phatic(JNIEnv *env, jobject obj,
		jstring str) {
	__debug(__PRETTY_FUNCTION__);
	String s = JString(env, str);
	return Classifier::phatic_classifier().predict(s)[1];
}

jobjectArray JNICALL Java_org_dll_Native_lexiconEmbeddings(JNIEnv *env,
		jobject _, jint lang, jobjectArray keywords) {
	__log(__PRETTY_FUNCTION__);
	vector<String> text = JArray<String>(env, keywords);
	int size = text.size();
	vector<vector<double>> matrix(size);
	if (lang) {
		auto &model = PretrainingAlbertChinese::instance();
#pragma omp parallel for

		for (int i = 0; i < size; ++i) {
			Vector embedding = model(text[i]);
			auto begin = embedding.data();
			matrix[i] = compress(begin, begin + embedding.size(), 6);
		}

	} else {
		__debug(__PRETTY_FUNCTION__);
		auto &model = PretrainingAlbertEnglish::instance();
#pragma omp parallel for
		for (int i = 0; i < size; ++i) {
			Vector embedding = model(text[i]);
			auto begin = embedding.data();
			matrix[i] = compress(begin, begin + embedding.size(), 6);
		}

	}
	return Object(env, matrix);
}

jdoubleArray JNICALL Java_org_dll_Native_lexiconEmbedding(JNIEnv *env,
		jobject _, jint lang, jstring keywords) {
//	__log(__PRETTY_FUNCTION__);
	String text = JString(env, keywords);
//	__log(text);
	auto embedding =
			lang ? PretrainingAlbertChinese::instance()(text) : PretrainingAlbertEnglish::instance()(
							text);
	auto begin = embedding.data();
	vector<double> matrix(begin, begin + embedding.size());
	return Object(env, matrix);
}

//inputs: String [] text;
//ouputs: String [][] segment;

jintArray JNICALL Java_org_dll_Native_depCN(JNIEnv *env, jobject _,
		jobjectArray seg, jobjectArray pos, jobjectArray dep, jintArray heads) {
	__debug(__PRETTY_FUNCTION__);
	auto &instance = SyntaxParser::instance();
	vector<String> segCPP = JArray<String>(env, seg);
	vector<String> posCPP = JArray<String>(env, pos);

	vector<String> depCPP;
	if (heads == nullptr) {
		heads = Object(env, instance.predict(segCPP, posCPP, depCPP));
	} else {
		JArray<int> headsJava(env, heads);
		vector<int> headsCPP = headsJava;
		instance.predict(segCPP, posCPP, depCPP, headsCPP);
		headsJava = headsCPP;
	}

	JArray<String> depJava(env, dep);
	depJava = depCPP;
	return heads;
}

//inputs: String [] text;
//ouputs: String [][] segment;

jobjectArray JNICALL Java_org_dll_Native_posCN(JNIEnv *env, jobject _,
		jobjectArray text, jobjectArray pos) {
	__debug(__PRETTY_FUNCTION__);
	vector<String> segCPP = JArray<String>(env, text);

	auto &instance = POSTagger::instance();
	if (pos == nullptr)
		return Object(env, instance.predict(segCPP));


	JArray<String> posJava(env, pos);
	vector<String> posCPP = posJava;

	__print(posCPP);

	instance.predict(segCPP, posCPP);

	posJava = posCPP;
	return pos;
}

jint JNICALL Java_org_dll_Native_keyword(JNIEnv *env, jobject obj, jint lang,
		jstring str) {
	__debug(__PRETTY_FUNCTION__)
	String s = JString(env, str);
	int index;
	if (lang)
		return ClassifierChar::instance().predict(s, index);
	else
		return ClassifierWord::instance().predict(s, index);
}

jdouble JNICALL Java_org_dll_Native_keywordCNDouble(JNIEnv *env, jobject obj,
		jstring str) {
	__debug(__PRETTY_FUNCTION__)
	String s = JString(env, str);
	return ClassifierChar::instance().predict_debug(s)[1];
}

jdouble JNICALL Java_org_dll_Native_keywordENDouble(JNIEnv *env, jobject _,
		jstring str) {
	__debug(__PRETTY_FUNCTION__)
	String s = JString(env, str);
	return ClassifierWord::instance().predict_debug(s)[1];
}

jintArray JNICALL Java_org_dll_Native_keywordCNs(JNIEnv *env, jobject _,
		jobjectArray str) {
	__debug(__PRETTY_FUNCTION__)
	vector<String> ss = JArray<String>(env, str);
	vector<int> index;
	return Object(env, ClassifierChar::instance().predict(ss, index));
}

jintArray JNICALL Java_org_dll_Native_keywordENs(JNIEnv *env, jobject _,
		jobjectArray str) {
	__debug(__PRETTY_FUNCTION__)
	vector<String> ss = JArray<String>(env, str);

	vector<int> index;
	return Object(env, ClassifierWord::instance().predict(ss, index));
}

jdouble JNICALL Java_org_dll_Native_similarity(JNIEnv *env, jobject obj,
		jstring x, jstring y) {
	__debug(__PRETTY_FUNCTION__)
	String s1 = JString(env, x);
	String s2 = JString(env, y);

	cout << "s1 = " << s1 << endl;
	cout << "s2 = " << s2 << endl;

	return Pairwise::paraphrase()(s1, s2);
}

jintArray JNICALL Java_org_dll_Native_ner(JNIEnv *env, jobject obj,
		jstring _service, jstring _text, jintArray _code) {
//	__debug(__PRETTY_FUNCTION__)
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
