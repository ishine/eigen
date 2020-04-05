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
	cout << "in " << __PRETTY_FUNCTION__ << endl;
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
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	String s = JString(env, str);
	return Classifier::qatype_classifier().predict(s)[1];
}

jdouble JNICALL Java_com_util_Native_phatic(JNIEnv *env, jobject obj,
		jstring str) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	String s = JString(env, str);
	return Classifier::phatic_classifier().predict(s)[1];
}

jstring JNICALL Java_com_util_Native_segmentCN(JNIEnv *env, jobject obj,
		jstring text) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
	String s = JString(env, text);
	return Object(env, CWSTagger::instance().predict(s));
}

void JNICALL Java_com_util_Native_reinitializeCWSTagger(JNIEnv *env,
		jobject obj) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	CWSTagger::instance(true);
}

void JNICALL Java_com_util_Native_reinitializeKeywordCN(JNIEnv *env,
		jobject obj) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	Classifier::keyword_cn_classifier(true);
}

void JNICALL Java_com_util_Native_reinitializeKeywordEN(JNIEnv *env,
		jobject obj) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	Classifier::keyword_en_classifier(true);
}

jdouble JNICALL Java_com_util_Native_keywordCN(JNIEnv *env, jobject obj,
		jstring str) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
	String s = JString(env, str);
	return Classifier::keyword_cn_classifier().predict(s)[1];
}

jdouble JNICALL Java_com_util_Native_keywordEN(JNIEnv *env, jobject obj,
		jstring str) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
	String s = JString(env, str);
	return Classifier::keyword_en_classifier().predict(s)[1];
}

jdouble JNICALL Java_com_util_Native_similarity(JNIEnv *env, jobject obj,
		jstring x, jstring y) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	String s1 = JString(env, x);
	String s2 = JString(env, y);

	cout << "s1 = " << s1 << endl;
	cout << "s2 = " << s2 << endl;

	return Paraphrase::instance()(s1, s2);
}

jintArray JNICALL Java_com_util_Native_ner(JNIEnv *env, jobject obj,
		jstring _service, jstring _text, jintArray _code) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
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
