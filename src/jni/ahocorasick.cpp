//"${JAVA_HOME}/include"
//"${JAVA_HOME}/include/win32"
#include <stdio.h>
#include <cstring>

#include "java.h"
#include "../ahocorasick/CWSTagger.h"
//template<>
//struct FindClass<Emit> {
//	static const char *name;
//	using jobject = jstring;
//	using jarray = jobjectArray;
//	static jobject* (JNIEnv::*GetArrayElements)(jarray array, jboolean *isCopy);
//	static void (JNIEnv::*ReleaseArrayElements)(jarray array, jobject *elems,
//			jint mode);
//};
//
//const char *FindClass<Emit>::name = "org/ahocorasick/trie/Emit";

//jobject Object(JNIEnv *env, const Emit &s) {
//	auto jclass = env->FindClass("org/ahocorasick/trie/Emit");
////	JavaNative Interface FieldDescriptors
//	auto jmethodID = env->GetMethodID(jclass, "<init>",
//			"(IILjava/lang/String;)V");
//	auto jobject = env->NewObject(jclass, jmethodID, s.start, s.end,
//			Object(env, s.value));
//	env->DeleteLocalRef(jclass);
//	return jobject;
//}

extern "C" {
//void JNICALL Java_org_dll_Native_initializeAhocorasickDictionary(JNIEnv *env,
//		jobject obj, jstring pwd) {
//	ahocorasick::initialize(CString(env, pwd));
//}
//
//void JNICALL Java_org_dll_Native_ahocorasickTest(JNIEnv *env, jobject obj) {
//	ahocorasick::test();
//}

//jobjectArray JNICALL Java_org_dll_Native_parseText(JNIEnv *env, jobject obj,
//		jstring jText) {
//	auto start = clock();
//	JString text(env, jText);
//
//	auto ending = clock();
//	cout << "initialization time cost = " << (ending - start) << endl;
//	start = ending;
//
//	auto emit = ahocorasick::instance.parseText(text.ptr, text.length());
//	ending = clock();
//	cout << "parsing time cost     = " << (ending - start) << endl;
//
//	return Object(env, emit);
//}

jobjectArray JNICALL Java_org_dll_Native_segmentCN(JNIEnv *env, jobject obj,
		jstring text) {
//	__debug(__PRETTY_FUNCTION__)
	String s = JString(env, text);
	return Object(env, CWSTagger::instance().segment(s));
}

//inputs: String [] text;
//ouputs: String [][] segment;

jobjectArray JNICALL Java_org_dll_Native_segmentCNs(JNIEnv *env, jobject _,
		jobjectArray text) {
	__debug(__PRETTY_FUNCTION__)
	return Object(env, CWSTagger::instance().segment(JArray<String>(env, text)));
}
//inputs: String [][] text;
//ouputs: String [][][] segment;
jobjectArray JNICALL Java_org_dll_Native_segmentCNss(JNIEnv *env, jobject _,
		jobjectArray text) {
//	__debug(__PRETTY_FUNCTION__)
	return Object(env,
			CWSTagger::instance().segment(JArray<vector<String>>(env, text)));
}

}

//https://linux.thai.net/~thep/datrie/datrie.html
//https://github.com/komiya-atsushi/darts-java
//https://www.hankcs.com/program/algorithm/aho-corasick-double-array-trie.html
