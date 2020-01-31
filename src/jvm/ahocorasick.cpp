//"${JAVA_HOME}/include"
//"${JAVA_HOME}/include/win32"
#include <stdio.h>
#include <cstring>

#include "java.h"
#include "../ahocorasick/public.h"
template<>
struct FindClass<Emit> {
	static const string name;
	using jobject = jstring;
	using jarray = jobjectArray;
	static jobject* (JNIEnv::*GetArrayElements)(jarray array, jboolean *isCopy);
	static void (JNIEnv::*ReleaseArrayElements)(jarray array, jobject *elems,
			jint mode);
};

const string FindClass<Emit>::name = "org/ahocorasick/trie/Emit";

jobject Object(JNIEnv *env, const Emit &s) {
	auto jclass = env->FindClass("org/ahocorasick/trie/Emit");
//	JavaNative Interface FieldDescriptors
	auto jmethodID = env->GetMethodID(jclass, "<init>",
			"(IILjava/lang/String;)V");
	auto jobject = env->NewObject(jclass, jmethodID, s.start, s.end,
			Object(env, s.value));
	env->DeleteLocalRef(jclass);
	return jobject;
}

extern "C" {
void JNICALL Java_com_util_Native_initializeAhocorasickDictionary(JNIEnv *env,
		jobject obj, jstring pwd) {
	ahocorasick::initialize(CString(env, pwd));
}

void JNICALL Java_com_util_Native_ahocorasickTest(JNIEnv *env, jobject obj) {
	ahocorasick::test();
}

jobjectArray JNICALL Java_com_util_Native_parseText(JNIEnv *env, jobject obj,
		jstring jText) {
	auto start = clock();
	JString text(env, jText);

	vector<Emit> emit;

	auto ending = clock();
	cout << "initialization time cost = " << (ending - start) << endl;
	start = ending;

	ahocorasick::instance.parseText(text.ptr, text.length(), emit);
	ending = clock();
	cout << "parsing time cost = " << (ending - start) << endl;

	return Object(env, emit);
}

}

