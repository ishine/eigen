//"${JAVA_HOME}/include"
//"${JAVA_HOME}/include/win32"
#include <stdio.h>
#include <cstring>

#include "java.h"
#include "../ahocorasick/public.h"

extern "C" {
void JNICALL Java_com_util_Native_initializeAhocorasickDictionary(JNIEnv *env,
		jobject obj, jstring pwd) {
	ahocorasick::initialize(CString(env, pwd));
}

void JNICALL Java_com_util_Native_parseTextVoid(JNIEnv *env, jobject obj,
		jstring jText, jobjectArray array) {

	JString text(env, jText);

	vector<int> begin, end;
	vector<String> value;

	ahocorasick::instance.parseText(text.ptr, text.length(), begin, end, value);

//	auto array = env->NewObjectArray(3, env->FindClass("java/lang/Object"),
//			nullptr);
	env->SetObjectArrayElement(array, 0, Object(env, begin));
	env->SetObjectArrayElement(array, 1, Object(env, end));
	env->SetObjectArrayElement(array, 2, Object(env, value));
//	return array;
}

jobjectArray JNICALL Java_com_util_Native_parseText(JNIEnv *env, jobject obj,
		jstring jText) {

	JString text(env, jText);

	vector<int> begin, end;
	vector<String> value;

	ahocorasick::instance.parseText(text.ptr, text.length(), begin, end, value);

	auto array = env->NewObjectArray(3, env->FindClass("java/lang/Object"),
			nullptr);
	env->SetObjectArrayElement(array, 0, Object(env, begin));
	env->SetObjectArrayElement(array, 1, Object(env, end));
	env->SetObjectArrayElement(array, 2, Object(env, value));
	return array;
}

}

