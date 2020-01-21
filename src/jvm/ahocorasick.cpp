//"${JAVA_HOME}/include"
//"${JAVA_HOME}/include/win32"
#include <stdio.h>
#include <cstring>

#include "java.h"
#include "../ahocorasick/public.h"

extern "C" {
void JNICALL Java_com_util_Native_initializeAhocorasickDictionary(JNIEnv *env,
		jobject obj, jstring pwd) {
	string path4Dictionary = CString(env, pwd);
	switch (path4Dictionary.back()) {
	case '/':
	case '\\':
		break;
	default:
		workingDirectory += '/';
	}
	cout << "initialize workingDirectory = " << workingDirectory << endl;
}

void JNICALL Java_com_util_Native_parseText(JNIEnv *env, jobject obj,
		jstring jText, jobjectArray array) {

	JString text(env, jText);

	vector<int> begin, end;
	vector<String> value;

	ahocorasick::instance.parseText(text.ptr, text.length(), begin, end, value);
//	jintArray begin;
//	jintArray end;
//	jobjectArray value;
//
	env->SetObjectArrayElement(array, 0, Object(env, begin));
	env->SetObjectArrayElement(array, 1, Object(env, end));
	env->SetObjectArrayElement(array, 2, Object(env, value));
}

}

