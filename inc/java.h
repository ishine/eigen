#pragma once

#include <jni.h>
#include "Utility.h"

struct CString {
	CString(JNIEnv *env, jstring str) :
			env(env), str(str) {
		ptr = env->GetStringUTFChars(str, NULL);
	}

	operator string() const {
		return string(ptr);
	}

	bool operator !() const {
		return !ptr;
	}

	int length() const {
		return strlen(ptr);
	}

	~CString() {
		env->ReleaseStringUTFChars(str, ptr);
	}
	JNIEnv *env;
	jstring str;
	const char *ptr;
};

struct JString {
	JString(JNIEnv *env, jstring str) :
			env(env), str(str) {
		ptr = env->GetStringChars(str, NULL);
	}

	operator String() const {
		return String(ptr, ptr + this->length());
	}

	bool operator !() const {
		return !ptr;
	}

	int length() const {
		return env->GetStringLength(str);
	}

	~JString() {
		env->ReleaseStringChars(str, ptr);
	}
	JNIEnv *env;
	jstring str;
	const jchar *ptr;
};

struct JInteger {
	JInteger(JNIEnv *env, jintArray arr) :
			env(env), arr(arr) {
		ptr = env->GetIntArrayElements(arr, JNI_FALSE);
	}

	operator vector<int>() const {
		return vector<int>(ptr, ptr + this->length());
	}

	operator VectorI() const {
		return Eigen::Map<VectorI>((int*)ptr, this->length());
	}

	jint operator [](size_t i) const {
		return ptr[i];
	}

	jint &operator [](size_t i) {
		return ptr[i];
	}

	operator String() const {
		return String(ptr, ptr + this->length());
	}

	bool operator !() const {
		return !ptr;
	}

	jsize length() const {
		return env->GetArrayLength(arr);
	}

	~JInteger() {
		env->ReleaseIntArrayElements(arr, ptr, 0);
	}

	JNIEnv *env;
	jintArray arr;
	jint *ptr;
};

jstring Object(JNIEnv *env, const string &s);
jstring Object(JNIEnv *env, const String &s);

jbooleanArray Object(JNIEnv *env, const vector<bool> &s);
jcharArray Object(JNIEnv *env, const vector<char> &s);
jbyteArray Object(JNIEnv *env, const vector<byte> &s);
jshortArray Object(JNIEnv *env, const vector<short> &s);
jintArray Object(JNIEnv *env, const vector<int> &s);
jintArray Object(JNIEnv *env, const VectorI &s);
jfloatArray Object(JNIEnv *env, const vector<float> &s);
jlongArray Object(JNIEnv *env, const vector<long> &s);
jdoubleArray Object(JNIEnv *env, const vector<double> &s);

template<typename _Ty> struct FindClass {
};

template<>
struct FindClass<bool> {
	static const string name;
};

template<>
struct FindClass<byte> {
	static const string name;
};

template<>
struct FindClass<short> {
	static const string name;
};

template<>
struct FindClass<int> {
	static const string name;
};

template<>
struct FindClass<long> {
	static const string name;
};

template<>
struct FindClass<float> {
	static const string name;
};

template<>
struct FindClass<double> {
	static const string name;
};


template<typename _Ty>
struct FindClass<vector<_Ty>> {
	static const string name;
};

template<typename _Ty>
const string FindClass<vector<_Ty>>::name = "[" + FindClass<_Ty>::name;

template<typename _Ty>
jobjectArray Object(JNIEnv *env, const vector<_Ty> &arr) {
	int sz = arr.size();

	jobjectArray obj = env->NewObjectArray(sz,
			env->FindClass(FindClass<_Ty>::name.data()), NULL);

	for (int k = 0; k < sz; k++) {
		jobject local = Object(env, arr[k]);
		env->SetObjectArrayElement(obj, k, local);
		env->DeleteLocalRef(local);
	}
	return obj;
}

std::ostream& operator <<(std::ostream &cout, const JInteger &v);
