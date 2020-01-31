#pragma once

#include <jni.h>
#include "../deeplearning/utility.h"

struct CString {
	CString(JNIEnv *env, jstring str) :
			env(env), str(str), ptr(env->GetStringUTFChars(str, nullptr)) {
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

	JNIEnv *const env;
	const jstring str;
	const char *const ptr;
};

struct JString {
	JString(JNIEnv *env, jstring str) :
			env(env), str(str), ptr(env->GetStringChars(str, nullptr)) {
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

	JNIEnv *const env;
	const jstring str;
	const jchar *const ptr;
};

template<typename _Ty> struct FindClass {
};

template<typename _Ty>
struct JArray {
	using jobject = typename FindClass<_Ty>::jobject;
	using jarray = typename FindClass<_Ty>::jarray;

	JArray(JNIEnv *env, jarray arr) :
			env(env), arr(arr), ptr(
					(env->*FindClass<_Ty>::GetArrayElements)(arr, JNI_FALSE)) {
	}

	operator vector<int>() const {
		return vector<int>(ptr, ptr + this->length());
	}

	operator VectorI() const {
		return Eigen::Map<VectorI>((int*) ptr, this->length());
	}

	const jobject& operator [](size_t i) const {
		return ptr[i];
	}

	jobject& operator [](size_t i) {
		return ptr[i];
	}

//	operator String() const {
//		return String(ptr, ptr + this->length());
//	}

	bool operator !() const {
		return !ptr;
	}

	jsize length() const {
		return env->GetArrayLength(arr);
	}

	~JArray() {
		(env->*FindClass<_Ty>::ReleaseArrayElements)(arr, ptr, 0);
	}

	JNIEnv *const env;
	const jarray arr;
	jobject *const ptr;
};

template<>
struct JArray<String> {
	JArray(JNIEnv *env, jobjectArray arr);

	struct reference {
		reference(JNIEnv *env, jobjectArray arr, jsize index);
		operator jobject();
		JNIEnv *const env;
		const jobjectArray arr;
		jsize index;

		reference& operator =(const String &value);
	};

	jobject operator [](size_t i) const;

	reference operator [](size_t i);
	bool operator !() const;

	jsize length() const;
	JNIEnv *const env;
	const jobjectArray arr;
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

template<>
struct FindClass<bool> {
	static const string name;
	using jobject = jboolean;
	using jarray = jbooleanArray;
	static jobject* (JNIEnv::*GetArrayElements)(jarray array, jboolean *isCopy);
	static void (JNIEnv::*ReleaseArrayElements)(jarray array, jobject *elems,
			jint mode);
};

template<>
struct FindClass<byte> {
	static const string name;
	using jobject = jbyte;
	using jarray = jbyteArray;
	static jobject* (JNIEnv::*GetArrayElements)(jarray array, jboolean *isCopy);
	static void (JNIEnv::*ReleaseArrayElements)(jarray array, jobject *elems,
			jint mode);
};

template<>
struct FindClass<short> {
	static const string name;
	using jobject = jshort;
	using jarray = jshortArray;
	static jobject* (JNIEnv::*GetArrayElements)(jarray array, jboolean *isCopy);
	static void (JNIEnv::*ReleaseArrayElements)(jarray array, jobject *elems,
			jint mode);
};

template<>
struct FindClass<int> {
	static const string name;
	using jobject = jint;
	using jarray = jintArray;

	static jobject* (JNIEnv::*GetArrayElements)(jarray array, jboolean *isCopy);
	static void (JNIEnv::*ReleaseArrayElements)(jarray array, jobject *elems,
			jint mode);
};

template<>
struct FindClass<long> {
	static const string name;
	using jobject = jlong;
	using jarray = jlongArray;
	static jobject* (JNIEnv::*GetArrayElements)(jarray array, jboolean *isCopy);
	static void (JNIEnv::*ReleaseArrayElements)(jarray array, jobject *elems,
			jint mode);
};

template<>
struct FindClass<float> {
	static const string name;
	using jobject = jfloat;
	using jarray = jfloatArray;
	static jobject* (JNIEnv::*GetArrayElements)(jarray array, jboolean *isCopy);
	static void (JNIEnv::*ReleaseArrayElements)(jarray array, jobject *elems,
			jint mode);
};

template<>
struct FindClass<double> {
	static const string name;
	using jobject = jdouble;
	using jarray = jdoubleArray;
	static jobject* (JNIEnv::*GetArrayElements)(jarray array, jboolean *isCopy);
	static void (JNIEnv::*ReleaseArrayElements)(jarray array, jobject *elems,
			jint mode);
};

template<typename _Ty>
struct FindClass<vector<_Ty>> {
	static const string name;
//	using jobject = jobject;
	using jarray = jobjectArray;
	static jobject* (JNIEnv::*GetArrayElements)(jarray array, jboolean *isCopy);
	static void (JNIEnv::*ReleaseArrayElements)(jarray array, jobject *elems,
			jint mode);
};

template<>
struct FindClass<String> {
	static const string name;
	using jobject = jstring;
	using jarray = jobjectArray;
	static jobject* (JNIEnv::*GetArrayElements)(jarray array, jboolean *isCopy);
	static void (JNIEnv::*ReleaseArrayElements)(jarray array, jobject *elems,
			jint mode);
};

template<typename _Ty>
const string FindClass<vector<_Ty>>::name = "[" + FindClass<_Ty>::name;

template<typename _Ty>
jobjectArray Object(JNIEnv *env, const vector<_Ty> &arr) {
	int sz = arr.size();

	auto jclass = env->FindClass(FindClass<_Ty>::name.data());
	auto obj = env->NewObjectArray(sz, jclass, nullptr);

	for (int k = 0; k < sz; k++) {
		auto local = Object(env, arr[k]);
		env->SetObjectArrayElement(obj, k, local);
		env->DeleteLocalRef(local);
	}

	env->DeleteLocalRef(jclass);
	return obj;
}

std::ostream& operator <<(std::ostream &cout, const JArray<int> &v);
