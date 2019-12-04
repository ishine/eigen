//"${JAVA_HOME}/include"
//"${JAVA_HOME}/include/win32"
#include <stdio.h>
#include <cstring>
#include "java.h"

extern "C" void JNICALL Java_com_util_Native_displayHelloWorld(JNIEnv *env,
		jobject obj) {
	cout << "Hello world!" << endl;
}

extern "C" jstring JNICALL Java_com_util_Native_reverse(JNIEnv *env,
		jobject obj, jstring str) {
	String s = JString(env, str);
	size_t length = s.size();
	for (size_t i = 0; i < length / 2; ++i) {
		std::swap(s[i], s[length - 1 - i]); // @suppress("Invalid arguments")
	}

	return Object(env, s);
}

#include "lagacy.h"

extern "C" jint JNICALL Java_com_util_Native_asm6args(JNIEnv *env,
		jobject obj, jint rcx, jint rdx, jint r8, jint r9, jint fifthArg,
		jint sixthArg) {
	return asm6args(rcx, rdx, r8, r9, fifthArg, sixthArg);
}

extern "C" jdouble JNICALL Java_com_util_Native_relu(JNIEnv *env,
		jobject obj, jdouble rcx) {
	return relu(rcx);
}

extern "C" jint JNICALL Java_com_util_Native_gcdint(JNIEnv *env,
		jobject obj, jint rcx, jint rdx) {
	return gcd_int(rcx, rdx);
}

extern "C" jlong JNICALL Java_com_util_Native_gcdlong(JNIEnv *env,
		jobject obj, jlong rcx, jlong rdx) {
	return gcd_long(rcx, rdx);
}

extern "C" jint JNICALL Java_com_util_Native_gcdinttemplate(JNIEnv *env,
		jobject obj, jint rcx, jint rdx) {
	return gcd(rcx, rdx);
}

extern "C" jlong JNICALL Java_com_util_Native_gcdlongtemplate(JNIEnv *env,
		jobject obj, jlong rcx, jlong rdx) {
	return gcd(rcx, rdx);
}


#include "Service.h"

extern "C" jint JNICALL Java_com_util_Native_service(JNIEnv *env, jobject obj,
		jstring str) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
	String s = JString(env, str);
	return Service::instance().predict(s);
}

extern "C" jobjectArray JNICALL Java_com_util_Native_SERVICE(JNIEnv *env, jobject obj,
		jstring str) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	String s = JString(env, str);
	cout << "s.size() = " << s.size() << endl;
	vector<vector<double>> debug;
	return Object(env, Service::INSTANCE().predict(s, debug));
}

#include "NERTaggerDict.h"

extern "C" jintArray JNICALL Java_com_util_Native_ner(JNIEnv *env, jobject obj,
		jstring _service, jstring _text, jintArray _code) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
	string service = CString(env, _service);
	String text = JString(env, _text);
	JInteger code(env, _code);
//	cout << "code from java = " << code << endl;

	vector<int> arr = code;
//	cout << "converted to C++ = " << arr << endl;

	NERTaggerDict::predict(service, text, arr);
	return Object(env, arr);
}

extern "C" jobjectArray JNICALL Java_com_util_Native_NER(JNIEnv *env,
		jobject obj, jstring _service, jstring _text, jintArray _code) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	string service = CString(env, _service);
	String text = JString(env, _text);
	JInteger code(env, _code);
	cout << "code from java = " << code << endl;

	vector<int> arr = code;
	cout << "converted to C++ = " << arr << endl;

	vector<vector<vector<double>>> debug;
	NERTaggerDict::_predict(service, text, arr, debug);
	return Object(env, debug);
}

jstring Object(JNIEnv *env, const string &s) {
	return env->NewStringUTF(s.data());
}

jstring Object(JNIEnv *env, const String &s) {
	return env->NewString(s.data(), s.size());
}

jintArray SetIntArrayRegion(JNIEnv *env, jsize size, const jint *array) {
	jintArray obj = env->NewIntArray(size);

	env->SetIntArrayRegion(obj, 0, size, array);

	return obj;
}

jintArray Object(JNIEnv *env, const vector<int> &s) {
	jsize size = s.size();

	if (sizeof(jint) == sizeof(int))
		return SetIntArrayRegion(env, size, (const jint*) s.data());

	vector<jint> v(s.begin(), s.end());

	return SetIntArrayRegion(env, size, (const jint*) v.data());

}

jfloatArray Object(JNIEnv *env, const vector<float> &s) {
	jsize size = s.size();

	const jfloat *array = s.data();

	jfloatArray obj = env->NewFloatArray(size);

	env->SetFloatArrayRegion(obj, 0, size, array);

	return obj;
}

jdoubleArray Object(JNIEnv *env, const vector<double> &s) {
	jsize size = s.size();

	const jdouble *array = s.data();

	jdoubleArray obj = env->NewDoubleArray(size);

	env->SetDoubleArrayRegion(obj, 0, size, array);

	return obj;
}

const string FindClass<bool>::name = "Z";
const string FindClass<byte>::name = "B";
const string FindClass<short>::name = "S";
const string FindClass<int>::name = "I";
const string FindClass<long>::name = "J";
const string FindClass<float>::name = "F";
const string FindClass<double>::name = "D";

std::ostream& operator <<(std::ostream &cout, const JInteger &v) {
	cout << '[';
	if (!v) {
		cout << v[0];
		for (jsize i = 1; i < v.length(); ++i) {
			cout << ", " << v[i];
		}
	}

	cout << ']';
	return cout;
}

