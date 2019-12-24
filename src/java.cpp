//"${JAVA_HOME}/include"
//"${JAVA_HOME}/include/win32"
#include "utility.h"
#include <stdio.h>
#include <cstring>

#include "java.h"

#include "bert.h"
#include "NERTagger.h"
#include "lagacy.h"
#include <classification.h>

extern "C" {
void JNICALL Java_com_util_Native_displayHelloWorld(JNIEnv *env, jobject obj) {
	cout << "Hello world!" << endl;
}

int JNICALL Java_com_util_Native_main(JNIEnv *env, jobject obj) {
	int main(int argc, char **argv);
	return main(0, 0);
}

jstring JNICALL Java_com_util_Native_reverse(JNIEnv *env, jobject obj,
		jstring str) {
	String s = JString(env, str);
	size_t length = s.size();
	for (size_t i = 0; i < length / 2; ++i) {
		std::swap(s[i], s[length - 1 - i]); // @suppress("Invalid arguments")
	}

	return Object(env, s);
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
	JInteger code(env, _code);
//	cout << "code from java = " << code << endl;

	VectorI arr = code;
//	cout << "converted to C++ = " << arr << endl;

	NERTaggerDict::predict(service, text, arr);
	return Object(env, arr);
}

jobjectArray JNICALL Java_com_util_Native_NER(JNIEnv *env, jobject obj,
		jstring _service, jstring _text, jintArray _code) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	string service = CString(env, _service);
	String text = JString(env, _text);
	JInteger code(env, _code);
	cout << "code from java = " << code << endl;

	VectorI arr = code;
	cout << "converted to C++ = " << arr << endl;

	vector<vector<vector<double>>> debug;
	NERTaggerDict::_predict(service, text, arr, debug);
	return Object(env, debug);
}
}

jstring Object(JNIEnv *env, const string &s) {
	return env->NewStringUTF(s.data());
}

jstring Object(JNIEnv *env, const String &s) {
	static_assert(sizeof (jchar) == sizeof (char16_t), "jchar and char16_t must have same sizes");
	return env->NewString((const jchar*) s.data(), s.size());
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

jintArray Object(JNIEnv *env, const VectorI &s) {
	jsize size = s.size();

	if (sizeof(jint) == sizeof(int))
		return SetIntArrayRegion(env, size, (const jint*) s.data());

	auto begin = s.data();
	vector<jint> v(begin, begin + s.size());

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

