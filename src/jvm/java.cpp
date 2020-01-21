//"${JAVA_HOME}/include"
//"${JAVA_HOME}/include/win32"
#include <stdio.h>
#include <cstring>

#include "java.h"

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
FindClass<bool>::jobject* (JNIEnv::*FindClass<bool>::GetArrayElements)(
		jarray array, jboolean *isCopy) = &JNIEnv::GetBooleanArrayElements;
void (JNIEnv::*FindClass<bool>::ReleaseArrayElements)(jarray array,
		jobject *elems, jint mode) = &JNIEnv::ReleaseBooleanArrayElements;

const string FindClass<byte>::name = "B";
FindClass<byte>::jobject* (JNIEnv::*FindClass<byte>::GetArrayElements)(
		jarray array, jboolean *isCopy) = &JNIEnv::GetByteArrayElements;
void (JNIEnv::*FindClass<byte>::ReleaseArrayElements)(jarray array,
		jobject *elems, jint mode) = &JNIEnv::ReleaseByteArrayElements;

const string FindClass<short>::name = "S";
FindClass<short>::jobject* (JNIEnv::*FindClass<short>::GetArrayElements)(
		jarray array, jboolean *isCopy) = &JNIEnv::GetShortArrayElements;
void (JNIEnv::*FindClass<short>::ReleaseArrayElements)(jarray array,
		jobject *elems, jint mode) = &JNIEnv::ReleaseShortArrayElements;

const string FindClass<int>::name = "I";
FindClass<int>::jobject* (JNIEnv::*FindClass<int>::GetArrayElements)(
		jarray array, jboolean *isCopy) = &JNIEnv::GetIntArrayElements;
void (JNIEnv::*FindClass<int>::ReleaseArrayElements)(jarray array,
		jobject *elems, jint mode) = &JNIEnv::ReleaseIntArrayElements;

const string FindClass<long>::name = "J";
FindClass<long>::jobject* (JNIEnv::*FindClass<long>::GetArrayElements)(
		jarray array, jboolean *isCopy) = &JNIEnv::GetLongArrayElements;
void (JNIEnv::*FindClass<long>::ReleaseArrayElements)(jarray array,
		jobject *elems, jint mode) = &JNIEnv::ReleaseLongArrayElements;

const string FindClass<float>::name = "F";
FindClass<float>::jobject* (JNIEnv::*FindClass<float>::GetArrayElements)(
		jarray array, jboolean *isCopy) = &JNIEnv::GetFloatArrayElements;
void (JNIEnv::*FindClass<float>::ReleaseArrayElements)(jarray array,
		jobject *elems, jint mode) = &JNIEnv::ReleaseFloatArrayElements;

const string FindClass<double>::name = "D";
FindClass<double>::jobject* (JNIEnv::*FindClass<double>::GetArrayElements)(
		jarray array, jboolean *isCopy) = &JNIEnv::GetDoubleArrayElements;
void (JNIEnv::*FindClass<double>::ReleaseArrayElements)(jarray array,
		jobject *elems, jint mode) = &JNIEnv::ReleaseDoubleArrayElements;

const string FindClass<String>::name = "java/lang/String";

JArray<String>::JArray(JNIEnv *env, jobjectArray arr) :
		env(env), arr(arr) {
}

JArray<String>::reference::reference(JNIEnv *env, jobjectArray arr, jsize index) :
		env(env), arr(arr), index(index) {
}

JArray<String>::reference::operator jobject() {
	return env->GetObjectArrayElement(arr, index);
}

JArray<String>::reference& JArray<String>::reference::operator =(
		const String &value) {
	jobject val = Object(env, value);

	env->SetObjectArrayElement(arr, index, val);
	return *this;
}

jobject JArray<String>::operator [](size_t i) const {
	return env->GetObjectArrayElement(arr, i);
}

JArray<String>::reference JArray<String>::operator [](size_t i) {
	return reference(env, arr, i);
}

bool JArray<String>::operator !() const {
	return !length();
}

jsize JArray<String>::length() const {
	return env->GetArrayLength(arr);
}

std::ostream& operator <<(std::ostream &cout, const JArray<int> &v) {
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

void print_primitive_type_size() {
	cout << "sizeof(jchar) = " << sizeof(jchar) << endl;
	cout << "sizeof(jbyte) = " << sizeof(jbyte) << endl;
	cout << "sizeof(jboolean) = " << sizeof(jboolean) << endl;
	cout << "sizeof(jshort) = " << sizeof(jshort) << endl;
	cout << "sizeof(jint) = " << sizeof(jint) << endl;
	cout << "sizeof(jlong) = " << sizeof(jlong) << endl;
	cout << "sizeof(jfloat) = " << sizeof(jfloat) << endl;
	cout << "sizeof(jdouble) = " << sizeof(jdouble) << endl;

	cout << "sizeof(char) = " << sizeof(char) << endl;
	cout << "sizeof(wchar_t) = " << sizeof(wchar_t) << endl;
	cout << "sizeof(short) = " << sizeof(short) << endl;
	cout << "sizeof(int) = " << sizeof(int) << endl;
	cout << "sizeof(long) = " << sizeof(long) << endl;
	cout << "sizeof(long long) = " << sizeof(long long) << endl;

	cout << "sizeof(unsigned char) = " << sizeof(unsigned char) << endl;
	cout << "sizeof(unsigned wchar_t) = " << sizeof(unsigned wchar_t) << endl;
	cout << "sizeof(unsigned short) = " << sizeof(unsigned short) << endl;
	cout << "sizeof(unsigned int) = " << sizeof(unsigned int) << endl;
	cout << "sizeof(unsigned long) = " << sizeof(unsigned long) << endl;
	cout << "sizeof(unsigned long long) = " << sizeof(unsigned long long)
			<< endl;

	cout << "sizeof(float) = " << sizeof(float) << endl;
	cout << "sizeof(double) = " << sizeof(double) << endl;
	cout << "sizeof(byte) = " << sizeof(byte) << endl;
	cout << "sizeof(word) = " << sizeof(word) << endl;
	cout << "sizeof(dword) = " << sizeof(dword) << endl;
	cout << "sizeof(qword) = " << sizeof(qword) << endl;
	cout << "sizeof(void*) = " << sizeof(void*) << endl;
}
