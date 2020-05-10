//"${JAVA_HOME}/include"
//"${JAVA_HOME}/include/win32"
#include <stdio.h>
#include <cstring>

#include "java.h"
#include <iostream>

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
//	__cout(__PRETTY_FUNCTION__)
//	cout << "s = " << s << endl;
//	jsize size = s.size();

	static_assert (sizeof(jint) == sizeof(int), "jint and int must have same sizes");
//	if (sizeof(jint) == sizeof(int))
	return SetIntArrayRegion(env, s.size(), (const jint*) s.data());

//	vector<jint> v(s.begin(), s.end());

//	return SetIntArrayRegion(env, size, (const jint*) v.data());

}

jintArray Object(JNIEnv *env, const VectorI &s) {
	jsize size = s.size();

	assert(sizeof(jint) == sizeof(int));
	if (sizeof(jint) == sizeof(int))
		return SetIntArrayRegion(env, size, (const jint*) s.data());

	auto begin = s.data();
	vector<jint> v(begin, begin + s.size());

	return SetIntArrayRegion(env, size, (const jint*) v.data());

}

jobjectArray Object(JNIEnv *env, const Matrix &A) {
	int n = A.rows();
	int m = A.cols();
	vector<vector<double>> matrix(n);

	for (int i = 0; i < n; ++i) {
		matrix[i].resize(m);
		for (int j = 0; j < m; ++j) {
			matrix[i][j] = A(i, j);
		}
	}
	return Object(env, matrix);
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

const char *FindClass<bool>::name = "Z";
const char *FindClass<byte>::name = "B";
const char *FindClass<char16_t>::name = "C";
const char *FindClass<short>::name = "S";
const char *FindClass<int>::name = "I";
const char *FindClass<long>::name = "J";
const char *FindClass<float>::name = "F";
const char *FindClass<double>::name = "D";
const char *FindClass<String>::name = "java/lang/String";
const char *FindClass<string>::name = "java/lang/String";
//const char *FindClass<vector<String>>::name = "[Ljava/lang/String;";
//const string FindClass<vector<vector<String>>>::name = "[[Ljava/lang/String;";

std::ostream& operator <<(std::ostream &cout, const JArray<int> &v) {
	cout << '[';
	if (!v) {
		cout << v[0];
		for (jsize i = 1; i < v.length; ++i) {
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

//https://www.cnblogs.com/nicholas_f/archive/2010/11/30/1892124.html
