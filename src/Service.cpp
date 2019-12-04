#include "Service.h"

int Service::predict(const word *predict_text) {
	String text = predict_text;
	return predict(text);
}

vector<vector<double>>& Service::predict(const word *predict_text,
		vector<vector<double>> &arr) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	String text = predict_text;
	return predict(text, arr);
}

int Service::predict(String &predict_text) {
	Matrix embedding;
	this->embedding(predict_text, embedding, this->max_length);

	Vector x;
	gru.call(embedding, x);

	dense_mean(x);

	l2_normalize(x);

	dense_pred(x);

	int index;
	x.maxCoeff(&index);
	return index;
}

vector<vector<double>>& Service::predict(String &predict_text,
		vector<vector<double>> &arr) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	Matrix embedding;
	this->embedding(predict_text, embedding, this->max_length);

	arr.push_back(convert2vector(embedding, 0));
	arr.push_back(convert2vector(embedding, embedding.rows() - 1));

	Vector x;
	x = gru.call(embedding, x, arr);
	arr.push_back(convert2vector(x));

	x = dense_mean(x);
	arr.push_back(convert2vector(x));

	x = l2_normalize(x);
	arr.push_back(convert2vector(x));

	x = dense_pred(x);
	arr.push_back(convert2vector(x));
//	cout << arr[7] << endl;

	int index;
	x.maxCoeff(&index);

	return arr;
}

Service::Service(const string &binaryFilePath) :
		Service(BinaryReader(binaryFilePath)) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;

}

Service::Service(const BinaryReader &dis) :
		Service((BinaryReader&) dis) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
}

#include "lagacy.h"

Service::Service(BinaryReader &dis) :
		embedding(Embedding(dis)), gru(
				BidirectionalGRU(dis, merge_mode::concat)), dense_mean(
				DenseLayer(dis)), dense_pred(DenseLayer(dis, false)) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	dis.close();

	cout << "constants in assembly language:" << endl;
	cout << "zero = " << zero << endl;
	cout << "one = " << one << endl;
	cout << "one_fifth = " << one_fifth << endl;
	cout << "half =  " << half << endl;
}

Service& Service::instance() {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
	static BinaryReader dis(serviceBinary());
	static Service service(dis);

	return service;
}

Service& Service::INSTANCE() {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	static BinaryReader dis(serviceBinary());
	static Service service(dis);

	return service;
}

extern "C" int cpp_service(const word *text) {
	cout << "in " << __FUNCTION__ << endl;
	return Service::instance().predict(text);
}

extern "C" vector<vector<double>>& _cpp_service(const word *text,
		vector<vector<double>> &arr) {
	return Service::instance().predict(text, arr);
}

//reading .h5 with HDF5++
//https://portal.hdfgroup.org/display/support/HDF5%201.10.5
