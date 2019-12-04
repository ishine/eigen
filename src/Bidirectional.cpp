#include "Bidirectional.h"

//Bidirectional::Bidirectional(RNN *forward, RNN *backward, merge_mode mode) :
//		forward(forward), backward(backward), mode(mode) {
//}

Matrix& Bidirectional::call_return_sequences(const Matrix &x, Matrix &ret) {
	Matrix forward;
	this->forward->call_return_sequences(x, forward);
	Matrix backward;
	this->backward->call_return_sequences_reverse(x, backward);

	switch (mode) {
	case sum:
		ret = forward + backward;
		break;
	case ave:
		ret = (forward + backward) / 2;
		break;
	case mul:
		ret = forward.array() * backward.array();
		break;
	case concat:
		ret << forward, backward;
		break;
	}
	return ret;
}

Vector& Bidirectional::call(const Matrix &x, Vector &ret) {
	Vector forward;
	this->forward->call(x, forward);
	Vector backward;
	this->backward->call_reverse(x, backward);

	switch (mode) {
	case sum:
		ret = forward + backward;
		break;
	case ave:
		ret = (forward + backward) / 2;
		break;
	case mul:
		ret = forward.cwiseProduct(backward);
		break;
	case concat:
		ret.resize(forward.cols() * 2);
		ret << forward, backward;
		break;
	}
	return ret;
}

Vector& Bidirectional::call(const Matrix &x, Vector &ret,
		vector<vector<double>> &arr) {
	Vector forward;
	forward = this->forward->call(x, forward, arr);
	arr.push_back(convert2vector(forward));

	Vector backward;
	backward = this->backward->call_reverse(x, backward, arr);
	arr.push_back(convert2vector(backward));

	switch (mode) {
	case sum:
		ret = forward + backward;
		break;
	case ave:
		ret = (forward + backward) / 2;
		break;
	case mul:
		ret = forward.cwiseProduct(backward);
		break;
	case concat:
		ret.resize(forward.cols() * 2);
		ret << forward, backward;
		break;
	}
	return ret;
}
