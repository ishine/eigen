#pragma once
#include "RNN.h"
enum merge_mode {
	sum, mul, ave, concat
};

struct Bidirectional {
	RNN::object forward, backward;

	merge_mode mode;

	Matrix& call_return_sequences(const Matrix &x, Matrix &ret);

	Vector& call(const Matrix &x, Vector &ret);
	Vector& call(const Matrix &x, Vector &ret, vector<vector<double>> &arr);
//private:
//	Bidirectional(RNN *forward, RNN *backward, merge_mode mode);
};
