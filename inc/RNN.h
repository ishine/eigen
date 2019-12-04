#pragma once
#include "Utility.h"

struct RNN {
	typedef ::object<RNN> object;

	virtual ~RNN() {
	}

	virtual Vector& call(const Matrix &x, Vector &ret) {
		return ret;
	}

	virtual Vector& call(const Matrix &x, Vector &ret,
			vector<vector<double>> &arr) {
		return ret;
	}

	virtual Vector& call_reverse(const Matrix &x, Vector &ret) {
		return ret;
	}

	virtual Vector& call_reverse(const Matrix &x, Vector &ret,
			vector<vector<double>> &arr) {
		return ret;
	}

	virtual Matrix& call_return_sequences(const Matrix &x, Matrix &ret) {
		return ret;
	}

	virtual Matrix& call_return_sequences_reverse(const Matrix &x,
			Matrix &ret) {
		return ret;
	}

	virtual vector<vector<vector<double>>> & weight(
			vector<vector<vector<double>>> &arr) {
		return arr;
	}
};
