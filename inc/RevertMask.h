#pragma once

#include <Utility.h>

struct RevertMask {
	RevertMask(double weight = 1e10);
	double weight;
	vector<Matrix>& operator()(vector<MatrixI> &mask, vector<Matrix> &res);
};
