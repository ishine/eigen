#include <RevertMask.h>
//#include "lagacy.h"
#include <matrix.h>

vector<Matrix>& RevertMask::operator ()(vector<MatrixI> &attention_mask,
		vector<Matrix> &res) {
	int batch_size = attention_mask.size();
	res.resize(batch_size);

	for (int i = 0; i < batch_size; ++i) {
		res[i] = (1 - attention_mask[i]).cast<double>() * weight;
	}
	return res;
}

RevertMask::RevertMask(double weight) :
		weight(weight) {
}
