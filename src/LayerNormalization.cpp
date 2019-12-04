#include <LayerNormalization.h>
//#include "lagacy.h"
#include <matrix.h>

vector<Matrix>& LayerNormalization::operator ()(vector<Matrix> &x) {
	auto &deviation = x - mean(x);

	vector<Matrix> deviation_copy = deviation;

	return deviation / sqrt(mean(square(deviation_copy)) + epsilon) * gamma + beta;
}

LayerNormalization::LayerNormalization(int num_attention_heads) :
		num_attention_heads(num_attention_heads) {
}


const double LayerNormalization::epsilon = 1e-12;
