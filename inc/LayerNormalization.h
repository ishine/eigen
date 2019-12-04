#pragma once

#include <Utility.h>

struct LayerNormalization {
	LayerNormalization(int num_attention_heads);
	int num_attention_heads;
	const static double epsilon;
	Vector gamma, beta;

	vector<Matrix>& operator()(vector<Matrix> &x);
};
