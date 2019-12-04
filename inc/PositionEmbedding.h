#pragma once

#include <Utility.h>

struct PositionEmbedding {
	PositionEmbedding(int num_attention_heads, int input_dim, int output_dim);

	Matrix embeddings;
	int num_attention_heads, input_dim, output_dim;

	vector<Matrix>& operator()(vector<Matrix> &sequence, const VectorI &mid);
	vector<Matrix>& operator()(vector<Matrix> &sequence);
};
