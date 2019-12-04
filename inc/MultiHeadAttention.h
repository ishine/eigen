#pragma once

#include <Utility.h>

struct MultiHeadAttention {
	MultiHeadAttention();
	vector<Matrix>& operator()(vector<Matrix> &sequence,
			const vector<Matrix> &attention_matrix,
			const vector<MatrixI> &mask);
	Matrix Wq, Wk, Wv, Wo;
	Vector bq, bk, bv, bo;

	int num_attention_heads;

	vector<Matrix>& scaled_dot_product_attention(vector<Matrix>&);
	vector<Matrix>& reshape_to_batches(vector<Matrix>&);
	vector<Matrix>& reshape_from_batches(vector<Matrix>&);
};
