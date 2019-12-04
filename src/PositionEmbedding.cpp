#include <PositionEmbedding.h>
//#include "lagacy.h"
#include <matrix.h>

vector<Matrix>& PositionEmbedding::operator ()(vector<Matrix> &sequence,
		const VectorI &mid) {
	int batch_size = sequence.size();
	int seq_len = sequence[0].rows();

	for (int i = 0; i < batch_size; ++i) {
		Matrix former = embeddings.topRows(mid[i]);
		Matrix latter = embeddings.middleRows(1, seq_len - mid[i]);
		Matrix pos_embeddings;
		pos_embeddings << former, latter;
		sequence[i] += pos_embeddings;
	}

	return sequence;
}

vector<Matrix>& PositionEmbedding::operator ()(vector<Matrix> &sequence) {
	int batch_size = sequence.size();
	int seq_len = sequence[0].rows();

	Matrix pos_embeddings = embeddings.topRows(seq_len);

	for (int i = 0; i < batch_size; ++i) {
		sequence[i] += pos_embeddings;
	}

	return sequence;
}

PositionEmbedding::PositionEmbedding(int num_attention_heads, int input_dim,
		int output_dim) :
		num_attention_heads(num_attention_heads), input_dim(input_dim), output_dim(
				output_dim) {
}
