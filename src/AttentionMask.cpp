#include <AttentionMask.h>
//#include "lagacy.h"
#include <matrix.h>

vector<MatrixI>& AttentionMask::operator ()(const MatrixI& segment_ids,
		vector<MatrixI>& mask) {
	int batch_size = segment_ids.rows();
	mask.resize(batch_size);

	MatrixI vector = segment_ids * 2;
	vector -= 1;

	for (int i = 0; i < batch_size; ++i) {
		vector(i, 0) = 0;
	}

	for (int i = 0; i < batch_size; ++i) {
		const auto &row = vector.row(i);
		mask[i] = row.transpose() * row;
	}

	if (diagnal_attention){
		int sequence_length = segment_ids.cols();
		const auto &eye = MatrixI::Identity(sequence_length, sequence_length);
		for (int i = 0; i < batch_size; ++i) {
			mask[i] -= eye;
		}
	}

	for (int i = 0; i < batch_size; ++i) {
		mask[i] = not_equal(mask[i], 1);
	}

	return mask;
}

AttentionMask::AttentionMask(bool diagnal_attention) :
		diagnal_attention(diagnal_attention) {
}
