#include <Parallelize.h>
//#include "lagacy.h"
#include <matrix.h>

vector<MatrixI>& Parallelize::operator ()(vector<MatrixI> &mask) {
	int batch_size = mask.size();
	mask.resize(batch_size * num_attention_heads);
	for (int i = batch_size - 1; i >= 0; --i) {
		for (int j = 0; j < num_attention_heads; ++j)
			mask[i * num_attention_heads + j] = mask[i];
	}

	return mask;
}

Parallelize::Parallelize(int num_attention_heads) :
		num_attention_heads(num_attention_heads) {
}
