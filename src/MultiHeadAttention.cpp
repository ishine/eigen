#include <MultiHeadAttention.h>
//#include "lagacy.h"
#include <matrix.h>

vector<Matrix>& MultiHeadAttention::operator ()(vector<Matrix> &sequence,
		const vector<Matrix> &attention_matrix, const vector<MatrixI> &mask) {

	vector<Matrix> q, k, v;
	q = k = v = sequence;

	q = reshape_to_batches(q * Wq + bq);
	k = reshape_to_batches(k * Wk + bk);
	v = reshape_to_batches(v * Wv + bv);

	vector<Matrix> &y = scaled_dot_product_attention(q, k, v, attention_matrix,
			mask, sequence);

	y = reshape_from_batches(y, num_attention_heads);

	return y * Wo + bo;
}

vector<Matrix>& MultiHeadAttention::scaled_dot_product_attention(
		vector<Matrix> &query, vector<Matrix> &key, vector<Matrix> &value,
		vector<Matrix> &attention_mask, vector<Matrix> &mask) {
	vector<Matrix> e = K.batch_dot(query, key, axes = 2);

	e = e / math.sqrt(K.int_shape(query)[-1]);

	if (mask is not None) {
		if (not return_sequences) {
		mask = mask[:, 0];}
}

e -= (1.0 - K.cast(mask, K.floatx())) * -10000.0; //  # as is in tensorflow version;

if (attention_mask is not None)
	e -= attention_mask;

a = tf.nn.softmax(e); //  # as is in tensorflow version

return K.batch_dot(a, value);
}

MultiHeadAttention::MultiHeadAttention() {
}
