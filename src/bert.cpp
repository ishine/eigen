#include "bert.h"
#include <matrix.h>
#include "lagacy.h"
Matrix& revert_mask(const MatrixI &mask, double weight) {
	static Matrix out;
	out = mask.cast<double>() * -weight;
	if (weight >= 0)
		out += weight;

	return out;
}

vector<Vector>& revert_mask(const vector<VectorI> &mask, double weight) {
	auto &out = -weight * mask;
	if (weight >= 0)
		out += weight;

	return out;
}

vector<MatrixI>& CrossAttentionMask::operator ()(
		const vector<VectorI> &segment_ids) {
	int batch_size = segment_ids.size();
	static vector<MatrixI> mask;

	mask.resize(batch_size);

	auto &vec = segment_ids * 2;
	vec -= 1;

	for (int k = 0; k < batch_size; ++k) {
		vec[k](0) = 0;
	}

	for (int k = 0; k < batch_size; ++k) {
		const auto &row = vec[k];
		mask[k] = row.transpose() * row;
	}

	if (diagnal_attention) {
		int sequence_length = segment_ids[0].size();
		const auto &eye = MatrixI::Identity(sequence_length, sequence_length);
		for (int k = 0; k < batch_size; ++k) {
			mask[k] -= eye;
		}
	}

	for (int k = 0; k < batch_size; ++k) {
		mask[k] = mask[k] != 1;
	}

	if (num_attention_heads)
		mask = parallelize(mask, num_attention_heads);
	return mask;
}

CrossAttentionMask::CrossAttentionMask(int num_attention_heads,
		bool diagnal_attention) :
		num_attention_heads(num_attention_heads), diagnal_attention(
				diagnal_attention) {
}

Vector& FeedForward::operator()(const Vector &x, Vector &ret) {
	ret = x * W1 + b1;
	return ret;
}

vector<Vector>& FeedForward::operator()(const vector<Vector> &x) {
	auto &y = x * W1;
	if (b1.data())
		y += b1;

	y = gelu(y);

	y *= W2;
	if (b2.data())
		y += b2;
	return y;
}

Tensor& FeedForward::operator()(const Tensor &x) {
	auto &y = x * W1;
	if (b1.data())
		y += b1;

	y = gelu(y);

	y *= W2;
	if (b2.data())
		y += b2;
	return y;
}

Matrix& FeedForward::operator()(Matrix &x) {

	x *= W1;
	if (b1.data())
		x += b1;
	return x;
}

FeedForward::FeedForward() {

}

FeedForward::FeedForward(BinaryReader &dis, bool use_bias) {
	dis.read(W1);
	if (use_bias) {
		dis.read(b1);
	}

	dis.read(W2);
	if (use_bias) {
		dis.read(b2);
	}
}

Tensor& LayerNormalization::operator ()(Tensor &x) {
	auto &deviation = x - mean(x);

	static Tensor deviation_copy;
	deviation_copy = deviation;

	return deviation / sqrt(mean(square(deviation_copy)) + epsilon) * gamma
			+ beta;
}

vector<Vector>& LayerNormalization::operator()(vector<Vector> &x) {
	auto &deviation = x - mean(x);

	static vector<Vector> deviation_copy;
	deviation_copy = deviation;

	return deviation / sqrt(mean(square(deviation_copy)) + epsilon) * gamma
			+ beta;
}

LayerNormalization::LayerNormalization() {
}

LayerNormalization::LayerNormalization(BinaryReader &dis) {
	dis.read(gamma);
	dis.read(beta);
}

const double LayerNormalization::epsilon = 1e-12;

vector<int>& MidIndex::operator()(const vector<VectorI> &input_ids) {
	int batch_size = input_ids.size();
	static vector<int> res;
	res.resize(batch_size);
	int seq_len = input_ids[0].size();

	for (int k = 0; k < batch_size; ++k) {
		for (int i = 0; i < seq_len; ++i) {
			if (input_ids[k](i) == SEP) {
				res[k] = i + 1;
				break;
			}
		}
	}
	return res;
}

MidIndex::MidIndex(int SEP) {
	this->SEP = SEP;
	cout << "in " << __PRETTY_FUNCTION__ << endl;
}

Tensor& MultiHeadAttention::operator ()(const Tensor &sequence,
		const Tensor &attention_matrix, const vector<Vector> &mask) {

	Tensor q, k, v;
	q = k = v = sequence;

	q = reshape_to_batches(q * Wq + bq);
	k = reshape_to_batches(k * Wk + bk);
	v = reshape_to_batches(v * Wv + bv);

	Tensor &y = scaled_dot_product_attention(q, k, v, attention_matrix, mask);

	y = reshape_from_batches(y);

	return y * Wo + bo;
}

vector<Vector>& MultiHeadAttention::operator ()(const Tensor &sequence,
		const vector<Vector> &mask, vector<Vector> &y) {

	Tensor k, v;
	k = v = sequence;

	vector<Vector> &q = extract(sequence, 0, y);

	q = reshape_to_batches(q * Wq + bq);
	k = reshape_to_batches(k * Wk + bk);
	v = reshape_to_batches(v * Wv + bv);

	y = scaled_dot_product_attention(q, k, v, mask);

	y = reshape_from_batches(y);
	y *= Wo;
	y += bo;
	return y;
}

Tensor& MultiHeadAttention::scaled_dot_product_attention(Tensor &query,
		Tensor &key, Tensor &value, const Tensor &attention_mask,
		const vector<Vector> &mask) {
	Tensor &e = batch_dot(query, key, true);

	e /= sqrt(query[0].cols());
	e -= mask;
	e -= attention_mask;

	Tensor &a = softmax(e);
	return batch_dot(a, value);
}

vector<Vector>& MultiHeadAttention::scaled_dot_product_attention(
		vector<Vector> &query, const Tensor &key, const Tensor &value,
		const vector<Vector> &mask) {
	vector<Vector> &e = batch_dot(query, key, true);

	e /= sqrt(query[0].cols());

	e -= mask;

	return batch_dot(softmax(e), value);
}

Tensor& MultiHeadAttention::reshape_to_batches(Tensor &x) {
	int batch_size = x.size();
	int hidden_size = x[0].cols();

	int size_per_head = hidden_size / num_attention_heads;

	x.resize(batch_size * num_attention_heads);
	for (int i = batch_size - 1; i >= 0; --i) {
		for (int j = 0; j < num_attention_heads; ++j)
			x[i * num_attention_heads + j] = x[i].middleCols(j * size_per_head,
					size_per_head);
	}

	return x;
}

vector<Vector>& MultiHeadAttention::reshape_to_batches(vector<Vector> &x) {
	int batch_size = x.size();
	int hidden_size = x[0].cols();

	int size_per_head = hidden_size / num_attention_heads;

	x.resize(batch_size * num_attention_heads);
	for (int i = batch_size - 1; i >= 0; --i) {
		for (int j = 0; j < num_attention_heads; ++j)
			x[i * num_attention_heads + j] = x[i].middleCols(j * size_per_head,
					size_per_head);
	}

	return x;
}

Tensor& MultiHeadAttention::reshape_from_batches(Tensor &x) {
	int batch_size = x.size() / num_attention_heads;
	int size_per_head = x[0].cols();
	int hidden_size = size_per_head * num_attention_heads;
	int seq_length = x[0].rows();

	for (int k = 0; k < batch_size; ++k) {
		Matrix res = Matrix::Zero(seq_length, hidden_size);
		for (int j = 0; j < num_attention_heads; ++j)
			res.middleCols(j * size_per_head, size_per_head) = x[k
					* num_attention_heads + j];
		x[k] = res;
	}

	x.resize(batch_size);
	return x;
}

vector<Vector>& MultiHeadAttention::reshape_from_batches(vector<Vector> &x) {
	int batch_size = x.size() / num_attention_heads;
	int size_per_head = x[0].cols();
	int hidden_size = size_per_head * num_attention_heads;

	for (int k = 0; k < batch_size; ++k) {
		Matrix res = Vector::Zero(hidden_size);
		for (int j = 0; j < num_attention_heads; ++j)
			res.middleCols(j * size_per_head, size_per_head) = x[k
					* num_attention_heads + j];
		x[k] = res;
	}

	x.resize(batch_size);
	return x;

}

MultiHeadAttention::MultiHeadAttention() {

}

MultiHeadAttention::MultiHeadAttention(BinaryReader &dis) {
	dis.read(Wq);
	dis.read(bq);

	dis.read(Wk);
	dis.read(bk);

	dis.read(Wv);
	dis.read(bv);

	dis.read(Wo);
	dis.read(bo);
}

Tensor& PositionEmbedding::operator ()(Tensor &sequence,
		const vector<int> &mid) {
	int batch_size = sequence.size();
	int seq_len = sequence[0].rows();

	for (int k = 0; k < batch_size; ++k) {
		Matrix former = embeddings.topRows(mid[k]);
		Matrix latter = embeddings.middleRows(1, seq_len - mid[k]);
		Matrix pos_embeddings;
		pos_embeddings << former, latter;
		sequence[k] += pos_embeddings;
	}

	return sequence;
}

Tensor& PositionEmbedding::operator ()(Tensor &sequence) {
	int batch_size = sequence.size();
	int seq_len = sequence[0].rows();

	Matrix pos_embeddings = embeddings.topRows(seq_len);

	for (int k = 0; k < batch_size; ++k) {
		sequence[k] += pos_embeddings;
	}

	return sequence;
}

vector<Vector>& PositionEmbedding::compute_mask(vector<VectorI> &inputToken) {
	vector<VectorI> &mask = inputToken != 0;

	parallelize(mask, num_attention_heads);
	return revert_mask(mask, 10000.0);
}

PositionEmbedding::PositionEmbedding(BinaryReader &dis, int num_attention_heads) :
		num_attention_heads(num_attention_heads) {
	dis.read(embeddings);
}

Tensor& RevertMask::operator ()(const vector<MatrixI> &attention_mask) {
	static Tensor res;
	return (*this)(attention_mask, res);
}

Tensor& RevertMask::operator ()(const vector<MatrixI> &attention_mask, Tensor &res) {

	int batch_size = attention_mask.size();

	res.resize(batch_size);

	for (int k = 0; k < batch_size; ++k) {
		res[k] = revert_mask(attention_mask[k], weight(step));
	}

	if (++step == weight.size())
		step = 0;

	return res;
}

RevertMask::RevertMask(double cross_attention) :
		cross_attention(cross_attention) {
	step = 0;
}

vector<VectorI>& SegmentInput::operator ()(const vector<VectorI> &inputToken,
		vector<int> &inputMid) {
	int batch_size = inputToken.size();
	int length = inputToken[0].size();
	static vector<VectorI> inputSegment;
	inputSegment.resize(batch_size);

	for (int k = 0; k < batch_size; ++k) {
		int mid = inputMid[k];
		stosd(&inputSegment[k](0), 0, mid);
		stosd(&inputSegment[k](mid), 1, length - mid);
	}
	return inputSegment;
}

BertEmbedding::BertEmbedding(BinaryReader &dis, int num_attention_heads,
		bool symmetric_positional_embedding,
		bool factorization_on_word_embedding_only) :
		factorization_on_word_embedding_only(
				factorization_on_word_embedding_only), symmetric_positional_embedding(
				symmetric_positional_embedding), WordEmbedding(dis), SegmentEmbedding(
				dis), PositionEmbedding(dis, num_attention_heads), LayerNormalization(
				dis), Dense(dis) {
	this->embed_dim = WordEmbedding.wEmbedding.cols();
	this->hidden_size = SegmentEmbedding.wEmbedding.cols();

	if (!this->factorization()) {
		dis.read(this->Dense.bDense);
	}
}

bool BertEmbedding::factorization(bool word_embedding_only) {
	if (word_embedding_only)
		return hidden_size != embed_dim && factorization_on_word_embedding_only;
	return hidden_size != embed_dim && !factorization_on_word_embedding_only;
}

Tensor& BertEmbedding::operator ()(vector<VectorI> &inputToken,
		const vector<int> &inputMid, const vector<VectorI> &inputSegment,
		vector<Vector> &mask) {
	auto &embeddings = WordEmbedding(inputToken);
	mask = this->PositionEmbedding.compute_mask(inputToken);

	if (factorization()) {
		embeddings = Dense(embeddings);
	}

	auto &segment_layer = SegmentEmbedding(inputSegment);
	embeddings += segment_layer;
	auto &embed_layer =
			symmetric_positional_embedding ?
					PositionEmbedding(embeddings, inputMid) :
					PositionEmbedding(embeddings);

	embed_layer = LayerNormalization(embed_layer);

	if (factorization(false)) {
		embeddings = Dense(embeddings);
	}

	return embeddings;
}

Encoder::Encoder(BinaryReader &dis) :
		MultiHeadAttention(dis), MultiHeadAttentionNorm(dis), FeedForward(dis), FeedForwardNorm(
				dis) {
}

Encoder::Encoder() {
}

Tensor& Encoder::wrap_attention(Tensor &input_layer,
		const Tensor &attention_matrix, const vector<Vector> &mask) {
	return MultiHeadAttentionNorm(
			input_layer
					+ MultiHeadAttention(input_layer, attention_matrix, mask));
}

vector<Vector>& Encoder::wrap_attention(Tensor &input_layer,
		const vector<Vector> &mask, vector<Vector> &y) {
	return MultiHeadAttentionNorm(
			MultiHeadAttention(input_layer, mask, y) + extract(input_layer, 0));
}

Tensor& Encoder::wrap_feedforward(Tensor &input_layer) {
	return FeedForwardNorm(input_layer + FeedForward(input_layer));
}

vector<Vector>& Encoder::wrap_feedforward(vector<Vector> &input_layer) {
	return FeedForwardNorm(input_layer + FeedForward(input_layer));
}

Tensor& Encoder::operator ()(Tensor &input_layer,
		const Tensor &attention_matrix, const vector<Vector> &mask) {
	Tensor &inputs = wrap_attention(input_layer, attention_matrix, mask);
	return wrap_feedforward(inputs);
}

vector<Vector>& Encoder::operator ()(Tensor &input_layer,
		const vector<Vector> &mask, vector<Vector> &y) {
	auto &inputs = wrap_attention(input_layer, mask, y);
	return wrap_feedforward(inputs);
}

Transformer::Transformer(BinaryReader &dis, bool cross_layer_parameter_sharing,
		int num_hidden_layers) :
		num_hidden_layers(num_hidden_layers) {
	if (cross_layer_parameter_sharing) {
		encoder = new Encoder(dis);
	} else {
		encoder = new Encoder[num_hidden_layers];

		for (int i = 0; i < num_hidden_layers; ++i) {
			new (&encoder[i]) Encoder(dis);
		}
	}
}

Tensor& Transformer::operator ()(Tensor &input_layer,
		const Tensor &attention_matrix, const vector<Vector> &mask) {
	auto &last_layer = input_layer;
	for (int i = 0; i < num_hidden_layers; ++i) {

		last_layer = (*this)[i](last_layer, attention_matrix, mask);
	}
	return last_layer;
}

Encoder& Transformer::operator [](int i) {
	return encoder.color ? encoder[i] : *encoder;
}

vector<Vector>& Transformer::operator ()(Tensor &input_layer,
		const vector<MatrixI> &attention_matrix, RevertMask &fn,
		const vector<Vector> &mask, vector<Vector> &y) {
	auto &last_layer = input_layer;
	for (int i = 0; i < num_hidden_layers; ++i) {
		if (i == num_hidden_layers - 1) {
			y = (*this)[i](last_layer, mask, y);
		} else
			last_layer = (*this)[i](last_layer, fn(attention_matrix), mask);
	}
	return y;
}

Paraphrase::Paraphrase(BinaryReader &dis, int num_attention_heads,
		bool symmetric_positional_embedding,
		bool factorization_on_word_embedding_only,
		bool cross_layer_parameter_sharing, int num_hidden_layers,
		double cross_attention) :
		RevertMask(cross_attention), BertEmbedding(dis, num_attention_heads,
				symmetric_positional_embedding,
				factorization_on_word_embedding_only), Transformer(dis,
				cross_layer_parameter_sharing, num_hidden_layers), poolerDense(
				dis), similarityDense(dis) {
}

vector<double>& Paraphrase::operator ()(vector<VectorI> &input_ids) {
	auto &inputMid = MidIndex(input_ids);

	auto &inputSegment = SegmentInput(input_ids, inputMid);

	auto &matrixAttention = CrossAttentionMask(inputSegment);

	static vector<Vector> mask;
	auto &embed_layer = BertEmbedding(input_ids, inputMid, inputSegment, mask);

	vector<Vector> clsEmbedding;
	Transformer(embed_layer, matrixAttention, RevertMask, mask, clsEmbedding);
	auto &sent = poolerDense(clsEmbedding);
	static vector<double> similarity;
	int batch_size = sent.size();
	similarity.resize(batch_size);
	sent = similarityDense(sent);
	for (int k = 0; k < batch_size; ++k) {
		similarity[k] = sent[k](0);
	}
	return similarity;
}
