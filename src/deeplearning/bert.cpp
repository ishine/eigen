#include "bert.h"
#include "matrix.h"
#include "../std/lagacy.h"

Matrix revert_mask(const MatrixI &mask, double weight) {
	Matrix out;
	out = mask.cast<double>() * -weight;
	if (weight >= 0)
		out += weight;

	return out;
}

vector<Vector> revert_mask(const vector<VectorI> &mask, double weight) {
	auto out = -weight * mask;
	if (weight >= 0)
		out += weight;

	return out;
}

vector<MatrixI> CrossAttentionMask::operator ()(
		const vector<VectorI> &segment_ids) {
	int batch_size = segment_ids.size();
	vector<MatrixI> mask;

	mask.resize(batch_size);

	auto vec = segment_ids * 2;
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
	return ret = x * W1 + b1;
}

vector<Vector> FeedForward::operator()(const vector<Vector> &x) {
	auto y = x * W1;
	if (b1.data())
		y += b1;

	y = activation(y);

	y *= W2;
	if (b2.data())
		y += b2;
	return y;
}

Vector FeedForward::operator()(const Vector &x) {
	Vector y;
	y = x * W1;
	if (b1.data())
		y += b1;

	y = activation(y);

	y *= W2;
	if (b2.data())
		y += b2;
	return y;
}

Tensor FeedForward::operator()(const Tensor &x) {
	auto y = x * W1;
	if (b1.data())
		y += b1;

	y = activation(y);

	y *= W2;
	if (b2.data())
		y += b2;
	return y;
}

Matrix FeedForward::operator()(const Matrix &x) {
	Matrix y;
	y = x * W1;
	if (b1.data())
		add(y, b1);

	y = activation(y);

	y *= W2;
	if (b2.data())
		add(y, b2);
	return y;
}

FeedForward::FeedForward() {

}

FeedForward::FeedForward(KerasReader &dis, bool use_bias) {
	dis >> W1;
	if (use_bias) {
		dis >> b1;
	}

	dis >> W2;
	if (use_bias) {
		dis >> b2;
	}
}

FeedForward::FeedForward(KerasReader &dis, Activation activation) {
	dis >> W1;
	dis >> b1;

	dis >> W2;

	dis >> b2;

	this->activation = activation;
}

Tensor& LayerNormalization::operator ()(Tensor &x) {
	auto &deviation = x - mean(x);

	Tensor deviation_copy;
	deviation_copy = deviation;

	auto mean_x = mean(square(deviation_copy));
	return deviation / sqrt(mean_x + epsilon) * gamma + beta;
}

Matrix& LayerNormalization::operator ()(Matrix &x) {
	x = subt(x, mean(x));

	Matrix &deviation = x;

	Matrix deviation_copy;
	deviation_copy = deviation;

	auto mean_square = mean(square(deviation_copy));
	mean_square += epsilon;
	divt(deviation, sqrt(mean_square));
	mul(deviation, gamma);
	add(deviation, beta);
	return deviation;
}

vector<Vector>& LayerNormalization::operator()(vector<Vector> &x) {
	auto &deviation = x - mean(x);

	vector<Vector> deviation_copy;
	deviation_copy = deviation;

	auto mean_x = mean(square(deviation_copy));
	return deviation / sqrt(mean_x + epsilon) * gamma + beta;
}

Vector& LayerNormalization::operator()(Vector &x) {
	x -= x.mean();
	auto &deviation = x;

	Vector deviation_copy;
	deviation_copy = deviation;
	deviation /= sqrt(square(deviation_copy).mean() + epsilon);
	mul(deviation, gamma);
	deviation += beta;
	return deviation;
}

LayerNormalization::LayerNormalization() {
}

LayerNormalization::LayerNormalization(KerasReader &dis) {
	__cout(__PRETTY_FUNCTION__)

	dis >> gamma;
	dis >> beta;
}

const double LayerNormalization::epsilon = 1e-12;

vector<int> MidIndex::operator()(const vector<VectorI> &input_ids) {
	int batch_size = input_ids.size();
	vector<int> res;
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

int MidIndex::operator()(const VectorI &input_ids) {
	int seq_len = input_ids.size();

	for (int i = 0; i < seq_len; ++i) {
		if (input_ids(i) == SEP) {
			return i + 1;
		}
	}
	return -1;
}

MidIndex::MidIndex(int SEP) {
	this->SEP = SEP;
	__cout(__PRETTY_FUNCTION__)

}

Tensor MultiHeadAttention::operator ()(const Tensor &sequence,
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

Tensor MultiHeadAttention::operator ()(const Tensor &sequence,
		const vector<Vector> &mask) {

	Tensor q, k, v;
	q = k = v = sequence;

	q = reshape_to_batches(q * Wq + bq);
	k = reshape_to_batches(k * Wk + bk);
	v = reshape_to_batches(v * Wv + bv);

	Tensor &y = scaled_dot_product_attention(q, k, v, mask);

	y = reshape_from_batches(y);

	return y * Wo + bo;
}

Matrix MultiHeadAttention::operator ()(const Matrix &sequence) {

	Matrix q, k, v;
	q = k = v = sequence;

	Tensor _q = reshape_to_batches(add(q *= Wq, bq));
	Tensor _k = reshape_to_batches(add(k *= Wk, bk));
	Tensor _v = reshape_to_batches(add(v *= Wv, bv));

	Tensor &y = scaled_dot_product_attention(_q, _k, _v);

	y = reshape_from_batches(y);
	assert(y.size() == 1);

	Matrix res;
	res = y[0];
	res *= Wo;
	add(res, bo);
	return res;
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

Vector& MultiHeadAttention::operator ()(const Matrix &sequence, Vector &y) {

	Matrix k, v;
	k = v = sequence;

	y = sequence.row(0);
	Vector &q = y;

	vector<Vector> _q = reshape_to_batches((q *= Wq) += bq);
	Tensor _k = reshape_to_batches(add(k *= Wk, bk));
	Tensor _v = reshape_to_batches(add(v *= Wv, bv));

	vector<Vector> &_y = reshape_from_batches(
			scaled_dot_product_attention(_q, _k, _v));

	assert(_y.size() == 1);

	y = _y[0];

	y *= Wo;
	y += bo;
	return y;
}

Tensor& MultiHeadAttention::scaled_dot_product_attention(Tensor &query,
		Tensor &key, Tensor &value, const Tensor &attention_mask,
		const vector<Vector> &mask) {
	Tensor &e = batch_dot(query, key, true);

	e /= sqrt(key[0].cols());
	e -= mask;
	e -= attention_mask;

	Tensor &a = softmax(e);
	return batch_dot(a, value);
}

Tensor& MultiHeadAttention::scaled_dot_product_attention(Tensor &query,
		Tensor &key, Tensor &value, const vector<Vector> &mask) {
	Tensor &e = batch_dot(query, key, true);

	e /= sqrt(key[0].cols());
	e -= mask;

	Tensor &a = softmax(e);
	return batch_dot(a, value);
}

Tensor& MultiHeadAttention::scaled_dot_product_attention(Tensor &query,
		Tensor &key, Tensor &value) {
	Tensor &e = batch_dot(query, key, true);

	e /= sqrt(key[0].cols());

	Tensor &a = softmax(e);
	return batch_dot(a, value);
}

vector<Vector>& MultiHeadAttention::scaled_dot_product_attention(
		vector<Vector> &query, const Tensor &key, const Tensor &value,
		const vector<Vector> &mask) {
	vector<Vector> &e = batch_dot(query, key, true);

	e /= sqrt(key[0].cols());

	e -= mask;

	return batch_dot(softmax(e), value);
}

vector<Vector>& MultiHeadAttention::scaled_dot_product_attention(
		vector<Vector> &query, const Tensor &key, const Tensor &value) {
	vector<Vector> &e = batch_dot(query, key, true);

	e /= sqrt(key[0].cols());

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

Tensor MultiHeadAttention::reshape_to_batches(Matrix &x) {
	Tensor _x;
	_x.resize(num_attention_heads);
//	_x[0] = x;

	int hidden_size = x.cols();

	int size_per_head = hidden_size / num_attention_heads;

	for (int j = 0; j < num_attention_heads; ++j)
		_x[j] = x.middleCols(j * size_per_head, size_per_head);

	return _x;
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

vector<Vector> MultiHeadAttention::reshape_to_batches(Vector &x) {
	vector<Vector> _x;
	_x.resize(num_attention_heads);
//	_x[0] = x;

	int hidden_size = x.cols();
	int size_per_head = hidden_size / num_attention_heads;

	for (int j = 0; j < num_attention_heads; ++j)
		_x[j] = x.middleCols(j * size_per_head, size_per_head);

	return _x;
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

MultiHeadAttention::MultiHeadAttention() { // @suppress("Class members should be properly initialized")
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

MultiHeadAttention::MultiHeadAttention(KerasReader &dis,
		int num_attention_heads) :
		num_attention_heads(num_attention_heads) {
	__cout(__PRETTY_FUNCTION__)

	dis >> Wq;
	dis >> bq;

	dis >> Wk;
	dis >> bk;

	dis >> Wv;
	dis >> bv;

	dis >> Wo;
	dis >> bo;
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

Matrix& PositionEmbedding::operator ()(Matrix &sequence, int mid) {
	int seq_len = sequence.rows();

	Matrix former = embeddings.topRows(mid);
	Matrix latter = embeddings.middleRows(1, seq_len - mid);
	Matrix pos_embeddings;
	pos_embeddings.resize(seq_len, embeddings.cols());
	pos_embeddings << former, latter;
	sequence += pos_embeddings;

	return sequence;
}

Matrix& PositionEmbedding::operator ()(Matrix &sequence) {
	int seq_len = sequence.rows();

	sequence += embeddings.topRows(seq_len);

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

vector<Vector> PositionEmbedding::compute_mask(vector<VectorI> &inputToken) {
	vector<VectorI> &mask = inputToken != 0;

	parallelize(mask, num_attention_heads);
	return revert_mask(mask, 10000.0);
}

PositionEmbedding::PositionEmbedding(KerasReader &dis, int num_attention_heads) :
		embeddings(dis.read_matrix()), num_attention_heads(num_attention_heads) {
}

Tensor& RevertMask::operator ()(const vector<MatrixI> &attention_mask) {
	Tensor res;
	return (*this)(attention_mask, res);
}

Tensor& RevertMask::operator ()(const vector<MatrixI> &attention_mask,
		Tensor &res) {

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

vector<VectorI> SegmentInput::operator ()(const vector<VectorI> &inputToken,
		vector<int> &inputMid) {
	int batch_size = inputToken.size();
	int length = inputToken[0].size();
	vector<VectorI> inputSegment;
	inputSegment.resize(batch_size);

	for (int k = 0; k < batch_size; ++k) {
		inputSegment[k].resize(length);
		int mid = inputMid[k];
		stosd(&inputSegment[k](0), 0, mid);
		stosd(&inputSegment[k](mid), 1, length - mid);
	}
	return inputSegment;
}

VectorI SegmentInput::operator ()(const VectorI &inputToken, int mid) {
	int length = inputToken.size();
//	cout << "inputToken.size() = " << inputToken.size() << endl;
	VectorI inputSegment;
	inputSegment.resize(length);

//	cout << "inputSegment.size() = " << inputSegment.size() << endl;

//	cout << "inputSegment(0) = " << inputSegment(0) << endl;

//	cout << "inputSegment(mid) = " << inputSegment(mid) << endl;

	stosd(&inputSegment(0), 0, mid);

	stosd(&inputSegment(mid), 1, length - mid);

	return inputSegment;
}

BertEmbedding::BertEmbedding(KerasReader &dis, int num_attention_heads) :
		wordEmbedding(dis),

		segmentEmbedding(dis),

		positionEmbedding(dis, num_attention_heads),

		layerNormalization(dis),

		embeddingMapping(dis, Activator::linear) {
	__cout(__PRETTY_FUNCTION__)

	embed_dim = wordEmbedding.wEmbedding.cols();
	hidden_size = embeddingMapping.weight.cols();
}

NonSegmentedBertEmbedding::NonSegmentedBertEmbedding(KerasReader &dis,
		int num_attention_heads) :
		wordEmbedding(dis),

		positionEmbedding(dis, num_attention_heads),

		layerNormalization(dis),

		embeddingMapping(dis, Activator::linear) {
	__cout(__PRETTY_FUNCTION__)

	embed_dim = wordEmbedding.wEmbedding.cols();
	hidden_size = embeddingMapping.weight.cols();
}

Tensor BertEmbedding::operator ()(vector<VectorI> &inputToken,
		const vector<int> &inputMid, const vector<VectorI> &inputSegment,
		vector<Vector> &mask) {
	auto embeddings = wordEmbedding(inputToken);
	mask = this->positionEmbedding.compute_mask(inputToken);

//	if (factorization(true)) {
//		embeddings = embeddingMapping(embeddings);
//	}

	auto segment_layer = segmentEmbedding(inputSegment);
	embeddings += segment_layer;
	auto &embed_layer = positionEmbedding(embeddings, inputMid);

	embed_layer = layerNormalization(embed_layer);

	if (hidden_size != embed_dim) {
		embeddings = embeddingMapping(embeddings);
	}

	return embeddings;
}

Matrix BertEmbedding::operator ()(VectorI &input_ids, int inputMid,
		const VectorI &inputSegment) {
	auto embeddings = wordEmbedding(input_ids);

//	cout << "wordEmbeddings = " << embeddings << endl;

//	if (factorization(true)) {
//		embeddings = embeddingMapping(embeddings);
//	}

	Matrix segment_layer;
	segmentEmbedding(inputSegment, segment_layer);
//	cout << "segment_layer = " << segment_layer << endl;

	embeddings += segment_layer;
//	cout << "embeddings = " << embeddings << endl;
	auto &embed_layer = positionEmbedding(embeddings, inputMid);
	embed_layer = layerNormalization(embed_layer);

	if (hidden_size != embed_dim) {
		embeddings = embeddingMapping(embeddings);
	}

	return embeddings;
}

Matrix BertEmbedding::operator ()(VectorI &input_ids,
		const VectorI &inputSegment) {
	auto embeddings = wordEmbedding(input_ids);

//	cout << "wordEmbeddings = " << embeddings << endl;

//	if (factorization(true)) {
//		embeddings = embeddingMapping(embeddings);
//	}

	Matrix segment_layer;
	segmentEmbedding(inputSegment, segment_layer);
//	cout << "segment_layer = " << segment_layer << endl;

	embeddings += segment_layer;
//	cout << "embeddings = " << embeddings << endl;
	auto &embed_layer = positionEmbedding(embeddings);
	embed_layer = layerNormalization(embed_layer);

	if (hidden_size != embed_dim) {
		embeddings = embeddingMapping(embeddings);
	}

	return embeddings;
}

Matrix NonSegmentedBertEmbedding::operator ()(const VectorI &input_ids) {
	auto embeddings = wordEmbedding(input_ids);

//	cout << "wordEmbeddings = " << embeddings << endl;

//	if (factorization(true)) {
//		embeddings = embeddingMapping(embeddings);
//	}

	auto &embed_layer = positionEmbedding(embeddings);
	embed_layer = layerNormalization(embed_layer);

	if (hidden_size != embed_dim) {
		embeddings = embeddingMapping(embeddings);
	}

	return embeddings;
}

Matrix NonSegmentedBertEmbedding::operator ()(const vector<int> &input_ids) {
	auto embeddings = wordEmbedding(input_ids);

//	cout << "wordEmbeddings = " << embeddings << endl;

//	if (factorization(true)) {
//		embeddings = embeddingMapping(embeddings);
//	}

	auto &embed_layer = positionEmbedding(embeddings);
	embed_layer = layerNormalization(embed_layer);

	if (hidden_size != embed_dim) {
		embeddings = embeddingMapping(embeddings);
	}

	return embeddings;
}

Encoder::Encoder(KerasReader &dis, int num_attention_heads,
		Activation hidden_act) :
		MultiHeadAttention(dis, num_attention_heads), MultiHeadAttentionNorm(
				dis), FeedForward(dis, hidden_act), FeedForwardNorm(dis) {
}

Tensor& Encoder::wrap_attention(Tensor &input_layer,
		const Tensor &attention_matrix, const vector<Vector> &mask) {
	return MultiHeadAttentionNorm(
			input_layer
					+ MultiHeadAttention(input_layer, attention_matrix, mask));
}

Tensor& Encoder::wrap_attention(Tensor &input_layer,
		const vector<Vector> &mask) {
	return MultiHeadAttentionNorm(
			input_layer + MultiHeadAttention(input_layer, mask));
}

Matrix& Encoder::wrap_attention(Matrix &input_layer) {
	input_layer += MultiHeadAttention(input_layer);
	return MultiHeadAttentionNorm(input_layer);
}

vector<Vector>& Encoder::wrap_attention(Tensor &input_layer,
		const vector<Vector> &mask, vector<Vector> &y) {
	return MultiHeadAttentionNorm(
			MultiHeadAttention(input_layer, mask, y) + extract(input_layer, 0));
}

Vector& Encoder::wrap_attention(Matrix &input_layer, Vector &y) {
	y = MultiHeadAttention(input_layer, y);
	y += input_layer.row(0);
	return MultiHeadAttentionNorm(y);
}

Tensor& Encoder::wrap_feedforward(Tensor &input_layer) {
	return FeedForwardNorm(input_layer + FeedForward(input_layer));
}

Matrix& Encoder::wrap_feedforward(Matrix &input_layer) {
	input_layer += FeedForward(input_layer);
	return FeedForwardNorm(input_layer);
}

vector<Vector>& Encoder::wrap_feedforward(vector<Vector> &input_layer) {
	return FeedForwardNorm(input_layer + FeedForward(input_layer));
}

Vector& Encoder::wrap_feedforward(Vector &input_layer) {
	input_layer += FeedForward(input_layer);
	return FeedForwardNorm(input_layer);
}

Tensor& Encoder::operator ()(Tensor &input_layer,
		const Tensor &attention_matrix, const vector<Vector> &mask) {
	Tensor &inputs = wrap_attention(input_layer, attention_matrix, mask);
	return wrap_feedforward(inputs);
}

Tensor& Encoder::operator ()(Tensor &input_layer, const vector<Vector> &mask) {
	Tensor &inputs = wrap_attention(input_layer, mask);
	return wrap_feedforward(inputs);
}

Matrix& Encoder::operator ()(Matrix &input_layer) {
	auto &inputs = wrap_attention(input_layer);
	return wrap_feedforward(inputs);
}

Encoder::Encoder() {
}

vector<Vector>& Encoder::operator ()(Tensor &input_layer,
		const vector<Vector> &mask, vector<Vector> &y) {
	auto &inputs = wrap_attention(input_layer, mask, y);
	return wrap_feedforward(inputs);
}

Vector& Encoder::operator ()(Matrix &input_layer, Vector &y) {
	auto &inputs = wrap_attention(input_layer, y);
	return wrap_feedforward(inputs);
}

AlbertTransformer::AlbertTransformer(KerasReader &dis, int num_hidden_layers,
		int num_attention_heads, Activation hidden_act) :
		num_hidden_layers(num_hidden_layers), encoder(dis, num_attention_heads,
				hidden_act) {
	__cout(__PRETTY_FUNCTION__)
}

BertTransformer::BertTransformer(KerasReader &dis, int num_hidden_layers,
		int num_attention_heads, Activation hidden_act) :
		num_hidden_layers(num_hidden_layers), encoder(num_hidden_layers) {
	__cout(__PRETTY_FUNCTION__)

	for (int i = 0; i < num_hidden_layers; ++i) {
		encoder[i] = Encoder(dis, num_attention_heads, hidden_act);
	}
}

Tensor& AlbertTransformer::operator ()(Tensor &input_layer,
		const Tensor &attention_matrix, const vector<Vector> &mask) {
	auto &last_layer = input_layer;
	for (int i = 0; i < num_hidden_layers; ++i) {
		last_layer = encoder(last_layer, attention_matrix, mask);
	}
	return last_layer;
}

Tensor& BertTransformer::operator ()(Tensor &input_layer,
		const Tensor &attention_matrix, const vector<Vector> &mask) {
	auto &last_layer = input_layer;
	for (int i = 0; i < num_hidden_layers; ++i) {

		last_layer = (*this)[i](last_layer, attention_matrix, mask);
	}
	return last_layer;
}

Encoder& BertTransformer::operator [](int i) {
	return encoder[i];
}

vector<Vector>& AlbertTransformer::operator ()(Tensor &input_layer,
		const vector<MatrixI> &attention_matrix, RevertMask &fn,
		const vector<Vector> &mask, vector<Vector> &y) {
	auto &last_layer = input_layer;
	for (int i = 0; i < num_hidden_layers; ++i) {
		if (i == num_hidden_layers - 1) {
			y = encoder(last_layer, mask, y);
		} else
			last_layer = encoder(last_layer, fn(attention_matrix), mask);
	}
	return y;
}

vector<Vector>& BertTransformer::operator ()(Tensor &input_layer,
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

vector<Vector>& AlbertTransformer::operator ()(Tensor &input_layer,
		const vector<Vector> &mask, vector<Vector> &y) {
	auto &last_layer = input_layer;
	for (int i = 0; i < num_hidden_layers; ++i) {
		if (i == num_hidden_layers - 1) {
			y = encoder(last_layer, mask, y);
		} else
			last_layer = encoder(last_layer, mask);
	}
	return y;
}

vector<Vector>& BertTransformer::operator ()(Tensor &input_layer,
		const vector<Vector> &mask, vector<Vector> &y) {
	auto &last_layer = input_layer;
	for (int i = 0; i < num_hidden_layers; ++i) {
		if (i == num_hidden_layers - 1) {
			y = (*this)[i](last_layer, mask, y);
		} else
			last_layer = (*this)[i](last_layer, mask);
	}
	return y;
}

Vector& AlbertTransformer::operator ()(Matrix &input_layer, Vector &y) {
	auto &last_layer = input_layer;
	for (int i = 0; i < num_hidden_layers; ++i) {
		if (i == num_hidden_layers - 1) {
			y = encoder(last_layer, y);
		} else
			last_layer = encoder(last_layer);
	}
	return y;
}

Vector& BertTransformer::operator ()(Matrix &input_layer, Vector &y) {
	auto &last_layer = input_layer;
	for (int i = 0; i < num_hidden_layers; ++i) {
		if (i == num_hidden_layers - 1) {
			y = (*this)[i](last_layer, y);
		} else
			last_layer = (*this)[i](last_layer);
	}
	return y;
}

//bool cross_layer_parameter_sharing = true;
Pairwise::Pairwise(KerasReader &dis, const string &vocab,
		int num_attention_heads, bool symmetric_position_embedding,
		int num_hidden_layers) :
		tokenizer(vocab), symmetric_position_embedding(
				symmetric_position_embedding), midIndex(tokenizer.vocab.at(u"[SEP]")), bertEmbedding(
		dis, num_attention_heads), transformer(
		dis, num_hidden_layers,
		num_attention_heads, {Activator::gelu}), poolerDense(dis), similarityDense(dis, Activator::sigmoid) {
			__cout(__PRETTY_FUNCTION__)
		}

//bool cross_layer_parameter_sharing = true;
PairwiseVector::PairwiseVector(KerasReader &dis, Activation hidden_act,
		int num_attention_heads, int num_hidden_layers) :
		bertEmbedding(dis, num_attention_heads),

		transformer(dis, num_hidden_layers, num_attention_heads, hidden_act),

		bilinear(dis, { Activator::softmax }) {
	__log(__PRETTY_FUNCTION__)
}

PairwiseVectorChar::PairwiseVectorChar(KerasReader &dis, const string &vocab,
		int num_attention_heads, int num_hidden_layers) :
		PairwiseVector(dis, { Activator::relu }, num_attention_heads,
				num_hidden_layers),

		word2id(Text(vocab).read_vocab(0)) {
	__log(__PRETTY_FUNCTION__)
}

PairwiseVectorSP::PairwiseVectorSP(KerasReader &dis, const string &path,
		int num_attention_heads, int num_hidden_layers) :
		PairwiseVector(dis, { Activator::gelu }, num_attention_heads,
				num_hidden_layers),

		sp(path) {
	__log(__PRETTY_FUNCTION__)
}

#include "../json/json.h"
Json::Value readFromStream(const string &json_file);

Pairwise& Pairwise::paraphrase() {
	static const auto &config = readFromStream(
			modelsDirectory() + "cn/paraphrase/config.json");

//	std::cout << config << std::endl;
//	for (auto &key : config.getMemberNames()) {
//		std::cout << key << " = " << config[key] << std::endl;
//	}

	static int num_attention_heads = 12;
//	static bool cross_layer_parameter_sharing =
//			config["cross_layer_parameter_sharing"].asBool();
	static const auto &position_embedding_type =
			config["position_embedding_type"].asString();
	static bool symmetric_position_embedding = position_embedding_type
			== "symmetric";

	static int num_hidden_layers = config["num_hidden_layers"].asInt();

	static Pairwise inst(
			(KerasReader&) (const KerasReader&) KerasReader(
					modelsDirectory() + "cn/paraphrase/model.h5"),
			modelsDirectory() + "cn/bert/vocab.txt", num_attention_heads,
//			cross_layer_parameter_sharing,
			symmetric_position_embedding, num_hidden_layers);
	__cout(__PRETTY_FUNCTION__)

	return inst;
}

PairwiseVectorChar& PairwiseVectorChar::lexicon() {
	static const auto &config = readFromStream(
			modelsDirectory() + "cn/lexicon/config.json");

	static int num_attention_heads = 12;

	static int num_hidden_layers = config["num_hidden_layers"].asInt();

	static PairwiseVectorChar inst(
			(KerasReader&) (const KerasReader&) KerasReader(
					modelsDirectory() + "cn/lexicon/model.h5"),
			modelsDirectory() + "cn/bert/vocab.txt", num_attention_heads,
			num_hidden_layers);
//	__cout(__PRETTY_FUNCTION__)
	return inst;
}

PairwiseVectorChar& PairwiseVectorChar::instantiateHyponym() {
	static const auto &config = readFromStream(
			modelsDirectory() + "cn/lexicon/config.json");

	auto &inst = lexicon();
	inst = PairwiseVectorChar(
			(KerasReader&) (const KerasReader&) KerasReader(
					modelsDirectory() + "cn/lexicon/model.h5"),
			modelsDirectory() + "cn/bert/vocab.txt", 12,
			config["num_hidden_layers"].asInt());
	__log(__PRETTY_FUNCTION__)
	return inst;
}

PairwiseVectorSP& PairwiseVectorSP::lexicon() {
	static const auto &config = readFromStream(
			modelsDirectory() + "en/lexicon/config.json");

	static int num_attention_heads = 12;

	static int num_hidden_layers = config["num_hidden_layers"].asInt();

	static PairwiseVectorSP inst(
			(KerasReader&) (const KerasReader&) KerasReader(
					modelsDirectory() + "en/lexicon/model.h5"),
			modelsDirectory() + "en/bert/albert_base/30k-clean.model",
			num_attention_heads, num_hidden_layers);
//	__cout(__PRETTY_FUNCTION__)
	return inst;
}

PairwiseVectorSP& PairwiseVectorSP::instantiateHyponym() {
	static const auto &config = readFromStream(
			modelsDirectory() + "en/lexicon/config.json");

	auto &inst = lexicon();
	(PairwiseVector&) inst = PairwiseVector(
			(KerasReader&) (const KerasReader&) KerasReader(
					modelsDirectory() + "en/lexicon/model.h5"),

			{ Activator::gelu }, 12, config["num_hidden_layers"].asInt());
//	__cout(__PRETTY_FUNCTION__)
	return inst;
}

Pairwise& Pairwise::lexicon() {
	static const auto &config = readFromStream(
			modelsDirectory() + "cn/lexicon_pairwise/config.json");

//	std::cout << config << std::endl;
//	for (auto &key : config.getMemberNames()) {
//		std::cout << key << " = " << config[key] << std::endl;
//	}

	static int num_attention_heads = 12;
//	static bool cross_layer_parameter_sharing =
//			config["cross_layer_parameter_sharing"].asBool();
	static const auto &position_embedding_type =
			config["position_embedding_type"].asString();

	static bool symmetric_position_embedding = position_embedding_type
			== "symmetric";

	static int num_hidden_layers = config["num_hidden_layers"].asInt();

	static Pairwise inst(
			(KerasReader&) (const KerasReader&) KerasReader(
					modelsDirectory() + "cn/lexicon_pairwise/model.h5"),
			modelsDirectory() + "cn/bert/vocab.txt", num_attention_heads,
//			cross_layer_parameter_sharing,
			symmetric_position_embedding, num_hidden_layers);
//	__cout(__PRETTY_FUNCTION__)
	return inst;
}

vector<double> Pairwise::operator ()(vector<VectorI> &input_ids) {
	auto inputMid = midIndex(input_ids);

	auto inputSegment = segmentInput(input_ids, inputMid);

//	auto &matrixAttention = CrossAttentionMask(inputSegment);

	vector<Vector> mask;
	auto embed_layer = bertEmbedding(input_ids, inputMid, inputSegment, mask);

	vector<Vector> clsEmbedding;
	transformer(embed_layer, mask, clsEmbedding);
	auto &sent = poolerDense(clsEmbedding);
	vector<double> similarity;
	int batch_size = sent.size();
	similarity.resize(batch_size);
	sent = similarityDense(sent);
	for (int k = 0; k < batch_size; ++k) {
		similarity[k] = sent[k](0);
	}
	return similarity;
}

double Pairwise::operator ()(VectorI &input_ids) {
//	cout << "input_ids = " << input_ids << endl;

	auto inputMid = midIndex(input_ids);

//	cout << "inputMid = " << inputMid << endl;

	auto inputSegment = segmentInput(input_ids, inputMid);

//	cout << "inputSegment = " << inputSegment << endl;

//	auto &matrixAttention = CrossAttentionMask(inputSegment);

	auto embed_layer =
			symmetric_position_embedding ?
					bertEmbedding(input_ids, inputMid, inputSegment) :
					bertEmbedding(input_ids, inputSegment);
//	cout << "embed_layer = " << embed_layer << endl;

	Vector clsEmbedding;
	transformer(embed_layer, clsEmbedding);

	auto &sent = poolerDense(clsEmbedding);

	sent = similarityDense(sent);

	return sent(0);
}

Vector PairwiseVector::operator ()(const VectorI &input_ids) {
	__cout(input_ids)

	auto embed_layer = bertEmbedding(input_ids);

	Vector clsEmbedding;
	transformer(embed_layer, clsEmbedding);

	return clsEmbedding;
}

Vector PairwiseVector::operator ()(const vector<int> &input_ids) {
	__cout(input_ids)

	auto embed_layer = bertEmbedding(input_ids);

	Vector clsEmbedding;
	transformer(embed_layer, clsEmbedding);

	return clsEmbedding;
}

Vector PairwiseVector::operator ()(const vector<int> &input_ids,
		const vector<int> &input_ids1) {
	Vector sent = (*this)(input_ids);
	Vector sent1 = (*this)(input_ids1);

	return bilinear(sent, sent1);
}

Vector PairwiseVector::operator ()(const VectorI &input_ids,
		const VectorI &input_ids1) {
	Vector sent = (*this)(input_ids);
	Vector sent1 = (*this)(input_ids1);

	return bilinear(sent, sent1);
}

Matrix PairwiseVector::operator ()(const vector<VectorI> &input_ids) {
	int n = input_ids.size();
	vector<Vector> sent(n);

#pragma omp parallel for
	for (int index = 0; index < n; ++index) {
		sent[index] = (*this)(input_ids[index]);
	}

	return (*this)(sent);
}

Matrix PairwiseVector::operator ()(const vector<vector<int>> &input_ids) {
	int n = input_ids.size();
	vector<Vector> sent(n);

#pragma omp parallel for
	for (int index = 0; index < n; ++index) {
		sent[index] = (*this)(input_ids[index]);
	}

	return (*this)(sent);
}

double Pairwise::operator ()(const vector<String> &s) {
	auto v = tokenizer.convert_tokens_to_ids(s);
	return (*this)(v);
}

double Pairwise::operator ()(String &x, String &y) {
	if (x.size() > 510) {
		x.resize(510);
	}

	if (y.size() > 510) {
		y.resize(510);
	}

//	cout << "x = " << x << endl;
//	cout << "y = " << y << endl;
	return (*this)(tokenizer.tokenize(x, y));
}

Vector PairwiseVectorChar::operator ()(const String &x, const String &y) {
//	cout << "x = " << x << endl;
//	cout << "y = " << y << endl;
	vector<String> s_x;
	s_x << u"[CLS]";
	for (auto ch : x) {
		s_x << String(1, tolower(ch));
	}
	s_x << u"[SEP]";

	vector<String> s_y = { u"[CLS]"};
	for (auto ch : y) {
		s_y << String(1, tolower(ch));
	}

	s_y << u"[SEP]";
	auto input_ids = string2id(s_x, word2id);
	auto input_ids1 = string2id(s_y, word2id);

	return (*this)(input_ids, input_ids1);
}

Vector PairwiseVectorSP::operator ()(const string &x, const string &y) {
	vector<string> s_x;
	s_x << "[CLS]";
	s_x << sp.EncodeAsPieces(x);
	s_x << "[SEP]";

	vector<string> s_y = { "[CLS]" };
	s_y << sp.EncodeAsPieces(y);
	s_y << "[SEP]";

	auto input_ids = sp.PieceToId(s_x);
	auto input_ids1 = sp.PieceToId(s_y);

	return (*this)(input_ids, input_ids1);
}

Matrix PairwiseVectorChar::operator ()(const vector<String> &str) {
//	cout << "x = " << x << endl;
//	cout << "y = " << y << endl;
	vector<VectorI> input_ids(str.size());
	int i = 0;
	for (auto &sent : str) {
		vector<String> s;
		s << u"[CLS]";

		for (auto ch : sent) {
			s << String(1, tolower(ch));
		}
		s << u"[SEP]";

		input_ids[i++] = string2id(s, word2id);
	}

	return (*this)(input_ids);
}

Matrix PairwiseVectorSP::operator ()(const vector<string> &str) {
	vector<vector<int>> input_ids(str.size());
	int i = 0;
	for (auto &sent : str) {
		vector<string> s = { "[CLS]" };

		s << sp.EncodeAsPieces(sent);
		s << "[SEP]";

		input_ids[i++] = sp.PieceToId(s);
	}

	return (*this)(input_ids);
}

Matrix PairwiseVector::operator ()(const vector<Vector> &sent) {
	int n = sent.size();

	Matrix scores;
	scores.resize(n, n);

	for (int i = 0; i < n; ++i) {
		scores(i, i) = 0;
	}

	int size = n * (n - 1) / 2;

	vector<std::pair<int, int>> indices(size);
	int index = 0;
	for (int j = 1; j < n; ++j)
		for (int i = 0; i < j; ++i)
			indices[index++] = { i, j };

#pragma omp parallel for
	for (int k = 0; k < size; ++k) {
		int i = indices[k].first;
		int j = indices[k].second;
		//guarantee that i < j
		auto probability = bilinear(sent[i], sent[j]);
		scores(i, j) = probability2score(probability);
		scores(j, i) = probability2score(symmetric_transform(probability));
	}

	return scores;
}

Vector PairwiseVectorChar::operator ()(const String &str) {

	vector<String> s;
	s << u"[CLS]";
	for (auto ch : str) {
		s << String(1, tolower(ch));
	}
	s << u"[SEP]";

	return (*this)(string2id(s, word2id));
}

Vector PairwiseVectorSP::operator ()(const string &str) {

	vector<string> s;
	s << "[CLS]";
	s << sp.EncodeAsPieces(str);
	s << "[SEP]";

	return (*this)(sp.PieceToId(s));
}

double Pairwise::operator ()(const char16_t *_x, const char16_t *_y) {
	String x = _x;
	String y = _y;
	cout << "first sentence: " << x << endl;
	cout << "second sentence: " << y << endl;
	return (*this)(x, y);
}

Vector PairwiseVectorChar::operator ()(const char16_t *_x, const char16_t *_y) {
	String x = _x;
	String y = _y;
	cout << "first sentence: " << x << endl;
	cout << "second sentence: " << y << endl;
	if (x.size() > 510) {
		x.resize(510);
	}

	if (y.size() > 510) {
		y.resize(510);
	}

	return (*this)(x, y);
}

vector<String> whitespace_tokenize(String &text) {
//        """Runs basic whitespace cleaning and splitting on a piece of text."""
	text = strip(text);
	if (!text)
		return vector<String>();
	return split(text);
}

FullTokenizer::FullTokenizer(const string &vocab_file, bool do_lower_case) :
		vocab(Text(vocab_file).read_vocab(0)), unk_token(u"[UNK]"),
		max_input_chars_per_word(200), do_lower_case(do_lower_case) {
			__cout(__PRETTY_FUNCTION__)
		}

FullTokenizer& FullTokenizer::instance_cn() {
	static FullTokenizer instance(modelsDirectory() + "cn/bert/vocab.txt");
	return instance;
}

FullTokenizer& FullTokenizer::instance_en() {
	static FullTokenizer instance(modelsDirectory() + "en/bert/vocab.txt");
	return instance;
}

VectorI FullTokenizer::convert_tokens_to_ids(const vector<String> &items) {
	VectorI output;
	output.resize(items.size());

	int index = 0;
	for (auto &item : items) {
		auto iter = vocab.find(item);
//		output(index) = iter == vocab.end() ? -vocab.at(lstrip(item)) : iter->second;
		output(index) = iter == vocab.end() ? 1 : iter->second;

		++index;
	}
	return output;
}

vector<String> FullTokenizer::wordpiece_tokenize(String &chars) {
//        """Tokenizes a piece of text into its word pieces.
//
//        This uses a greedy longest-match-first algorithm to perform tokenization
//        using the given vocabulary.
//
//        For example:
//            input = "unaffable"
//            output = ["un", "##aff", "##able"]
//
//        Args:
//            text: A single token or whitespace separated tokens. This should have
//                already been passed through `FullTokenizer.
//
//        Returns:
//            A list of wordpiece tokens.
//        """

	vector<String> output_tokens;

	if (chars.size() > max_input_chars_per_word) {
		output_tokens << unk_token;
	} else {
		if (do_lower_case)
			tolower(chars);

		bool is_bad = false;
		size_t start = 0;

		while (start < chars.size()) {
			auto end = chars.size();
			String cur_substr, substr;
			while (start < end) {
				substr = chars.substr(start, end - start);
				if (start > 0) {
					substr = u"##" + substr;
				}

				if (vocab.count(substr)) {
					cur_substr = substr;
					break;
				}
				--end;
			}
			if (!cur_substr) {
				is_bad = true;
				if (unknownSet.count(substr)) {
					unknownSet[substr] += 1;
				} else {
					cout << "unknown word encountered " << substr << ", from "
							<< chars << endl;
					unknownSet[substr] = 1;
				}
				break;
			}

			output_tokens << cur_substr;
			start = end;
		}

		if (is_bad)
			output_tokens << unk_token;
	}

	chars.clear();
	return output_tokens;
}

vector<String> FullTokenizer::_run_split_on_punc(String &text) {
//        """Splits punctuation on a piece of text."""
	size_t i = 0;
	bool start_new_word = true;
	vector<String> output;

	while (i < text.size()) {
		auto ch = text[i];
		if (_is_punctuation(ch)) {
			output << String(1, ch);
			start_new_word = true;
		} else {
			if (start_new_word)
				output << String();
			start_new_word = false;
			output.back() += ch;
		}
		++i;
	}

	return output;
}

vector<String> FullTokenizer::tokenize(const String &x, const String &y) {
	vector<String> s = { u"[CLS]"};

s << tokenize(x);
s << u"[SEP]";
s << tokenize(y);
s << u"[SEP]";
return s;
}

vector<String> FullTokenizer::tokenize(const String &text) {
	//# This was added on November 1st, 2018 for the multilingual and Chinese
	//# models. This is also applied to the English models now, but it doesn't
	//# matter since the English models were not trained on any Chinese data
	//# and generally don't have any Chinese data in them (there are Chinese
	//# characters in the vocabulary because Wikipedia does have some Chinese
	//# words in the English Wikipedia.).

//        """Adds whitespace around any CJK character."""
	//cout << "text = " << text << endl;

	vector<String> output;
	String word;

	for (auto ch : text) {
		if (ch == 0 || ch == 0xfffd || iswcntrl(ch)) {
			continue;
		}

		if (iswspace(ch)) {
			if (!!word) {
				//	cout << "word = " << word << endl;
				output << wordpiece_tokenize(word);
			}
			continue;
		}

		if (_is_chinese_char(ch) || _is_punctuation(ch)) {
			if (!!word) {
				//		cout << "word = " << word << endl;
				output << wordpiece_tokenize(word);
			}

			String substr(1, ch);
			if (vocab.count(substr))
				output << substr;
			else
				output << this->unk_token;
		} else {
			word += ch;
		}
	}

	if (!!word) {
		//	cout << "word = " << word << endl;
		output << wordpiece_tokenize(word);
	}

//	cout << "output = " << output << endl;
	return output;
}

bool FullTokenizer::_is_punctuation(word cp) {
//        """Checks whether `chars` is a punctuation character."""
//    # We treat all non-letter/number ASCII as punctuation.
//    # Characters such as "^", "$", and "`" are not in the Unicode
//    # Punctuation class but we treat them as punctuation anyways, for
//    # consistency.
	if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64)
			|| (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
		return true;
	return iswpunct(cp);
}

bool FullTokenizer::_is_chinese_char(word cp) {
//        """Checks whether CP is the codepoint of a CJK character."""
//# This defines a "chinese character" as anything in the CJK Unicode block:
//#   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
//#
//# Note that the CJK Unicode block is NOT all Japanese and Korean characters,
//# despite its name. The modern Korean Hangul alphabet is a different block,
//# as is Japanese Hiragana and Katakana. Those alphabets are used to write
//# space-separated words, so they are not treated specially and handled
//# like the all of the other languages.
	if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
//            (cp >= 0x20000 && cp <= 0x2A6DF) ||
//            (cp >= 0x2A700 && cp <= 0x2B73F) ||
//            (cp >= 0x2B740 && cp <= 0x2B81F) ||
//            (cp >= 0x2B820 && cp <= 0x2CEAF) ||
			(cp >= 0xF900 && cp <= 0xFAFF) //||
//            (cp >= 0x2F800 && cp <= 0x2FA1F))
			)
		return true;

	return false;
}

String& FullTokenizer::_clean_text(String &text) {
//        """Performs invalid character removal and whitespace cleanup on text."""
	for (size_t i = 0; i < text.size();) {
		word ch = text[i];
		if (ch == 0 || ch == 0xfffd || iswcntrl(ch)) {
			text.erase(i);
			continue;
		}

		if (iswspace(ch)) {
			text[i] = ' ';
		}
		++i;
	}
	return text;
}

struct ClusteringAlgorithm {
	struct less {
		double *priority_of_cluster;
		less(double *priority_of_cluster = nullptr) :
				priority_of_cluster(priority_of_cluster) {
			__cout(__PRETTY_FUNCTION__)
		}

		bool operator ()(int x, int y) {
			return priority_of_cluster[x] < priority_of_cluster[y];
		}
	};

	ClusteringAlgorithm(Matrix &scores, const vector<int> &frequency) :
			scores(scores),

			n(scores.rows()),

			max_num_of_clusters(sqrt(2 * n)),

			heads(n, -1),

			num_of_children(n, 0),

			priority_of_cluster(n, 0.0),

			pq(less(&priority_of_cluster[0])) {

		for (int child = 0; child < n; ++child) {
			int parent;
			scores.col(child).maxCoeff(&parent);
			heads[child] = parent;

			++num_of_children[parent];
			priority_of_cluster[parent] += scores(parent, child);
		}
		__cout(num_of_children)

		double max_frequency = frequency[0];
		for (int parent = 0; parent < n; ++parent) {
			double term_weight = frequency[parent] / max_frequency;
			priority_of_cluster[parent] += 2 * term_weight * term_weight;

			pq.insert(parent);
		}
		__cout(priority_of_cluster)
	}

	Matrix &scores;
	int n;
	int max_num_of_clusters;
	vector<int> heads;
	vector<int> num_of_children;
	vector<double> priority_of_cluster;
	priority_dict<int, less> pq;

	bool sanctity_check() {
		bool success = true;
		for (int child = 0; child < n; ++child) {
			int parent = heads[child];
			if (parent < 0)
				continue;
			int ancestor = heads[parent];
			if (ancestor >= 0) {
				cout << "parent of " << child << " = " << parent
						<< ", parent of " << parent << " = " << ancestor
						<< endl;
				success = false;
			}
		}
		return success;
	}

	void run() {
		while (!pq.empty()) {
			int parent = pq.pop();

			__cout(parent)
			__cout(num_of_children);
			__cout(priority_of_cluster);
			if (parent < 0)
				continue;

			int numOfChildren = num_of_children[parent];
			__cout(numOfChildren)
			__cout(priority_of_cluster[parent])

			if (!numOfChildren) {
//				cout << "leaf node detected, with priority = " << priority_of_cluster[parent] << endl;
				continue;
			}

			if (numOfChildren <= 2) {
//the parent of this child has too few children, so this child should abandon its current parent and find another parent!
				if (change_parent_for(find_child(parent)))
					continue;

				cout << "failed to make adjustment for " << parent << endl;
				break;
			}

			if (numOfChildren > max_num_of_clusters) {
//this parent has too many children, so this parent should abandon one of its current children and assign this abandoned child to another parent!
				if (change_parent_for(find_worst_child(parent)))
					continue;

				cout << "failed to make adjustment for " << parent << endl;
				break;
			}

			//parent should have no head, so remove its forefather
			remove_child(parent);
		}
	}

	int find_child(int parent) {
		for (int child = 0; child < n; ++child) {
			if (heads[child] == parent) {
				return child;
			}
		}
		return -1;
	}

	int find_worst_child(int parent) {
		double min_score = oo;
		double unwanted_child = -1;
		for (int child = 0; child < n; ++child) {
			if (heads[child] == parent) {
				auto _min_score = scores(parent, child);
				if (_min_score < min_score) {
					min_score = _min_score;
					unwanted_child = child;
				}
			}
		}
		return unwanted_child;
	}

	int remove_child(int child) {
		int parent = heads[child];
		if (parent >= 0) {
			auto &score = scores(parent, child);
			pq.erase(parent);
			--num_of_children[parent];
			priority_of_cluster[parent] -= score;

			score = 0;
			pq.insert(parent);

			heads[child] = -1;
		}

		return parent;
	}

	int assign_parent_for(int child) {
		int parent;
		scores.col(child).maxCoeff(&parent);
//		if (heads[child] == parent){
//			cout << "algorithm could not find a better parent" << endl;
//			return false;
//		}

		heads[child] = parent;

		pq.erase(parent);
		++num_of_children[parent];
		priority_of_cluster[parent] += scores(parent, child);

		pq.insert(parent);
		return parent;
	}

	bool change_parent_for(int child) {
		int old_parent = remove_child(child);
		int new_parent = assign_parent_for(child);
		return new_parent != old_parent;
	}
};

vector<int> lexiconStructure(Matrix &scores, const vector<int> &frequency) {
	ClusteringAlgorithm cluster(scores, frequency);
	cluster.run();
	assert(cluster.sanctity_check());
	return cluster.heads;
}

vector<int> lexiconStructure(const vector<String> &keywords,
		const vector<int> &frequency) {
//	cout << "keywords = " << keywords << endl;
//	cout << "frequency = " << frequency << endl;
	auto scores = PairwiseVectorChar::lexicon()(keywords);
	return lexiconStructure(scores, frequency);
}

vector<int> lexiconStructureCN(const vector<vector<double>> &_embedding,
		const vector<int> &frequency) {
//	cout << "keywords = " << keywords << endl;
//	cout << "frequency = " << frequency << endl;
	int size = _embedding.size();
	vector<Vector> embedding(size);
	for (int i = 0; i < size; ++i) {
		int sz = _embedding[i].size();
		embedding[i].resize(sz);
		movsq(embedding[i].data(), _embedding[i].data(), sz);
	}

	auto scores = PairwiseVectorChar::lexicon()(embedding);
	return lexiconStructure(scores, frequency);
}

vector<int> lexiconStructure(const vector<string> &keywords,
		const vector<int> &frequency) {
	auto scores = PairwiseVectorSP::lexicon()(keywords);
	return lexiconStructure(scores, frequency);
}

double PairwiseVector::probability2score(const Vector &probability) {
	static const int k = 5;
	static double interval[][2] = { { 0, 0.01 },

	{ 0.02, 0.2 },

	{ 1, 2 },

	{ 2, 3 },

	{ 3.5, 4 } };

	static auto probability2score = [](double p, double a, double b) {
		return (b - a) * k / (k - 1) * p + (k * a - b) / (k - 1);
	};

	int argmax;
	probability.maxCoeff(&argmax);
	auto pair = interval[argmax];
	return probability2score(probability(argmax), pair[0], pair[1]);
}

Vector& PairwiseVector::symmetric_transform(Vector &y_pred) {
	std::swap(y_pred(0), y_pred(4));
	return y_pred;
}

const String& PairwiseVector::lexicon_label(const Vector &y_pred) {

	static String labels[] = { u"hypernym",u"unrelated", u"related",u"synonym", u"hyponym"};

int argmax;
y_pred.maxCoeff(&argmax);
return labels[argmax];
}

sentencepiece::SentencePieceProcessor& en_tokenizer() {

	static sentencepiece::SentencePieceProcessor processor;

	static const auto status = processor.Load(
			modelsDirectory() + "en/bert/albert_base/30k-clean.model");

	return processor;

}
