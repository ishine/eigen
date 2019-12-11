#pragma once
#include "keras.h"

template<typename _Ty>
vector<_Ty>& parallelize(vector<_Ty> &mask, int num_attention_heads) {
	int batch_size = mask.size();
	mask.resize(batch_size * num_attention_heads);
	for (int i = batch_size - 1; i >= 0; --i) {
		for (int j = 0; j < num_attention_heads; ++j)
			mask[i * num_attention_heads + j] = mask[i];
	}

	return mask;
}

struct FeedForward {
	/**
	 *
	 */

	Matrix W1, W2;
	Vector b1, b2;

	Vector& operator()(const Vector &x, Vector &ret);
	Vector& operator()(Vector &x);

	Matrix& operator()(Matrix &x, Matrix &wDense);
	Matrix& operator()(Matrix &x);

	Tensor& operator()(const Tensor &x);
	vector<Vector>& operator()(const vector<Vector> &x);
	FeedForward(BinaryReader &dis, bool bias = true);
	FeedForward();
};

struct CrossAttentionMask {
	CrossAttentionMask(int num_attention_heads = 0, bool diagnal_attention =
			false);
	int num_attention_heads;
	bool diagnal_attention;

	vector<MatrixI>& operator()(const vector<VectorI> &segment_ids);

	vector<MatrixI>& operator()(const vector<VectorI> &segment_ids,
			int num_attention_heads);
};

struct LayerNormalization {
	LayerNormalization(BinaryReader &dis);
	LayerNormalization();
	const static double epsilon;
	Vector gamma, beta;

	Tensor& operator()(Tensor &x);
	vector<Vector>& operator()(vector<Vector> &x);
};

struct MidIndex {
	MidIndex(int SEP = 102);
	int SEP;
	vector<int>& operator()(const vector<VectorI> &token);
};

struct MultiHeadAttention {
	MultiHeadAttention(BinaryReader &dis);
	MultiHeadAttention();

	Tensor& operator()(const Tensor &sequence, const Tensor &attention_matrix,
			const vector<Vector> &mask);

	vector<Vector>& operator ()(const Tensor &sequence,
			const vector<Vector> &mask);

	vector<Vector>& operator ()(const Tensor &sequence,
			const vector<Vector> &mask, vector<Vector> &y);

	Matrix Wq, Wk, Wv, Wo;
	Vector bq, bk, bv, bo;

	int num_attention_heads;

	Tensor& scaled_dot_product_attention(Tensor &query, Tensor &key,
			Tensor &value, const Tensor &attention_mask,
			const vector<Vector> &mask);

	vector<Vector>& scaled_dot_product_attention(vector<Vector> &query,
			const Tensor &key, const Tensor &value, const vector<Vector> &mask);

	Tensor& reshape_to_batches(Tensor&);
	vector<Vector>& reshape_to_batches(vector<Vector>&);
	Tensor& reshape_from_batches(Tensor&);
	vector<Vector>& reshape_from_batches(vector<Vector>&);
};

struct PositionEmbedding {
	PositionEmbedding(BinaryReader &dis, int num_attention_heads);

	Matrix embeddings;
	int num_attention_heads;

	Tensor& operator()(Tensor &sequence, const vector<int> &mid);
	Tensor& operator()(Tensor &sequence);

	vector<Vector>& compute_mask(vector<VectorI> &inputToken);
};

struct RevertMask {
	RevertMask(double cross_attention);
	double cross_attention;
	Vector weight;
	int step;
	Tensor& operator()(const vector<MatrixI> &mask, Tensor &res);
	Tensor& operator()(const vector<MatrixI> &mask);
};

struct SegmentInput {
	vector<VectorI>& operator()(const vector<VectorI> &token,
			vector<int> &inputMid);
};

struct BertEmbedding {
	BertEmbedding(BinaryReader &dis, int num_attention_heads,
			bool symmetric_positional_embedding,
			bool factorization_on_word_embedding_only);

	bool factorization_on_word_embedding_only, symmetric_positional_embedding;

	Embedding WordEmbedding;
	Embedding SegmentEmbedding;
	::PositionEmbedding PositionEmbedding;
	::LayerNormalization LayerNormalization;
	DenseLayer Dense;
	int embed_dim, hidden_size;

	Tensor& operator ()(vector<VectorI> &inputToken,
			const vector<int> &inputMid, const vector<VectorI> &inputSegment,
			vector<Vector> &mask);

	vector<Vector>& compute_mask(vector<VectorI> &inputToken);

	bool factorization(bool word_embedding_only = true);
};

struct Encoder {
	Encoder(BinaryReader &dis);
	Encoder();
	::MultiHeadAttention MultiHeadAttention;
	LayerNormalization MultiHeadAttentionNorm;
	::FeedForward FeedForward;
	LayerNormalization FeedForwardNorm;

	Tensor& wrap_attention(Tensor &input_layer, const Tensor &attention_matrix,
			const vector<Vector> &mask);

	vector<Vector>& wrap_attention(Tensor &input_layer,
			const vector<Vector> &mask);

	vector<Vector>& wrap_attention(Tensor &input_layer,
			const vector<Vector> &mask, vector<Vector> &y);

	Tensor& wrap_feedforward(Tensor &input_layer);
	vector<Vector>& wrap_feedforward(vector<Vector> &input_layer);

	Tensor& operator ()(Tensor &input_layer, const Tensor &attention_matrix,
			const vector<Vector> &mask);
	vector<Vector>& operator ()(Tensor &input_layer,
			const vector<Vector> &mask);
	vector<Vector>& operator ()(Tensor &input_layer, const vector<Vector> &mask,
			vector<Vector> &y);
};

struct Transformer {
	Transformer(BinaryReader &dis, bool cross_layer_parameter_sharing,
			int num_hidden_layers);
	int num_hidden_layers;
	object<Encoder> encoder;
	Encoder& operator [](int i);

	Tensor& operator ()(Tensor &input_layer,
			const vector<MatrixI> &attention_matrix, RevertMask &fn,
			const vector<Vector> &mask);

	Tensor& operator ()(Tensor &input_layer, const Tensor &attention_matrix,
			const vector<Vector> &mask);

	vector<Vector>& operator ()(Tensor &input_layer,
			const vector<MatrixI> &attention_matrix, RevertMask &fn,
			const vector<Vector> &mask, vector<Vector> &y);

};

struct Paraphrase {
	Paraphrase(BinaryReader &dis, int num_attention_heads,
			bool symmetric_positional_embedding = false,
			bool factorization_on_word_embedding_only = true,
			bool cross_layer_parameter_sharing = true, int num_hidden_layers =
					12, double cross_attention = 3.0);
	::MidIndex MidIndex;
	::SegmentInput SegmentInput;
	::CrossAttentionMask CrossAttentionMask;
	::RevertMask RevertMask;
	::BertEmbedding BertEmbedding;

	::Transformer Transformer;
	DenseLayer poolerDense;
	DenseLayer similarityDense;

	vector<double>& operator ()(vector<VectorI> &input_ids);
};

