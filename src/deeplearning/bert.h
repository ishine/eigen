#pragma once
#include "../std/utility.h"
#include<vector>
using std::vector;

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
	Activation activation = { Activator::relu };
	Vector& operator()(const Vector &x, Vector &ret);
	Vector operator()(const Vector &x);

	Matrix& operator()(Matrix &x, Matrix &wDense);
	Matrix operator()(const Matrix &x);

	Tensor operator()(const Tensor &x);
	vector<Vector> operator()(const vector<Vector> &x);
	FeedForward(KerasReader &dis, bool bias = true);
	FeedForward(KerasReader &dis, Activation activation);
	FeedForward();
};

struct CrossAttentionMask {
	CrossAttentionMask(int num_attention_heads = 0, bool diagnal_attention =
			false);
	int num_attention_heads;
	bool diagnal_attention;

	vector<MatrixI> operator()(const vector<VectorI> &segment_ids);

	vector<MatrixI> operator()(const vector<VectorI> &segment_ids,
			int num_attention_heads);
};

struct LayerNormalization {
	LayerNormalization(KerasReader &dis);
	LayerNormalization();
	const static double epsilon;
	Vector gamma, beta;

	Tensor& operator()(Tensor &x);
	Matrix& operator()(Matrix &x);
	vector<Vector>& operator()(vector<Vector> &x);
	Vector& operator()(Vector &x);
};

struct MidIndex {
//	MidIndex(int SEP = 102);
	MidIndex(int SEP);
	int SEP;
	vector<int> operator()(const vector<VectorI> &token);
	int operator()(const VectorI &token);
};

struct MultiHeadAttention {
	MultiHeadAttention(KerasReader &dis, int num_attention_heads);
	MultiHeadAttention();

	Tensor operator()(const Tensor &sequence, const Tensor &attention_matrix,
			const vector<Vector> &mask);
	Tensor operator()(const Tensor &sequence, const vector<Vector> &mask);
	Matrix operator()(const Matrix &sequence);

	vector<Vector>& operator ()(const Tensor &sequence,
			const vector<Vector> &mask, vector<Vector> &y);

	Vector& operator ()(const Matrix &sequence, Vector &y);

	Matrix Wq, Wk, Wv, Wo;
	Vector bq, bk, bv, bo;

	int num_attention_heads;

	Tensor& scaled_dot_product_attention(Tensor &query, Tensor &key,
			Tensor &value, const Tensor &attention_mask,
			const vector<Vector> &mask);

	Tensor& scaled_dot_product_attention(Tensor &query, Tensor &key,
			Tensor &value, const vector<Vector> &mask);

	Tensor& scaled_dot_product_attention(Tensor &query, Tensor &key,
			Tensor &value);

	vector<Vector>& scaled_dot_product_attention(vector<Vector> &query,
			const Tensor &key, const Tensor &value, const vector<Vector> &mask);

	vector<Vector>& scaled_dot_product_attention(vector<Vector> &query,
			const Tensor &key, const Tensor &value);

	Tensor& reshape_to_batches(Tensor &x);
	Tensor reshape_to_batches(Matrix &x);

	vector<Vector>& reshape_to_batches(vector<Vector>&);
	vector<Vector> reshape_to_batches(Vector&);

	Tensor& reshape_from_batches(Tensor&);
	vector<Vector>& reshape_from_batches(vector<Vector>&);
};

struct PositionEmbedding {
	PositionEmbedding(KerasReader &dis, int num_attention_heads);

	Matrix embeddings;
	int num_attention_heads;

	Tensor& operator()(Tensor &sequence, const vector<int> &mid);
	Tensor& operator()(Tensor &sequence);
	Matrix& operator()(Matrix &sequence, int mid);
	Matrix& operator()(Matrix &sequence);
	Matrix compute_mask(vector<VectorI> &inputToken);
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
	vector<VectorI> operator()(const vector<VectorI> &token,
			vector<int> &inputMid);
	VectorI operator()(const VectorI &token, int inputMid);
};

struct BertEmbedding {
	BertEmbedding(KerasReader &dis, int num_attention_heads);

	Embedding wordEmbedding;
	Embedding segmentEmbedding;
	PositionEmbedding positionEmbedding;
	LayerNormalization layerNormalization;
	DenseLayer embeddingMapping;
	int embed_dim, hidden_size;

	Tensor operator ()(vector<VectorI> &inputToken, const vector<int> &inputMid,
			const vector<VectorI> &inputSegment, Matrix &mask);

	Matrix operator ()(VectorI &inputToken, int inputMid,
			const VectorI &inputSegment);

	Matrix operator ()(VectorI &inputToken, const VectorI &inputSegment);

	vector<Vector>& compute_mask(vector<VectorI> &inputToken);
};

struct NonSegmentedBertEmbedding {
	NonSegmentedBertEmbedding(KerasReader &dis, int num_attention_heads);

	Embedding wordEmbedding;
	PositionEmbedding positionEmbedding;
	LayerNormalization layerNormalization;
	DenseLayer embeddingMapping;
	int embed_dim, hidden_size;

	Tensor operator ()(vector<VectorI> &inputToken, const vector<int> &inputMid,
			const vector<VectorI> &inputSegment, vector<Vector> &mask);

	Matrix operator ()(VectorI &inputToken, int inputMid,
			const VectorI &inputSegment);

	Matrix operator ()(VectorI &inputToken, const VectorI &inputSegment);

	Matrix operator ()(const VectorI &inputToken);

	vector<Vector>& compute_mask(vector<VectorI> &inputToken);
};

struct Encoder {
	Encoder();
	Encoder(KerasReader &dis, int num_attention_heads, Activation hidden_act);

	::MultiHeadAttention MultiHeadAttention;
	LayerNormalization MultiHeadAttentionNorm;
	::FeedForward FeedForward;
	LayerNormalization FeedForwardNorm;

	Tensor& wrap_attention(Tensor &input_layer, const Tensor &attention_matrix,
			const vector<Vector> &mask);

	Tensor& wrap_attention(Tensor &input_layer, const vector<Vector> &mask);
	Matrix& wrap_attention(Matrix &input_layer);

	vector<Vector>& wrap_attention(Tensor &input_layer,
			const vector<Vector> &mask, vector<Vector> &y);

	Vector& wrap_attention(Matrix &input_layer, Vector &y);

	Tensor& wrap_feedforward(Tensor &input_layer);
	Matrix& wrap_feedforward(Matrix &input_layer);
	vector<Vector>& wrap_feedforward(vector<Vector> &input_layer);
	Vector& wrap_feedforward(Vector &input_layer);

	Tensor& operator ()(Tensor &input_layer, const Tensor &attention_matrix,
			const vector<Vector> &mask);
	Tensor& operator ()(Tensor &input_layer, const vector<Vector> &mask);
	Matrix& operator ()(Matrix &input_layer);

	Vector& operator ()(Matrix &input_layer, Vector &y);

	vector<Vector>& operator ()(Tensor &input_layer, const vector<Vector> &mask,
			vector<Vector> &y);
};

struct AlbertTransformer {
	AlbertTransformer(KerasReader &dis, int num_hidden_layers,
			int num_attention_heads, Activation hidden_act);
	int num_hidden_layers;
	Encoder encoder;

	Tensor& operator ()(Tensor &input_layer,
			const vector<MatrixI> &attention_matrix, RevertMask &fn,
			const vector<Vector> &mask);

	Tensor& operator ()(Tensor &input_layer, const Matrix &mask);

	Tensor& operator ()(Tensor &input_layer, const Tensor &attention_matrix,
			const vector<Vector> &mask);

	vector<Vector>& operator ()(Tensor &input_layer, const Matrix &mask,
			vector<Vector> &y);

	Vector& operator ()(Matrix &input_layer, Vector &y);

	vector<Vector>& operator ()(Tensor &input_layer,
			const vector<MatrixI> &attention_matrix, RevertMask &fn,
			const vector<Vector> &mask, vector<Vector> &y);

};

struct BertTransformer {
	BertTransformer(KerasReader &dis, int num_hidden_layers,
			int num_attention_heads,
			Activation hidden_act = { Activator::gelu });
	int num_hidden_layers;
	vector<Encoder> encoder;

	Encoder& operator [](int i);

	Tensor& operator ()(Tensor &input_layer,
			const vector<MatrixI> &attention_matrix, RevertMask &fn,
			const vector<Vector> &mask);

	Tensor& operator ()(Tensor &input_layer, const vector<Vector> &mask);

	Tensor& operator ()(Tensor &input_layer, const Tensor &attention_matrix,
			const vector<Vector> &mask);

	vector<Vector>& operator ()(Tensor &input_layer, const vector<Vector> &mask,
			vector<Vector> &y);

	Vector& operator ()(Matrix &input_layer, Vector &y);

	vector<Vector>& operator ()(Tensor &input_layer,
			const vector<MatrixI> &attention_matrix, RevertMask &fn,
			const vector<Vector> &mask, vector<Vector> &y);

};

struct FullTokenizer {
	//Runs end-to-end tokenziation."""

	FullTokenizer(const string &vocab_file, bool do_lower_case = true);
	//    """Runs WordPiece tokenziation."""

	dict<String, int> vocab;
	String unk_token;
	size_t max_input_chars_per_word;
	bool do_lower_case;

	dict<String, int> unknownSet;

	vector<String> basic_tokenize(const String &text);

	vector<String> _run_split_on_punc(String &text);

	bool _is_punctuation(word cp);

	bool _is_chinese_char(word cp);

	String& _clean_text(String &text);

	vector<String> wordpiece_tokenize(String &text);

	vector<String> tokenize(const String &text);

	vector<String> tokenize(const String &text, const String &_text);

	VectorI convert_tokens_to_ids(const vector<String> &items);

	static FullTokenizer& instance_cn();
	static FullTokenizer& instance_en();
};

struct Pairwise {
	Pairwise(KerasReader &dis, const string &vocab, int num_attention_heads,
			bool symmetric_position_embedding = true,
			int num_hidden_layers = 12);
	FullTokenizer tokenizer;
	bool symmetric_position_embedding;

	MidIndex midIndex;
	SegmentInput segmentInput;
//	::CrossAttentionMask CrossAttentionMask;
//	::RevertMask RevertMask;
	BertEmbedding bertEmbedding;

	AlbertTransformer transformer;
	DenseLayer poolerDense;
	DenseLayer similarityDense;

	vector<double> operator ()(vector<VectorI> &input_ids);
	double operator ()(VectorI &input_ids);
	double operator ()(const vector<String> &s);
	double operator ()(String &x, String &y);
	double operator ()(const char16_t *x, const char16_t *y);
	static Pairwise& paraphrase();
	static Pairwise& lexicon();
};

struct PairwiseVector {
	PairwiseVector(KerasReader &dis, Activation hidden_act,
			int num_attention_heads, int num_hidden_layers,
			Activation bilinear_act);
	NonSegmentedBertEmbedding bertEmbedding;

	AlbertTransformer transformer;
	Bilinear bilinear;

//	static double probability2score(const Vector &y_pred);
	static Vector& symmetric_transform(Vector &y_pred);
	Matrix operator ()(const vector<VectorI> &input_ids);
	Vector operator ()(const VectorI &input_ids);
	Vector operator ()(const VectorI &input_ids, const VectorI &input_ids1);

	Matrix operator ()(const vector<Vector> &s);

//	static const String& lexicon_label(const Vector &y_pred);
};

struct PairwiseVectorChar: PairwiseVector {
	PairwiseVectorChar(KerasReader &dis, const string &vocab,
			int num_attention_heads, int num_hidden_layers = 12);
	dict<String, int> word2id;
	vector<String> tokenize(const String &text);
	using PairwiseVector::operator ();
	Matrix operator ()(const vector<String> &s);
	Vector operator ()(const String &x, const String &y);
	Vector operator ()(const String &x);
	Vector operator ()(const char16_t *x, const char16_t *y);
	static PairwiseVectorChar& instance();
};

#include "../sentencepiece/sentencepiece_processor.h"
//#include "../json/json.h"

struct PairwiseVectorSP: PairwiseVector {
	PairwiseVectorSP(KerasReader &dis, int num_hidden_layers,
			sentencepiece::SentencePieceProcessor *tokenizer);
	using PairwiseVector::operator ();

	sentencepiece::SentencePieceProcessor *tokenizer;
	vector<string> tokenize(const string &text);
	Matrix operator ()(const vector<string> &s);
	Vector operator ()(const string &x, const string &y);
	Vector operator ()(const string &x);
	Vector operator ()(String &x);
	Vector operator ()(String &x, String &y);
	Vector operator ()(const char *x, const char *y);

	static PairwiseVectorSP& instance();

	static PairwiseVectorSP& initialize(const string &config,
			const string &path, const string &vocab);
};

vector<int> lexiconStructure(const vector<String> &keywords,
		const vector<int> &frequency);

vector<int> lexiconStructure(int lang, const vector<vector<double>> &embedding,
		const vector<vector<double>> &score_matrix,
		const vector<int> &frequency);

vector<int> lexiconStructure(const vector<string> &keywords,
		const vector<int> &frequency);

sentencepiece::SentencePieceProcessor& en_tokenizer();
