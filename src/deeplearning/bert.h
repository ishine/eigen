#pragma once
#include "../std/utility.h"
#include<vector>
using std::vector;

#include "keras.h"

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

	Matrix operator()(const Matrix &sequence);

	Vector& operator ()(const Matrix &sequence, Vector &y);

	Matrix Wq, Wk, Wv, Wo;
	Vector bq, bk, bv, bo;

	int num_attention_heads;

	Tensor& scaled_dot_product_attention(Tensor &query, Tensor &key,
			Tensor &value);

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

	Matrix operator ()(VectorI &inputToken, int inputMid,
			const VectorI &inputSegment);

	Matrix operator ()(const VectorI &inputToken, const VectorI &inputSegment);

	Matrix operator ()(const VectorI &inputToken);
};

struct Encoder {
	Encoder();
	Encoder(KerasReader &dis, int num_attention_heads, Activation hidden_act);

	::MultiHeadAttention MultiHeadAttention;
	LayerNormalization MultiHeadAttentionNorm;
	::FeedForward FeedForward;
	LayerNormalization FeedForwardNorm;

	Matrix& wrap_attention(Matrix &input_layer);

	Vector& wrap_attention(Matrix &input_layer, Vector &y);

	Tensor& wrap_feedforward(Tensor &input_layer);
	Matrix& wrap_feedforward(Matrix &input_layer);
	vector<Vector>& wrap_feedforward(vector<Vector> &input_layer);
	Vector& wrap_feedforward(Vector &input_layer);

	Matrix& operator ()(Matrix &input_layer);

	Vector& operator ()(Matrix &input_layer, Vector &y);
};

struct AlbertTransformer {
	AlbertTransformer(KerasReader &dis, int num_hidden_layers,
			int num_attention_heads, Activation hidden_act);
	int num_hidden_layers;
	Encoder encoder;

	Vector& operator ()(Matrix &input_layer, Vector &y);
};

struct BertTransformer {
	BertTransformer(KerasReader &dis, int num_hidden_layers,
			int num_attention_heads,
			Activation hidden_act = { Activator::gelu });
	int num_hidden_layers;
	vector<Encoder> encoder;

	Encoder& operator [](int i);
	Vector& operator ()(Matrix &input_layer, Vector &y);
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

struct PretrainingAlbert {
	PretrainingAlbert(KerasReader &dis, Activation hidden_act,
			int num_attention_heads, int num_hidden_layers);
	BertEmbedding bertEmbedding;

	AlbertTransformer transformer;

	Vector operator ()(const VectorI &input_ids);
};

struct PretrainingAlbertChinese: PretrainingAlbert {
	PretrainingAlbertChinese(KerasReader &dis, int num_attention_heads,
			int num_hidden_layers = 12);
	vector<String> tokenize(const String &text);
	using PretrainingAlbert::operator ();
	Vector operator ()(const String &x);
	static PretrainingAlbertChinese& instance();
};

struct PretrainingAlbertEnglish: PretrainingAlbert {
	PretrainingAlbertEnglish(KerasReader &dis, int num_hidden_layers);
	using PretrainingAlbert::operator ();

	vector<string> tokenize(const string &text);
	Vector operator ()(const string &x);
	Vector operator ()(String &x);

	static PretrainingAlbertEnglish& instance();

	static PretrainingAlbertEnglish& initialize(const string &config,
			const string &path, const string &vocab);
};

