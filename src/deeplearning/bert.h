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
	Vector operator()(const Vector &x);

	Matrix& operator()(Matrix &x, Matrix &wDense);
	Matrix operator()(const Matrix &x);

	Tensor operator()(const Tensor &x);
	vector<Vector> operator()(const vector<Vector> &x);
	FeedForward(HDF5Reader &dis, bool bias = true);
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
	LayerNormalization(HDF5Reader &dis);
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
	MultiHeadAttention(HDF5Reader &dis, int num_attention_heads);
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
	PositionEmbedding(HDF5Reader &dis, int num_attention_heads);

	Matrix embeddings;
	int num_attention_heads;

	Tensor& operator()(Tensor &sequence, const vector<int> &mid);
	Tensor& operator()(Tensor &sequence);
	Matrix& operator()(Matrix &sequence, int mid);
	Matrix& operator()(Matrix &sequence);
	vector<Vector> compute_mask(vector<VectorI> &inputToken);
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
	BertEmbedding(HDF5Reader &dis, int num_attention_heads,
			bool factorization_on_word_embedding_only);

	bool factorization_on_word_embedding_only;

	Embedding wordEmbedding;
	Embedding segmentEmbedding;
	PositionEmbedding positionEmbedding;
	LayerNormalization layerNormalization;
	DenseLayer embeddingMapping;
	int embed_dim, hidden_size;

	Tensor& operator ()(vector<VectorI> &inputToken,
			const vector<int> &inputMid, const vector<VectorI> &inputSegment,
			vector<Vector> &mask);

	Matrix operator ()(VectorI &inputToken, int inputMid,
			const VectorI &inputSegment);

	Matrix operator ()(VectorI &inputToken, const VectorI &inputSegment);

	vector<Vector>& compute_mask(vector<VectorI> &inputToken);

	bool factorization(bool word_embedding_only = true);
};

struct Encoder {
	Encoder(HDF5Reader &dis, int num_attention_heads);
	Encoder();
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

struct Transformer {
	Transformer(HDF5Reader &dis, bool cross_layer_parameter_sharing,
			int num_hidden_layers, int num_attention_heads);
	int num_hidden_layers;
	object<Encoder> encoder;
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

vector<String> whitespace_tokenize(String &text);

struct BasicTokenizer {
//    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

	BasicTokenizer(bool do_lower_case = true);
//        """Constructs a BasicTokenizer.
//        Args:
	bool do_lower_case;

	vector<String> tokenize(String &text);

	String& _run_strip_accents(String &text);

	vector<String> _run_split_on_punc(String &text);

	String _tokenize_chinese_chars(const String &text);

	bool _is_punctuation(word cp);

	bool _is_chinese_char(word cp);

	String& _clean_text(String &text);
};

struct WordpieceTokenizer {
//    """Runs WordPiece tokenziation."""
	dict<String, int> vocab;
	String unk_token;
	size_t max_input_chars_per_word;
	dict<String, int> unknownSet;
	WordpieceTokenizer(const string &vocab_file);

	static dict<String, int> load_vocab(const string &vocab_file);

	WordpieceTokenizer(dict<String, int> vocab, String unk_token = u"[UNK]",
	size_t max_input_chars_per_word = 200);

	vector<String> tokenize(String &text);
//    vector<String> unknown_words(){
//        cout << "unknown characters:" << endl;
//        items = [*unknownSet.items()];
//        items.sort(key=lambda xy: xy[1], reverse=true);
//        for (key, repetition in items){
//            printf("%s = %s\n", key, repetition);
////#             print('%s = %s' % (key, repetition), file='report.txt')
//        }
//        return [key for key, _ in items];
//    }
};

struct FullTokenizer: BasicTokenizer, WordpieceTokenizer {
	//Runs end-to-end tokenziation."""

	FullTokenizer(const string &vocab_file, bool do_lower_case = true);

	vector<String> tokenize(String &text);

	VectorI convert_tokens_to_ids(vector<String> &items);
};

struct Paraphrase {
	Paraphrase(HDF5Reader &dis, const string &vocab, int num_attention_heads,
			bool factorization_on_word_embedding_only = true,
			bool cross_layer_parameter_sharing = true,
			bool symmetric_positional_embedding = true, int num_hidden_layers =
					12);
	FullTokenizer tokenizer;
	bool symmetric_positional_embedding;

	MidIndex midIndex;
	SegmentInput segmentInput;
//	::CrossAttentionMask CrossAttentionMask;
//	::RevertMask RevertMask;
	BertEmbedding bertEmbedding;

	Transformer transformer;
	DenseLayer poolerDense;
	DenseLayer similarityDense;

	vector<double> operator ()(vector<VectorI> &input_ids);
	double operator ()(VectorI &input_ids);

	double operator ()(String &x, String &y);
	double operator ()(const char16_t *x, const char16_t *y);
	static Paraphrase& instance();
};

