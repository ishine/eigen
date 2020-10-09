#include "bert.h"
#include "matrix.h"
#include "../std/lagacy.h"
#include "utility.h"

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
	__debug(x);
	print_shape(x)
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
	__debug(__PRETTY_FUNCTION__);

	dis >> gamma;
	dis >> beta;
}

const double LayerNormalization::epsilon = 1e-12;

VectorI MidIndex::operator()(const MatrixI &input_ids) {
	int batch_size = input_ids.size();
	VectorI res;
	res.resize(batch_size);
	int seq_len = input_ids[0].size();

	for (int k = 0; k < batch_size; ++k) {
		for (int i = 0; i < seq_len; ++i) {
			if (input_ids[k][i] == SEP) {
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
		if (input_ids[i] == SEP) {
			return i + 1;
		}
	}
	return -1;
}

MidIndex::MidIndex(int SEP) {
	this->SEP = SEP;
	__debug(__PRETTY_FUNCTION__);

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
		Tensor &key, Tensor &value) {
	Tensor &e = batch_dot(query, key, true);

	e /= sqrt(key[0].cols());

	Tensor &a = softmax(e);
	return batch_dot(a, value);
}

vector<Vector>& MultiHeadAttention::scaled_dot_product_attention(
		vector<Vector> &query, const Tensor &key, const Tensor &value) {
	__debug(query.size());
	__debug(query[0].size());
	print_tensor(key);
	print_tensor(value);
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
	__debug(__PRETTY_FUNCTION__);

	dis >> Wq;
	dis >> bq;

	dis >> Wk;
	dis >> bk;

	dis >> Wv;
	dis >> bv;

	dis >> Wo;
	dis >> bo;
}

Tensor& PositionEmbedding::operator ()(Tensor &sequence, const VectorI &mid) {
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

PositionEmbedding::PositionEmbedding(KerasReader &dis, int num_attention_heads) :
		embeddings(dis.read_matrix()), num_attention_heads(num_attention_heads) {
}

MatrixI SegmentInput::operator ()(const MatrixI &inputToken,
		VectorI &inputMid) {
	int batch_size = inputToken.size();
	int length = inputToken[0].size();
	MatrixI inputSegment;
	inputSegment.resize(batch_size);

	for (int k = 0; k < batch_size; ++k) {
		inputSegment[k].resize(length);
		int mid = inputMid[k];
		stosd(&inputSegment[k][0], 0, mid);
		stosd(&inputSegment[k][mid], 1, length - mid);
	}
	return inputSegment;
}

VectorI SegmentInput::operator ()(const VectorI &inputToken, int mid) {
	int length = inputToken.size();
//	cout << "inputToken.size() = " << inputToken.size() << endl;
	VectorI inputSegment(length);

//	cout << "inputSegment.size() = " << inputSegment.size() << endl;

//	cout << "inputSegment(0) = " << inputSegment(0) << endl;

//	cout << "inputSegment(mid) = " << inputSegment(mid) << endl;

	stosd(&inputSegment[0], 0, mid);

	stosd(&inputSegment[mid], 1, length - mid);

	return inputSegment;
}

BertEmbedding::BertEmbedding(KerasReader &dis, int num_attention_heads) :
		wordEmbedding(dis),

		segmentEmbedding(dis),

		positionEmbedding(dis, num_attention_heads),

		layerNormalization(dis),

		embeddingMapping(dis, Activator::linear) {
	__debug(__PRETTY_FUNCTION__);

	embed_dim = wordEmbedding.wEmbedding.cols();
	hidden_size = embeddingMapping.weight.cols();
}

Matrix BertEmbedding::operator ()(VectorI &input_ids, int inputMid,
		const VectorI &inputSegment) {
	auto embeddings = wordEmbedding(input_ids);

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

Matrix BertEmbedding::operator ()(const VectorI &input_ids,
		const VectorI &inputSegment) {
	auto embeddings = wordEmbedding(input_ids);

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

Matrix BertEmbedding::operator ()(const VectorI &input_ids) {
	return (*this)(input_ids, VectorI(input_ids.size()));
}

Encoder::Encoder(KerasReader &dis, int num_attention_heads,
		Activation hidden_act) :
		MultiHeadAttention(dis, num_attention_heads), MultiHeadAttentionNorm(
				dis), FeedForward(dis, hidden_act), FeedForwardNorm(dis) {
}

Matrix& Encoder::wrap_attention(Matrix &input_layer) {
	input_layer += MultiHeadAttention(input_layer);
	return MultiHeadAttentionNorm(input_layer);
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

Matrix& Encoder::operator ()(Matrix &input_layer) {
	auto &inputs = wrap_attention(input_layer);
	return wrap_feedforward(inputs);
}

Encoder::Encoder() {
}

Vector& Encoder::operator ()(Matrix &input_layer, Vector &y) {
	auto &inputs = wrap_attention(input_layer, y);
	return wrap_feedforward(inputs);
}

AlbertTransformer::AlbertTransformer(KerasReader &dis, int num_hidden_layers,
		int num_attention_heads, Activation hidden_act) :
		num_hidden_layers(num_hidden_layers), encoder(dis, num_attention_heads,
				hidden_act) {
	__debug(__PRETTY_FUNCTION__);

}

BertTransformer::BertTransformer(KerasReader &dis, int num_hidden_layers,
		int num_attention_heads, Activation hidden_act) :
		num_hidden_layers(num_hidden_layers), encoder(num_hidden_layers) {
	__debug(__PRETTY_FUNCTION__);

	for (int i = 0; i < num_hidden_layers; ++i) {
		encoder[i] = Encoder(dis, num_attention_heads, hidden_act);
	}
}

Encoder& BertTransformer::operator [](int i) {
	return encoder[i];
}

Vector& AlbertTransformer::operator ()(Matrix &input_layer, Vector &y) {
	auto &last_layer = input_layer;
	for (int i = 0; i < num_hidden_layers; ++i) {
//		__debug(last_layer);
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
			__debug(__PRETTY_FUNCTION__);
		}

//bool cross_layer_parameter_sharing = true;
PretrainingAlbert::PretrainingAlbert(KerasReader &dis, Activation hidden_act,
		int num_attention_heads, int num_hidden_layers) :
		bertEmbedding(dis, num_attention_heads),

		transformer(dis, num_hidden_layers, num_attention_heads, hidden_act) {
	__log(__PRETTY_FUNCTION__);

}

PretrainingAlbertChinese::PretrainingAlbertChinese(KerasReader &dis,
		int num_attention_heads, int num_hidden_layers) :
		PretrainingAlbert(dis, { Activator::relu }, num_attention_heads,
				num_hidden_layers) {
	__log(__PRETTY_FUNCTION__);

}

PretrainingAlbertEnglish::PretrainingAlbertEnglish(KerasReader &dis,
		int num_hidden_layers) :
		PretrainingAlbert(dis, { Activator::gelu },
		/*num_attention_heads = 12*/12, num_hidden_layers) {
	__log(__PRETTY_FUNCTION__);

}

#include "../json/json.h"
Json::Value readFromStream(const string &json_file);

Pairwise& Pairwise::paraphrase() {
	static const auto &config = readFromStream(
			weightsDirectory() + "cn/paraphrase/config.json");

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
					weightsDirectory() + "cn/paraphrase/model.h5"),
			weightsDirectory() + "cn/bert/vocab.txt", num_attention_heads,
//			cross_layer_parameter_sharing,
			symmetric_position_embedding, num_hidden_layers);
	__debug(__PRETTY_FUNCTION__);

	return inst;
}

PretrainingAlbertChinese& PretrainingAlbertChinese::instance() {
	static PretrainingAlbertChinese inst(
			(KerasReader&) (const KerasReader&) KerasReader(
					weightsDirectory() + "cn/pretraining/model.h5"),

			12, //num_attention_heads = 12
			readFromStream(weightsDirectory() + "cn/pretraining/config.json")["num_hidden_layers_for_prediction"].asInt());

	return inst;
}

PretrainingAlbertEnglish& PretrainingAlbertEnglish::instance() {
	__debug(__PRETTY_FUNCTION__);

	static PretrainingAlbertEnglish inst(
			(KerasReader&) (const KerasReader&) KerasReader(
					weightsDirectory() + "en/pretraining/model.h5"),

			readFromStream(weightsDirectory() + "en/pretraining/config.json")["num_hidden_layers_for_prediction"].asInt());

	return inst;
}

Pairwise& Pairwise::lexicon() {
	static const auto &config = readFromStream(
			weightsDirectory() + "cn/lexicon_pairwise/config.json");

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
					weightsDirectory() + "cn/lexicon_pairwise/model.h5"),
			weightsDirectory() + "cn/bert/vocab.txt", num_attention_heads,
//			cross_layer_parameter_sharing,
			symmetric_position_embedding, num_hidden_layers);
//	__debug(__PRETTY_FUNCTION__);
	return inst;
}

double Pairwise::operator ()(VectorI &input_ids) {
	auto inputMid = midIndex(input_ids);
	auto inputSegment = segmentInput(input_ids, inputMid);

	auto embed_layer =
			symmetric_position_embedding ?
					bertEmbedding(input_ids, inputMid, inputSegment) :
					bertEmbedding(input_ids, inputSegment);

	Vector clsEmbedding;
	transformer(embed_layer, clsEmbedding);

	auto &sent = poolerDense(clsEmbedding);

	sent = similarityDense(sent);

	return sent(0);
}

Vector PretrainingAlbert::operator ()(const VectorI &input_ids) {
//	__log(input_ids);

	auto embed_layer = bertEmbedding(input_ids);

//	__debug(embed_layer);

	Vector clsEmbedding;
	transformer(embed_layer, clsEmbedding);

	return clsEmbedding;
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

vector<String> PretrainingAlbertChinese::tokenize(const String &text) {
	vector<String> s_x;
	s_x.push_back(u"[CLS]");
	s_x += FullTokenizer::instance_cn().tokenize(text);
	s_x.push_back(u"[SEP]");
	return s_x;
}

#include "sentencepiece.h"

vector<string> PretrainingAlbertEnglish::tokenize(const string &text) {
	vector<string> s_x;
	s_x.push_back("[CLS]");
	s_x += en_tokenizer().EncodeAsPieces(text);
	s_x.push_back("[SEP]");
	return s_x;
}

Vector PretrainingAlbertChinese::operator ()(const String &str) {
	return (*this)(
			FullTokenizer::instance_cn().convert_tokens_to_ids(tokenize(str)));
}

Vector PretrainingAlbertEnglish::operator ()(String &str) {
	return (*this)(Text::unicode2utf(tolower(str)));
}

Vector PretrainingAlbertEnglish::operator ()(const string &str) {
	return (*this)(en_tokenizer().PieceToId(tokenize(str)));
}

double Pairwise::operator ()(const char16_t *_x, const char16_t *_y) {
	String x = _x;
	String y = _y;
	cout << "first sentence: " << x << endl;
	cout << "second sentence: " << y << endl;
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
			__debug(__PRETTY_FUNCTION__);
		}

FullTokenizer& FullTokenizer::instance_cn() {
	static FullTokenizer instance(weightsDirectory() + "cn/bert/vocab.txt");
	return instance;
}

FullTokenizer& FullTokenizer::instance_en() {
	static FullTokenizer instance(weightsDirectory() + "en/bert/vocab.txt");
	return instance;
}

VectorI FullTokenizer::convert_tokens_to_ids(const vector<String> &items) {
	VectorI output(items.size());

	int index = 0;
	for (auto &item : items) {
		auto iter = vocab.find(item);
//		output(index) = iter == vocab.end() ? -vocab.at(lstrip(item)) : iter->second;
		output[index] = iter == vocab.end() ? 1 : iter->second;

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
		output_tokens.push_back(unk_token);
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

			output_tokens.push_back(cur_substr);
			start = end;
		}

		if (is_bad)
			output_tokens.push_back(unk_token);
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
			output += { String(1, ch) };
			start_new_word = true;
		} else {
			if (start_new_word)
				output += { u""};
			start_new_word = false;
			output.back() += ch;
		}
		++i;
	}

	return output;
}

vector<String> FullTokenizer::tokenize(const String &x, const String &y) {
	vector<String> s;
	s += { u"[CLS]"};
	s += tokenize(x);
	s += { u"[SEP]"};
	s += tokenize(y);
	s += { u"[SEP]"};
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
				output += wordpiece_tokenize(word);
			}
			continue;
		}

		if (_is_chinese_char(ch) || _is_punctuation(ch)) {
			if (!!word) {
				//		cout << "word = " << word << endl;
				output += wordpiece_tokenize(word);
			}

			String substr(1, ch);
			if (vocab.count(substr))
				output.push_back(substr);
			else
				output.push_back(this->unk_token);
		} else {
			word += ch;
		}
	}

	if (!!word) {
		//	cout << "word = " << word << endl;
		output += wordpiece_tokenize(word);
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
			__debug(__PRETTY_FUNCTION__);
		}

		bool operator ()(int x, int y) {
			return priority_of_cluster[x] < priority_of_cluster[y];
		}
	};

	ClusteringAlgorithm(Matrix &scores, const VectorI &frequency,
//			int maxNumOfClusters,
			int maxNumOfChildren) :
			scores(scores),

			n(scores.rows()),

//			maxNumOfClusters(maxNumOfClusters),

			maxNumOfChildren(std::min(maxNumOfChildren, (int) sqrt(2 * n))),

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
		__debug(num_of_children);

		double max_frequency = frequency[0];
		for (int parent = 0; parent < n; ++parent) {
			double term_weight = frequency[parent] / max_frequency;
			priority_of_cluster[parent] += 2 * term_weight * term_weight;

			pq.insert(parent);
		}
	}

	Matrix &scores;
	int n, maxNumOfChildren;
	VectorI heads;
	VectorI num_of_children;
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

			__debug(parent);
			__debug(num_of_children);
			__debug(priority_of_cluster);
			if (parent < 0)
				continue;

			int numOfChildren = num_of_children[parent];
			__debug(numOfChildren);
			__debug(priority_of_cluster[parent]);

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

			if (numOfChildren > maxNumOfChildren) {
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
		int unwanted_child = -1;
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

VectorI lexiconStructure(Matrix &scores, const VectorI &frequency,
		int maxNumOfChildren) {
	ClusteringAlgorithm cluster(scores, frequency, maxNumOfChildren);
	cluster.run();
	assert(cluster.sanctity_check());
	return cluster.heads;
}
