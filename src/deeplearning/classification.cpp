#include "classification.h"
#include "bert.h"
#include "../std/lagacy.h"

Vector Classifier::predict(const String &predict_text) {
	auto text = predict_text;
	return predict(text);
}

Vector ClassifierWord::predict(String &predict_text) {
	return predict(Text::unicode2utf(tolower(predict_text)));
}

Vector Classifier::predict(String &predict_text) {
//	cout << "predict: " << predict_text << endl;
	Matrix embedding;

	this->embedding(string2id(predict_text, word2id), embedding);

//	this->embedding(string2id(tokenizer->tokenize(predict_text), word2id),
//			embedding);

	Matrix lCNN;
	lCNN = con1D0(embedding, lCNN);
	lCNN = con1D1(lCNN, embedding);
	con1D2(lCNN, embedding);

	Vector x;
	lstm(embedding, x);

	dense_tanh(x);

	dense_pred(x);
//	cout << "probabilities: " << x << endl;
	return x;
}

Vector ClassifierChar::predict(const String &predict_text) {
//	cout << "predict: " << predict_text << endl;
	Matrix embedding;

	this->embedding(string2id(predict_text, word2id), embedding);

//	printf("embedding.shape = (%lld * %lld)\n", embedding.rows(), embedding.cols());

	Matrix lCNN;
	lCNN = con1D0(embedding, lCNN);
	lCNN = con1D1(lCNN, embedding);
//	printf("lCNN.shape = (%lld * %lld)\n", lCNN.rows(), lCNN.cols());
	Vector lGRU;
	gru(lCNN, lGRU);

//	printf("lGRU.shape = (%lld * %lld)\n", lGRU.rows(), lGRU.cols());
	return dense_pred(lGRU);
}

Vector ClassifierChar::predict_debug(const String &predict_text) {
	cout << "predict: " << predict_text << endl;
	Matrix embedding;
	auto ids = string2id(predict_text, word2id);
	cout << "ids = " << ids << endl;

	this->embedding(ids, embedding);

	cout << "embedding.shape = (" << embedding.rows() << " * "
			<< embedding.cols() << ")" << endl;

	cout << "embedding = " << embedding << endl;

	Matrix lCNN;
	lCNN = con1D0(embedding, lCNN);

	cout << "lCNN.shape = (" << lCNN.rows() << " * " << lCNN.cols() << ")"
			<< endl;
	cout << "lCNN = " << lCNN << endl;

	lCNN = con1D1(lCNN, embedding);
	cout << "lCNN.shape = (" << lCNN.rows() << " * " << lCNN.cols() << ")"
			<< endl;
	cout << "lCNN = " << lCNN << endl;

	Vector lGRU;
	gru(lCNN, lGRU);

	cout << "lGRU.shape = (" << lGRU.rows() << " * " << lGRU.cols() << ")"
			<< endl;
	cout << "lGRU = " << lGRU << endl;

//	printf("lGRU.shape = (%lld * %lld)\n", lGRU.rows(), lGRU.cols());
	return dense_pred(lGRU);
}

Vector ClassifierWord::predict(const string &predict_text) {
//	cout << "predict: " << predict_text << endl;
	Matrix embedding;

	this->embedding(string2id(tokenizer->EncodeAsPieces(predict_text), word2id),
			embedding);

	Matrix lCNN;
	lCNN = con1D0(embedding, lCNN);
	lCNN = con1D1(lCNN, embedding);
	Vector lGRU;
	return dense_pred(gru(lCNN, lGRU));
}

Vector ClassifierWord::predict_debug(String &predict_text) {
	return predict_debug(Text::unicode2utf(tolower(predict_text)));
}

Vector ClassifierWord::predict_debug(const string &predict_text) {
//	cout << "predict: " << predict_text << endl;
	Matrix embedding;
	auto tokens = tokenizer->EncodeAsPieces(predict_text);
//	cout << "tokens = " << tokens << endl;
	auto ids = string2id(tokens, word2id);
//	cout << "ids = " << ids << endl;

	this->embedding(ids, embedding);

//	print_shape(embedding)
//	__cout(embedding);

	Matrix lCNN;
	lCNN = con1D0(embedding, lCNN);

//	print_shape(lCNN)
//	__cout(lCNN);

	lCNN = con1D1(lCNN, embedding);

	Vector lGRU;
	gru(lCNN, lGRU);

	return dense_pred(lGRU);
}

Vector Classifier::predict_debug(const String &predict_text) {
	Matrix embedding;

	auto inputs = string2id(predict_text, word2id);
	cout << "inputs = \n" << inputs << endl;

	this->embedding(inputs, embedding);
	cout << "lEmbedding = \n" << embedding << endl;

	Matrix lCNN;
	lCNN = con1D0(embedding, lCNN);
	cout << "lCNN0 = \n" << lCNN << endl;

	lCNN = con1D1(lCNN, embedding);
	cout << "lCNN1 = \n" << lCNN << endl;

	lCNN = con1D2(lCNN, embedding);
	cout << "lCNN2 = \n" << lCNN << endl;

	Vector x;
	lstm(lCNN, x);
	cout << "lLSTM = \n" << x << endl;

	dense_tanh(x);
	cout << "lDense0 = \n" << x << endl;

	dense_pred(x);
	cout << "lDense = \n" << x << endl;

	return x;
}

int Classifier::predict(const String &predict_text, int &argmax) {
	predict(predict_text).maxCoeff(&argmax);
	return argmax;
}

int ClassifierChar::predict(const String &predict_text, int &argmax) {
//	__cout(__PRETTY_FUNCTION__)
	predict(predict_text).maxCoeff(&argmax);
//	cout << "argmax = " << argmax << endl;
	return argmax;
}

vector<int>& ClassifierChar::predict(const vector<String> &predict_text,
		vector<int> &argmax) {
//	__cout(__PRETTY_FUNCTION__)
	auto size = predict_text.size();
	argmax.resize(size);
#pragma omp parallel for
	for (size_t i = 0; i < size; ++i) {
		predict(predict_text[i], argmax[i]);
	}

//	cout << "argmax = " << argmax << endl;
	return argmax;
}

int ClassifierWord::predict(const string &predict_text, int &argmax) {
	predict(predict_text).maxCoeff(&argmax);
	return argmax;
}

int ClassifierWord::predict(String &predict_text, int &argmax) {
	return predict(Text::unicode2utf(tolower(predict_text)), argmax);
}

vector<int>& ClassifierWord::predict(vector<String> &predict_text,
		vector<int> &argmax) {
	auto size = predict_text.size();
	argmax.resize(size);
//#pragma omp parallel for num_threads(cpu_count)
#pragma omp parallel for
	for (size_t i = 0; i < size; ++i) {
		predict(predict_text[i], argmax[i]);
	}
	return argmax;

}

vector<int>& ClassifierWord::predict(const vector<string> &predict_text,
		vector<int> &argmax) {
	auto size = predict_text.size();
	argmax.resize(size);
//#pragma omp parallel for num_threads(cpu_count)
#pragma omp parallel for
	for (size_t i = 0; i < size; ++i) {
		predict(predict_text[i], argmax[i]);
	}
	return argmax;
}

vector<vector<double>>& Classifier::predict(String &predict_text,
		vector<vector<double>> &arr) {
	__cout(__PRETTY_FUNCTION__)
	Matrix embedding;
	this->embedding(string2id(predict_text, word2id), embedding);

	arr.push_back(convert2vector(embedding, 0));
	arr.push_back(convert2vector(embedding, embedding.rows() - 1));

	Vector x;
	x = lstm(embedding, x, arr);
	arr.push_back(convert2vector(x));

	x = dense_tanh(x);
	arr.push_back(convert2vector(x));

	x = l2_normalize(x);
	arr.push_back(convert2vector(x));

	x = dense_pred(x);
	arr.push_back(convert2vector(x));
//	cout << arr[7] << endl;

	int index;
	x.maxCoeff(&index);

	return arr;
}

Classifier::Classifier(const string &binaryFilePath,
		const string &vocabFilePath) :
		Classifier(
				(KerasReader&) (const KerasReader&) KerasReader(binaryFilePath),
				vocabFilePath) {
	__cout(__PRETTY_FUNCTION__)
}

ClassifierChar::ClassifierChar(const string &binaryFilePath,
		const string &vocabFilePath) :
		ClassifierChar(
				(KerasReader&) (const KerasReader&) KerasReader(binaryFilePath),
				vocabFilePath) {
	__log(__PRETTY_FUNCTION__)
}

ClassifierWord::ClassifierWord(const string &binaryFilePath,
		const string &vocabFilePath) :
		ClassifierWord(
				(KerasReader&) (const KerasReader&) KerasReader(binaryFilePath),
				vocabFilePath) {
	__log(__PRETTY_FUNCTION__)
}

Classifier::Classifier(KerasReader &dis) :
		embedding(Embedding(dis)), con1D0(dis), con1D1(dis), con1D2(dis), lstm(
				BidirectionalLSTM(dis, Bidirectional::sum)), dense_tanh(
				DenseLayer(dis)), dense_pred(
				DenseLayer(dis, Activator::softmax)) {
	__cout(__PRETTY_FUNCTION__)
}

Classifier::Classifier(KerasReader &dis, const string &vocab) :
		word2id(Text(vocab).read_vocab_char()), embedding(Embedding(dis)), con1D0(
				dis), con1D1(dis), con1D2(dis), lstm(
				BidirectionalLSTM(dis, Bidirectional::sum)), dense_tanh(
				DenseLayer(dis)), dense_pred(
				DenseLayer(dis, Activator::softmax)) {
	__cout(__PRETTY_FUNCTION__)
}

ClassifierChar::ClassifierChar(KerasReader &dis, const string &vocab) :
		word2id(Text(vocab).read_vocab_char()), embedding(dis), con1D0(dis), con1D1(
				dis), gru(dis, Bidirectional::sum), dense_pred(dis,
				Activator::softmax) {
	__cout(__PRETTY_FUNCTION__)
}

ClassifierWord::ClassifierWord(KerasReader &dis, const string &vocab) :
		word2id(Text(vocab).read_vocab_cstr()), embedding(dis), con1D0(dis), con1D1(
				dis), gru(dis, Bidirectional::sum), dense_pred(dis,
				Activator::softmax), tokenizer(&en_tokenizer()) {
	__cout(__PRETTY_FUNCTION__)
}

Classifier& Classifier::qatype_classifier() {
	__cout(__PRETTY_FUNCTION__)
	static Classifier service(modelsDirectory() + "cn/qatype/model.h5",
			modelsDirectory() + "cn/qatype/vocab.txt");

	return service;
}

Classifier& Classifier::phatic_classifier() {
	__cout(__PRETTY_FUNCTION__)
	static Classifier service(modelsDirectory() + "cn/phatic/model.h5",
			modelsDirectory() + "cn/phatic/vocab.txt");

	return service;
}


ClassifierChar& ClassifierChar::instance() {
	__cout(__PRETTY_FUNCTION__)
	static ClassifierChar service(modelsDirectory() + "cn/keyword/model.h5", modelsDirectory() + "cn/keyword/vocab.txt");

	return service;
}

ClassifierWord& ClassifierWord::instance() {
	__cout(__PRETTY_FUNCTION__)
	static ClassifierWord service(modelsDirectory() + "en/keyword/model.h5", modelsDirectory() + "en/bert/albert_base/30k-clean.vocab");

	return service;
}

//reading .h5 with HDF5++
//https://portal.hdfgroup.org/display/support/HDF5%201.10.5
