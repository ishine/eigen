#include "classification.h"
#include "bert.h"
Vector Classifier::predict(const String &predict_text) {
	auto text = predict_text;
	return predict(text);
}

Vector Classifier::predict(String &predict_text) {
//	cout << "predict: " << predict_text << endl;
	Matrix embedding;

	this->embedding(string2id(tokenizer->tokenize(predict_text), word2id),
			embedding);

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
	auto x = predict(predict_text);
	int index;
	x.maxCoeff(&index);
	return index;
}

vector<vector<double>>& Classifier::predict(String &predict_text,
		vector<vector<double>> &arr) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
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
		const string &vocabFilePath, FullTokenizer *tokenizer) :
		Classifier((HDF5Reader&) (const HDF5Reader&) HDF5Reader(binaryFilePath),
				tokenizer) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	Text(vocabFilePath) >> word2id;
}

#include "lagacy.h"

Classifier::Classifier(HDF5Reader &dis, FullTokenizer *tokenizer) :
		embedding(Embedding(dis)), con1D0(dis), con1D1(dis), con1D2(dis), lstm(
				BidirectionalLSTM(dis, Bidirectional::sum)), dense_tanh(
				DenseLayer(dis)), dense_pred(
				DenseLayer(dis, true, Activator::softmax)), tokenizer(tokenizer) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
}

Classifier::Classifier(HDF5Reader &dis, const string &vocab,
		FullTokenizer *tokenizer) :
		Classifier(dis, tokenizer) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
}

Classifier& Classifier::qatype_classifier() {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	static Classifier service(cnModelsDirectory() + "qatype/model.h5",
			cnModelsDirectory() + "qatype/vocab.txt",
			&FullTokenizer::instance_cn());

	return service;
}

Classifier& Classifier::phatic_classifier() {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	static Classifier service(cnModelsDirectory() + "phatic/model.h5",
			cnModelsDirectory() + "phatic/vocab.txt",
			&FullTokenizer::instance_cn());

	return service;
}

Classifier& Classifier::keyword_cn_classifier(bool reinitialize) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
	static string h5FilePath = cnModelsDirectory() + "keyword/model.h5";
	static string vocab = cnModelsDirectory() + "keyword/vocab.txt";
	static Classifier service(h5FilePath, vocab, &FullTokenizer::instance_cn());

	if (reinitialize) {
		service = Classifier(h5FilePath, vocab, &FullTokenizer::instance_cn());
	}
	return service;
}

Classifier& Classifier::keyword_en_classifier(bool reinitialize) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
	static string h5FilePath = modelsDirectory() + "en/keyword/model.h5";
	static string vocab = modelsDirectory() + "en/keyword/vocab.txt";
	static Classifier service(h5FilePath, vocab, &FullTokenizer::instance_en());

	if (reinitialize) {
		service = Classifier(h5FilePath, vocab, &FullTokenizer::instance_en());
	}

	return service;
}

//reading .h5 with HDF5++
//https://portal.hdfgroup.org/display/support/HDF5%201.10.5
