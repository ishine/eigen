#include "Embedding.h"
Matrix& Embedding::operator()(String &words, Matrix &wordEmbedding,
		size_t max_length) {
	if (max_length && words.length() > max_length) {
		words = words.substr(0, max_length);
	}
	return operator()(words, wordEmbedding);
}

Matrix& Embedding::operator()(const String &words, Matrix &wordEmbedding) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;

	int length = words.size();
//	cout << "length = " << length << endl;

	wordEmbedding.resize(length, wEmbedding.cols());

	for (int j = 0; j < wordEmbedding.rows(); ++j) {
		int index = 1;
		word ch = words[j];
		if (char2id.count(ch))
			index = char2id[ch];

		wordEmbedding.row(j) = wEmbedding.row(index);
	}
	return wordEmbedding;
}

Matrix& Embedding::call(const String &words, Matrix &wordEmbedding) {

	int length = words.size();
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	cout << "length = " << length << endl;

	wordEmbedding.resize(length, wEmbedding.cols());

	for (int j = 0; j < wordEmbedding.rows(); ++j) {
		int index = 1;
		word ch = words[j];

		if (char2id.count(ch))
			index = char2id[ch];

		wordEmbedding.row(j) = this->wEmbedding.row(index);
	}

	return wordEmbedding;
}

Matrix& Embedding::operator()(const vector<int> &words, Matrix &wordEmbedding) {

	int length = words.size();
	wordEmbedding.resize(length, wEmbedding.cols());

	for (int j = 0; j < wordEmbedding.rows(); ++j) {
		int index = words[j];
		wordEmbedding.row(j) = wEmbedding.row(index);
	}
	return wordEmbedding;
}

Matrix& Embedding::operator()(const vector<int> &words, Matrix &wordEmbedding, Matrix &wEmbedding) {
	wEmbedding = this->wEmbedding;
	return this->operator ()(words, wordEmbedding);
}

Matrix& Embedding::operator()(const String &words, Matrix &wordEmbedding, Matrix &wEmbedding) {
	wEmbedding = this->wEmbedding;
	return this->operator ()(words, wordEmbedding);
}

void Embedding::initialize(BinaryReader &dis, bool dic) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;

	if (dic) {
		dis.read(char2id);
	}

	dis.read(wEmbedding);
}
//
Embedding::Embedding(BinaryReader &dis, bool dic, bool return_embeddings) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	initialize(dis, dic);
}

Embedding::Embedding(unordered_map<word, int> &char2id, Matrix &wEmbedding) :
		char2id(char2id), wEmbedding(wEmbedding) {
}

Embedding::Embedding(BinaryReader &dis) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
//	cout << "in " << __func__ << endl;

	initialize(dis, true);
}
//
