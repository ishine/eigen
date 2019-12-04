#pragma once
#include "Utility.h"

struct Embedding {
	unordered_map<word, int> char2id;
	Matrix wEmbedding;
	Matrix& call(const String &word, Matrix&wordEmbedding);

	Matrix& operator()(String &word, Matrix &wordEmbedding, size_t max_length);

	Matrix& operator()(const String &word, Matrix &wordEmbedding);

	Matrix& operator()(const String &word, Matrix &wordEmbedding, Matrix &wEmbedding);

	Matrix& operator()(const vector<int> &word, Matrix &wordEmbedding);

	Matrix& operator()(const vector<int> &word, Matrix &wordEmbedding, Matrix &wEmbedding);

	void initialize(BinaryReader &dis, bool dic);


	Embedding(BinaryReader &dis);
	Embedding(unordered_map<word, int> &char2id, Matrix &wEmbedding);
	Embedding(BinaryReader &dis, bool dic, bool return_embeddings = false);
};
