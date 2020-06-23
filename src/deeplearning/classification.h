#include "keras.h"

#include "matrix.h"

struct Classifier {
	dict<char16_t, int> word2id;
	Embedding embedding;
	Conv1D con1D0, con1D1, con1D2;
	BidirectionalLSTM lstm;
	DenseLayer dense_tanh, dense_pred;

	Vector predict(const String &predict_text);
	Vector predict(String &predict_text);
	Vector predict_debug(const String &predict_text);
	int predict(const String &predict_text, int &argmax);

	vector<vector<double>>& predict(String &predict_text,
			vector<vector<double>> &arr);

	Classifier(const string &binaryFilePath, const string &vocabFilePath);
	Classifier(KerasReader &dis);
	Classifier(KerasReader &dis, const string &vocab);

	vector<vector<vector<double>>>& weight(vector<vector<vector<double>>> &arr);

	static Classifier& phatic_classifier();
	static Classifier& qatype_classifier();
};

struct ClassifierChar {
	dict<char16_t, int> word2id;
	Embedding embedding;
	Conv1DSame con1D0, con1D1;
	BidirectionalGRU gru;
	DenseLayer dense_pred;

	Vector predict(const String &predict_text);
	Vector predict_debug(const String &predict_text);
	int predict(const String &predict_text, int &argmax);
	vector<int>& predict(const vector<String> &predict_text,
			vector<int> &argmax);

	vector<vector<double>>& predict(String &predict_text,
			vector<vector<double>> &arr);

	ClassifierChar(const string &binaryFilePath, const string &vocabFilePath);
	ClassifierChar(KerasReader &dis, const string &vocab);

	vector<vector<vector<double>>>& weight(vector<vector<vector<double>>> &arr);

	static ClassifierChar& instance();
};

#include "sentencepiece.h"

struct ClassifierWord {
	dict<string, int> word2id;
	Embedding embedding;
	Conv1DSame con1D0, con1D1;
	BidirectionalGRU gru;
	DenseLayer dense_pred;
	sentencepiece::SentencePieceProcessor *tokenizer;

	Vector predict(const string &predict_text);
	Vector predict(string &predict_text);
	Vector predict(String &predict_text);

	Vector predict_debug(const string &predict_text);
	Vector predict_debug(String &predict_text);

	int predict(const string &predict_text, int &argmax);
	int predict(String &predict_text, int &argmax);

	vector<int>& predict(const vector<string> &predict_text,
			vector<int> &argmax);

	vector<int>& predict(vector<String> &predict_text,
			vector<int> &argmax);

	vector<vector<double>>& predict(string &predict_text,
			vector<vector<double>> &arr);

	ClassifierWord(const string &binaryFilePath, const string &vocabFilePath);
	ClassifierWord(KerasReader &dis, const string &vocab);

	vector<vector<vector<double>>>& weight(vector<vector<vector<double>>> &arr);

	static ClassifierWord& instance();
};

