#include "keras.h"

#include "matrix.h"
#include "bert.h"

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

	static ClassifierChar& keyword_cn_classifier();
	static void instantiate_keyword_cn_classifier();

};

struct ClassifierWord {
	dict<String, int> word2id;
	Embedding embedding;
	Conv1DSame con1D0, con1D1;
	BidirectionalGRU gru;
	DenseLayer dense_pred;
	FullTokenizer *tokenizer;

	Vector predict(const String &predict_text);
	Vector predict(String &predict_text);
	Vector predict_debug(const String &predict_text);
	int predict(const String &predict_text, int &argmax);
	vector<int>& predict(const vector<String> &predict_text,
			vector<int> &argmax);

	vector<vector<double>>& predict(String &predict_text,
			vector<vector<double>> &arr);

	ClassifierWord(const string &binaryFilePath, const string &vocabFilePath,
			FullTokenizer *tokenizer);
	ClassifierWord(KerasReader &dis, const string &vocab,
			FullTokenizer *tokenizer);

	vector<vector<vector<double>>>& weight(vector<vector<vector<double>>> &arr);

	static ClassifierWord& keyword_en_classifier();
	static void instantiate_keyword_en_classifier();
};

