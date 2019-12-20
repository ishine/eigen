#include "keras.h"

#include "matrix.h"

struct Classifier {

	unordered_map<String, int> word2id;
	Embedding embedding;
	Conv1D con1D0, con1D1, con1D2;
	BidirectionalLSTM lstm;
	DenseLayer dense_tanh, dense_pred;

	Vector& predict(const String &predict_text);
	Vector& predict_debug(const String &predict_text);
	int predict(const String &predict_text, int &argmax);

	vector<vector<double>>& predict(String &predict_text,
			vector<vector<double>> &arr);

	Classifier(const string &binaryFilePath, const string &vocabFilePath);
	Classifier(BinaryReader &dis);
	Classifier(BinaryReader &dis, const string &vocab);

	vector<vector<vector<double>>>& weight(vector<vector<vector<double>>> &arr);

	static Classifier& phatic_classifier();
	static Classifier& qatype_classifier();
};

