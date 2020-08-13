#include "utility.h"
#include "keras.h"

struct CWSTaggerLSTM {

	dict<char16_t, int> word2id;
	Embedding embedding; //, repertoire_embedding;

	Conv1D con1D0, con1D1;
	BidirectionalLSTM lstm;
	Conv1D con1D2;

	CRF wCRF;

	VectorI& predict(VectorI &predict_text);
	vector<String> predict(const String &predict_text);

	vector<vector<vector<double>>>& _predict(const String &predict_text,
			vector<vector<vector<double>>> &arr);

	CWSTaggerLSTM(const string &h5FilePath, const string &vocabFilePath);
	CWSTaggerLSTM(KerasReader &dis, const string &vocabFilePath);

	static CWSTaggerLSTM& instance(bool reinitialize = false);
};

struct CWSTagger {

	dict<char16_t, int> word2id;
	Embedding embedding; //, repertoire_embedding;

	Conv1DSame con1D;
	CRF wCRF;

	VectorI& predict(VectorI &predict_text);
	vector<String> predict(const String &predict_text);
	vector<vector<String>> predict(const vector<String> &predict_text);
	vector<vector<vector<String>>> predict(
			const vector<vector<String>> &predict_text);

	vector<vector<vector<double>>>& _predict(const String &predict_text,
			vector<vector<vector<double>>> &arr);

	CWSTagger(const string &h5FilePath, const string &vocabFilePath);
	CWSTagger(KerasReader &dis, const string &vocabFilePath);

	static CWSTagger& instance_crf();
};

