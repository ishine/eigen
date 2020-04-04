#include "utility.h"
#include "keras.h"

struct CWSTaggerLSTM {

	dict<String, int> word2id;
	Embedding embedding; //, repertoire_embedding;

	Conv1D con1D0, con1D1;
	BidirectionalLSTM lstm;
	Conv1D con1D2;

	CRF wCRF;

	VectorI& predict(VectorI &predict_text);
	String predict(const String &predict_text);

	vector<vector<vector<double>>>& _predict(const String &predict_text,
			vector<vector<vector<double>>> &arr);

	CWSTaggerLSTM(const string &h5FilePath, const string &vocabFilePath);
	CWSTaggerLSTM(HDF5Reader &dis, const string &vocabFilePath);

	static CWSTaggerLSTM& instance(bool reinitialize = false);
};

struct CWSTagger {

	dict<String, int> word2id;
	Embedding embedding; //, repertoire_embedding;

	LSTM lstm;
	CRF wCRF;

	VectorI& predict(VectorI &predict_text);
	String predict(const String &predict_text);

	vector<vector<vector<double>>>& _predict(const String &predict_text,
			vector<vector<vector<double>>> &arr);

	CWSTagger(const string &h5FilePath, const string &vocabFilePath);
	CWSTagger(HDF5Reader &dis, const string &vocabFilePath);

	static CWSTagger& instance(bool reinitialize = false);
};

