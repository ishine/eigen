#include "utility.h"
#include "keras.h"

struct CWSTagger {

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

	CWSTagger(HDF5Reader &dis, const string &vocabFilePath);

	static CWSTagger& instance();
	static void reinitialize();
};

