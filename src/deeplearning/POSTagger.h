#include "utility.h"
#include "keras.h"

struct POSTagger {
	vector<String> convertToPOStags(const VectorI &ids);
	vector<String> posTags;
	dict<char16_t, int> word2id;
	Embedding embedding;
	BidirectionalGRU gru;
	BidirectionalLSTM lstm0, lstm1, lstm2;
	CRF wCRF;

	VectorI predict(const vector<VectorI> &predict_text);
	vector<String> predict(const vector<String> &predict_text);
	vector<vector<String>> predict(const vector<vector<String>> &predict_text);

	POSTagger(const string &h5FilePath, const string &vocabFilePath,
			const string &posTagsFilePath);
	POSTagger(KerasReader &dis, const string &vocabFilePath,
			const string &posTagsFilePath);

	static POSTagger& instance();
	static POSTagger& instantiate();
};

