#include "utility.h"
#include "keras.h"

struct NERTagger {
	typedef ::object<NERTagger> object;

	dict<char16_t, int> word2id;
	Embedding embedding, repertoire_embedding;

	BidirectionalLSTM lstm;

	Conv1D con1D0, con1D1, con1D2;

	CRF wCRF;

	VectorI& predict(const VectorI &predict_text, VectorI &repertoire_code);

	vector<vector<vector<double>>>& _predict(const String &predict_text,
			VectorI&repertoire_code, vector<vector<vector<double>>> &arr);

	NERTagger(KerasReader &dis);
};

struct NERTaggerDict {

	static ::dict<string, NERTagger::object> dict;

	static NERTagger::object& getTagger(const string &service);

	static vector<int> get_repertoire_code(const string &service,
			const String &text);

	static VectorI& predict(const string &service, const String &text,
			VectorI&);
	static vector<vector<vector<double>>>& _predict(const string &service,
			const String &text, VectorI&, vector<vector<vector<double>>>&);
};
