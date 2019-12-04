#include "Utility.h"
#include "Embedding.h"

#include "BidirectionalLSTM.h"
#include "Conv1D.h"

#include "CRF.h"

struct NERTagger {
	typedef ::object<NERTagger> object;

	Embedding embedding, repertoire_embedding;

	BidirectionalLSTM lstm;

	Conv1D con1D0, con1D1, con1D2;

	CRF wCRF;

	vector<int>& predict(const String &predict_text,
			vector<int> &repertoire_code);

	vector<vector<vector<double>>>& _predict(const String &predict_text,
			vector<int> &repertoire_code, vector<vector<vector<double>>> &arr);

	NERTagger(BinaryReader &dis);
};
