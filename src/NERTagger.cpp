#include "NERTagger.h"

VectorI& NERTagger::predict(const VectorI &predict_text,
		VectorI &repertoire_code) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
//	cout << "predict_text = " << predict_text.size() << endl;
//	cout << "repertoire_code = " << repertoire_code << endl;

	Matrix lEmbedding;
	embedding(predict_text, lEmbedding);

	Matrix lRepertoire;
	repertoire_embedding(repertoire_code, lRepertoire);

	lEmbedding += lRepertoire;

	Matrix lLSTM;
	lstm(lEmbedding, lLSTM);

	Matrix lCNN;
	lCNN = con1D0(lEmbedding, lCNN);
	lCNN = con1D1(lCNN, lEmbedding);
	lCNN = con1D2(lCNN, lEmbedding);

	Matrix lConcatenate(lLSTM.rows(), lLSTM.cols() + lCNN.cols());
//	lConcatenate.resize();
	lConcatenate << lLSTM, lCNN;

	return wCRF.call(lConcatenate, repertoire_code);
}

vector<vector<vector<double>>>& NERTagger::_predict(const String &predict_text,
		VectorI &repertoire_code, vector<vector<vector<double>>> &result) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	Matrix lEmbedding;
	embedding(string2id(predict_text, this->word2id), lEmbedding);
	result.push_back(convert2vector(lEmbedding)); // i = 0

	Matrix lRepertoire;
	repertoire_embedding(repertoire_code, lRepertoire);
	result.push_back(convert2vector(lRepertoire)); // i = 1

	Matrix x = lEmbedding + lRepertoire;
	result.push_back(convert2vector(x)); // i = 2

	Matrix lLSTM;
	lstm(x, lLSTM);
	result.push_back(convert2vector(lLSTM)); // i = 3

	Matrix lCNN;
	lCNN = con1D0(x, lCNN);
	result.push_back(convert2vector(lCNN)); // i = 4

	lCNN = con1D1(lCNN, x);
	result.push_back(convert2vector(lCNN)); // i = 5

	lCNN = con1D2(lCNN, x);
	result.push_back(convert2vector(lCNN)); // i = 6

	Matrix lConcatenate(lLSTM.rows(), lLSTM.cols() + lCNN.cols());
	lConcatenate << lLSTM, lCNN;

	result.push_back(convert2vector(lConcatenate)); // i = 7

	Matrix label;
	wCRF.viterbi_one_hot(lConcatenate, label);
	result.push_back(convert2vector(label)); // i = 8

	return result;
}

NERTagger::NERTagger(HDF5Reader &dis) :
		embedding(Embedding(dis)), repertoire_embedding(Embedding(dis)), lstm(
				BidirectionalLSTM(dis, Bidirectional::sum)), con1D0(Conv1D(dis)), con1D1(
				Conv1D(dis)), con1D2(Conv1D(dis)), wCRF(CRF(dis)) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
//	dis.close();
}

NERTagger::object& NERTaggerDict::getTagger(const string &service) {
	if (!dict.count(service)) {
		cout << "in " << __PRETTY_FUNCTION__ << endl;
		HDF5Reader dis(nerBinary(service));
		dict[service] = new NERTagger(dis);
	}

	return dict[service];

}

vector<int> NERTaggerDict::get_repertoire_code(const string &service,
		const String &text) {
	vector<int> repertoire_code;
	return repertoire_code;
}

VectorI& NERTaggerDict::predict(const string &service, const String &text,
		VectorI &repertoire_code) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
	auto &ptr = getTagger(service);
	return ptr->predict(string2id(text, ptr->word2id), repertoire_code);
}

vector<vector<vector<double>>>& NERTaggerDict::_predict(const string &service,
		const String &text, VectorI &repertoire_code,
		vector<vector<vector<double>>> &arr) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	return getTagger(service)->_predict(text, repertoire_code, arr);
}

extern "C" void cpp_ner_initialize(const char *service) {
	NERTaggerDict::getTagger(service);
}

unordered_map<string, NERTagger::object> NERTaggerDict::dict;
