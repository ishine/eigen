#include "NERTagger.h"

VectorI& NERTagger::predict(const String &predict_text,
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
	lstm.call_return_sequences(lEmbedding, lLSTM);

	Matrix lCNN;
	lCNN = con1D0.conv_same(lEmbedding, lCNN);
	lCNN = con1D1.conv_same(lCNN, lEmbedding);
	lCNN = con1D2.conv_same(lCNN, lEmbedding);

	Matrix lConcatenate(lLSTM.rows(), lLSTM.cols() + lCNN.cols());
//	lConcatenate.resize();
	lConcatenate << lLSTM, lCNN;

	return wCRF.call(lConcatenate, repertoire_code);
}

vector<vector<vector<double>>>& NERTagger::_predict(const String &predict_text,
		VectorI &repertoire_code, vector<vector<vector<double>>> &result) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	Matrix lEmbedding;
	embedding.call(predict_text, lEmbedding);
	result.push_back(convert2vector(lEmbedding)); // i = 0

	Matrix lRepertoire;
	repertoire_embedding(repertoire_code, lRepertoire);
	result.push_back(convert2vector(lRepertoire)); // i = 1

	Matrix x = lEmbedding + lRepertoire;
	result.push_back(convert2vector(x)); // i = 2

	Matrix lLSTM;
	lstm.call_return_sequences(x, lLSTM);
	result.push_back(convert2vector(lLSTM)); // i = 3

	Matrix lCNN;
	lCNN = con1D0.conv_same(x, lCNN);
	result.push_back(convert2vector(lCNN)); // i = 4

	lCNN = con1D1.conv_same(lCNN, x);
	result.push_back(convert2vector(lCNN)); // i = 5

	lCNN = con1D2.conv_same(lCNN, x);
	result.push_back(convert2vector(lCNN)); // i = 6

	Matrix lConcatenate(lLSTM.rows(), lLSTM.cols() + lCNN.cols());
	lConcatenate << lLSTM, lCNN;

	result.push_back(convert2vector(lConcatenate)); // i = 7

	Matrix label;
	wCRF.viterbi_one_hot(lConcatenate, label);
	result.push_back(convert2vector(label)); // i = 8

	return result;
}

NERTagger::NERTagger(BinaryReader &dis) :
		embedding(Embedding(dis)), repertoire_embedding(Embedding(dis, false)), lstm(
				BidirectionalLSTM(dis, merge_mode::sum)), con1D0(Conv1D(dis)), con1D1(
				Conv1D(dis)), con1D2(Conv1D(dis)), wCRF(CRF(dis)) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	dis.close();
}

NERTagger::object& NERTaggerDict::getTagger(const string &service) {
	if (!dict.count(service)) {
		cout << "in " << __PRETTY_FUNCTION__ << endl;
		BinaryReader dis(nerBinary(service));
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

	return getTagger(service)->predict(text, repertoire_code);
}

vector<vector<vector<double>>>& NERTaggerDict::_predict(const string &service,
		const String &text, VectorI &repertoire_code,
		vector<vector<vector<double>>> &arr) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	return getTagger(service)->_predict(text, repertoire_code, arr);
}

extern "C" VectorI& cpp_ner(const char *service, const word *text,
		VectorI &repertoire_code) {
	cout << "in " << __FUNCTION__ << endl;
	cout << "service = " << service << endl;
	String words = text;
	cout << "text.size = " << words.size() << endl;
	cout << "repertoire_code.size = " << repertoire_code.size() << endl;
	cout << "repertoire_code = " << repertoire_code << endl;
	assert(words.size() == (size_t )repertoire_code.size());
	auto &ret = NERTaggerDict::predict(service, words, repertoire_code);
	cout << "repertoire_code = " << repertoire_code << endl;
	cout << "ret = " << ret << endl;
	if (&ret == &repertoire_code) {
		cout << "same memory" << endl;
	} else {
		cout << "not same memory" << endl;
	}
	cout << &repertoire_code << endl;
	return ret;
}

extern "C" vector<vector<vector<double>>>& _cpp_ner(const char *service,
		const word *text, VectorI &repertoire_code,
		vector<vector<vector<double>>> &arr) {
//	arr.resize(0);
	cout << "in " << __FUNCTION__ << endl;
	cout << "service = " << service << endl;
	String words = text;
	cout << "text.size = " << words.size() << endl;
	cout << "repertoire_code.size = " << repertoire_code.size() << endl;
	cout << "repertoire_code = " << repertoire_code << endl;
	assert(words.size() == (size_t)repertoire_code.size());

	return NERTaggerDict::_predict(service, words, repertoire_code, arr);
}

extern "C" void cpp_ner_initialize(const char *service) {
	NERTaggerDict::getTagger(service);
}

unordered_map<string, NERTagger::object> NERTaggerDict::dict;
