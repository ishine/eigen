#include "NERTagger.h"

vector<int>& NERTagger::predict(const String &predict_text,
		vector<int> &repertoire_code) {
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
		vector<int> &repertoire_code, vector<vector<vector<double>>> &result) {
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

