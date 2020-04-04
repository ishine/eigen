#include "CWSTagger.h"

String convertToSegment(const String &predict_text, const VectorI &argmax) {
	String arr;
	String sstr;

	assert((size_t )argmax.size() == predict_text.size());
//	size_t j = 0;
	for (Eigen::Index i = 0; i < argmax.size(); ++i) {
//		if (argmax[i] < 0)
//			continue;

		if (!iswspace(predict_text[i]))
			sstr += predict_text[i];

		if (argmax[i] & 1 && sstr.size()) {
			if (arr.size()) {
				sstr = u" " + sstr;
			}
			arr += sstr;
			sstr.clear();
		}
	}

	if (sstr.size())
		arr += sstr;
	return arr;
}

String CWSTaggerLSTM::predict(const String &predict_text) {
	VectorI seg = string2id(predict_text, this->word2id);
//	cout << "seg = " << seg << endl;
	return convertToSegment(predict_text, this->predict(seg));
}

VectorI& CWSTaggerLSTM::predict(VectorI &predict_text) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
//	cout << "predict_text = " << predict_text.size() << endl;
//	cout << "repertoire_code = " << repertoire_code << endl;

	Matrix lEmbedding;
	embedding(predict_text, lEmbedding);

//	Matrix lRepertoire;
//	repertoire_embedding(repertoire_code, lRepertoire);
//	lEmbedding += lRepertoire;

	Matrix lLSTM;
	lstm(lEmbedding, lLSTM);

	Matrix lCNN;
	lCNN = con1D0(lEmbedding, lCNN);
	lCNN = con1D1(lCNN, lEmbedding);
	lCNN = con1D2(lCNN, lEmbedding);

	Matrix lConcatenate(lLSTM.rows(), lLSTM.cols() + lCNN.cols());
//	lConcatenate.resize();
	lConcatenate << lLSTM, lCNN;

	return wCRF(lConcatenate, predict_text);
}

vector<vector<vector<double>>>& CWSTaggerLSTM::_predict(
		const String &predict_text, vector<vector<vector<double>>> &result) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	Matrix x;
	embedding(string2id(predict_text, this->word2id), x);
	result.push_back(convert2vector(x)); // i = 0

//	Matrix lRepertoire;
//	repertoire_embedding(repertoire_code, lRepertoire);
//	result.push_back(convert2vector(lRepertoire)); // i = 1

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

CWSTaggerLSTM::CWSTaggerLSTM(const string &h5FilePath,
		const string &vocabFilePath) :
		CWSTaggerLSTM((HDF5Reader&) (const HDF5Reader&) HDF5Reader(h5FilePath),
				vocabFilePath) {
}

CWSTaggerLSTM::CWSTaggerLSTM(HDF5Reader &dis, const string &vocabFilePath) :
		embedding(Embedding(dis)),
//		repertoire_embedding(Embedding(dis)),
		con1D0(Conv1D(dis)), con1D1(Conv1D(dis)), lstm(
				BidirectionalLSTM(dis, Bidirectional::sum)), con1D2(
				Conv1D(dis)), wCRF(CRF(dis)) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	Text(vocabFilePath) >> word2id;
}

CWSTaggerLSTM& CWSTaggerLSTM::instance(bool reinitialize) {
	static string modelFile = cnModelsDirectory() + "cws/model-cnn.h5";
	static string vocab = cnModelsDirectory() + "cws/vocab.txt";

	static CWSTaggerLSTM instance(modelFile, vocab);
	if (reinitialize) {
		instance = CWSTaggerLSTM(modelFile, vocab);
	}
	return instance;
}

String CWSTagger::predict(const String &predict_text) {
	VectorI seg = string2id(predict_text, this->word2id);
//	cout << "seg = " << seg << endl;
	return convertToSegment(predict_text, this->predict(seg));
}

VectorI& CWSTagger::predict(VectorI &predict_text) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
//	cout << "predict_text = " << predict_text.size() << endl;
//	cout << "repertoire_code = " << repertoire_code << endl;

	Matrix lEmbedding;
	embedding(predict_text, lEmbedding);

//	Matrix lRepertoire;
//	repertoire_embedding(repertoire_code, lRepertoire);
//	lEmbedding += lRepertoire;

	Matrix lLSTM;
	lstm.call_return_sequences(lEmbedding, lLSTM);

	return wCRF(lLSTM, predict_text);
}

vector<vector<vector<double>>>& CWSTagger::_predict(const String &predict_text,
		vector<vector<vector<double>>> &result) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	Matrix x;
	embedding(string2id(predict_text, this->word2id), x);
	result.push_back(convert2vector(x)); // i = 0

//	Matrix lRepertoire;
//	repertoire_embedding(repertoire_code, lRepertoire);
//	result.push_back(convert2vector(lRepertoire)); // i = 1

	Matrix lLSTM;
	lstm.call_return_sequences(x, lLSTM);
	result.push_back(convert2vector(lLSTM)); // i = 3

	Matrix label;
	wCRF.viterbi_one_hot(lLSTM, label);
	result.push_back(convert2vector(label)); // i = 8

	return result;
}

CWSTagger::CWSTagger(const string &h5FilePath, const string &vocabFilePath) :
		CWSTagger((HDF5Reader&) (const HDF5Reader&) HDF5Reader(h5FilePath),
				vocabFilePath) {
}

CWSTagger::CWSTagger(HDF5Reader &dis, const string &vocabFilePath) :
		embedding(Embedding(dis)),
//		repertoire_embedding(Embedding(dis)),
		lstm(LSTM(dis)), wCRF(CRF(dis)) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	Text(vocabFilePath) >> word2id;
}

CWSTagger& CWSTagger::instance(bool reinitialize) {
	static string modelFile = cnModelsDirectory() + "cws/model.h5";
	static string vocab = cnModelsDirectory() + "cws/vocab.txt";

	static CWSTagger instance(modelFile, vocab);
	if (reinitialize) {
		instance = CWSTagger(modelFile, vocab);
	}
	return instance;
}
