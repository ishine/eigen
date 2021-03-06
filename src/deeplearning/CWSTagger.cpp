#include "CWSTagger.h"
#include "utility.h"

vector<String> convertToSegment(const String &predict_text,
		const VectorI &argmax) {
	vector<String> arr;
	String sstr;

	assert((size_t ) argmax.size() == predict_text.size());
//	size_t j = 0;
	for (int i = 0, size = argmax.size(); i < size; ++i) {
//		if (argmax[i] < 0)
//			continue;

		if (!iswspace(predict_text[i]))
			sstr += predict_text[i];

		if ((argmax[i] & 1) && sstr.size()) {
//			if (arr.size()) {
//				sstr = u" " + sstr;
//			}
			arr.push_back(sstr);
			sstr.clear();
		}
	}

	if (sstr.size())
		arr.push_back(sstr);
	return arr;
}

vector<String> CWSTaggerLSTM::predict(const String &predict_text) {
	VectorI seg = string2id(predict_text, this->word2id);
//	cout << "seg = " << seg << endl;
	return convertToSegment(predict_text, this->predict(seg));
}

VectorI& CWSTaggerLSTM::predict(VectorI &predict_text) {
//	__debug(__PRETTY_FUNCTION__)
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
	__debug(__PRETTY_FUNCTION__)
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
		CWSTaggerLSTM(
				(KerasReader&) (const KerasReader&) KerasReader(h5FilePath),
				vocabFilePath) {
}

CWSTaggerLSTM::CWSTaggerLSTM(KerasReader &dis, const string &vocabFilePath) :
		embedding(Embedding(dis)),
//		repertoire_embedding(Embedding(dis)),
		con1D0(Conv1D(dis)), con1D1(Conv1D(dis)), lstm(
				BidirectionalLSTM(dis, Bidirectional::sum)), con1D2(
				Conv1D(dis)), wCRF(CRF(dis)) {
	__debug(__PRETTY_FUNCTION__)
	Text(vocabFilePath) >> word2id;
}

CWSTaggerLSTM& CWSTaggerLSTM::instance(bool reinitialize) {
	static string modelFile = weightsDirectory() + "cn/cws/model-cnn.h5";
	static string vocab = weightsDirectory() + "cn/cws/vocab.txt";

	static CWSTaggerLSTM instance(modelFile, vocab);
	if (reinitialize) {
		instance = CWSTaggerLSTM(modelFile, vocab);
	}
	return instance;
}

vector<String> CWSTagger::predict(const String &predict_text) {
	VectorI seg = string2id(predict_text, this->word2id);
	__debug(seg)
	return convertToSegment(predict_text, this->predict(seg));
}

vector<vector<String>> CWSTagger::predict(const vector<String> &predict_text) {
	int length = predict_text.size();
	vector<vector<String>> texts(length);
//#pragma omp parallel for num_threads(cpu_count)
	for (int i = 0; i < length; ++i) {
		if (!predict_text[i]) {
			continue;
		}
		texts[i] = this->predict(predict_text[i]);
	}
	return texts;
}

vector<vector<vector<String>>> CWSTagger::predict(
		const vector<vector<String>> &predict_text) {
	int length = predict_text.size();
	vector<vector<vector<String>>> texts(length);
//#pragma omp parallel for num_threads(cpu_count)
#pragma omp parallel for
	for (int i = 0; i < length; ++i) {
		if (!predict_text[i]) {
			continue;
		}
		texts[i] = this->predict(predict_text[i]);
	}
	return texts;
}

VectorI& CWSTagger::predict(VectorI &predict_text) {
//	__debug(__PRETTY_FUNCTION__)
//	cout << "predict_text = " << predict_text.size() << endl;

	Matrix lEmbedding;
	embedding(predict_text, lEmbedding);
	__debug(lEmbedding)
	return wCRF(con1D(lEmbedding), predict_text);
}

vector<vector<vector<double>>>& CWSTagger::_predict(const String &predict_text,
		vector<vector<vector<double>>> &result) {
	__debug(__PRETTY_FUNCTION__)
	Matrix x;
	embedding(string2id(predict_text, this->word2id), x);
	result.push_back(convert2vector(x)); // i = 0

//	Matrix lRepertoire;
//	repertoire_embedding(repertoire_code, lRepertoire);
//	result.push_back(convert2vector(lRepertoire)); // i = 1

	Matrix lLSTM = con1D(x);
	result.push_back(convert2vector(lLSTM)); // i = 3

	Matrix label;
	wCRF.viterbi_one_hot(lLSTM, label);
	result.push_back(convert2vector(label)); // i = 8

	return result;
}

CWSTagger::CWSTagger(const string &h5FilePath, const string &vocabFilePath) :
		CWSTagger((KerasReader&) (const KerasReader&) KerasReader(h5FilePath),
				vocabFilePath) {
	__log(__PRETTY_FUNCTION__)
}

CWSTagger::CWSTagger(KerasReader &dis, const string &vocabFilePath) :
		word2id(Text(vocabFilePath).read_vocab_char()), embedding(dis), con1D(
				dis), wCRF(dis) {
	__log(__PRETTY_FUNCTION__)
}

CWSTagger& CWSTagger::instance_crf() {
	__debug(__PRETTY_FUNCTION__)
	static CWSTagger inst(weightsDirectory() + "cn/cws/model.h5", weightsDirectory() + "cn/cws/vocab.txt");
	return inst;
}
