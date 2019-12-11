#include "keras.h"

#include "matrix.h"

struct Service {

	/**
	 * 
	 */

	Embedding embedding;

	BidirectionalGRU gru;
	DenseLayer dense_mean;
	DenseLayer dense_pred;
	int max_length = 30;
	int predict(const word *str);
	int predict(String &predict_text);
//	int predict(const String &predict_text);
	vector<vector<double>>& predict(const word *str,
			vector<vector<double>> &arr);
	vector<vector<double>>& predict(String &predict_text,
			vector<vector<double>> &arr);

	Service(const string &binaryFilePath);
	Service(BinaryReader &dis);
	Service(const BinaryReader &dis);

	vector<vector<vector<double>>>& weight(vector<vector<vector<double>>> &arr);

	static Service& instance();
	static Service& INSTANCE();
};


