#pragma once
#include "Utility.h"
#include "matrix.h"

struct CRF {
	Vector bias;
	Matrix G;
	Matrix kernel;
	Vector left_boundary;
	Vector right_boundary;
	VectorActivator activation = nullptr;

	CRF(Matrix kernel, Matrix G, Vector bias, Vector left_boundary,
			Vector right_boundary);

	Matrix& viterbi_one_hot(const Matrix &X, Matrix &oneHot);

	VectorI& call(const Matrix &X, VectorI &best_paths);
	CRF(BinaryReader &dis);
};

struct Conv1D {

	Conv1D(const Tensor &w, const Vector &bias, MatrixActivator activate = relu);

	Tensor w;
	Vector bias;
	MatrixActivator activate;

	Conv1D(BinaryReader &dis, bool bias = true,
			MatrixActivator activate = relu);

	static int initial_offset(int xshape, int yshape, int wshape, int sshape);

//	#stride=(1,1)
	Matrix& conv_same(const Matrix &x, Matrix &y, int s = 1);
};

struct DenseLayer {
	/**
	 *
	 */

	Matrix wDense;
	Vector bDense;

	Vector& operator()(const Vector &x, Vector &ret);
	Vector& operator()(Vector &x);

	Matrix& operator()(Matrix &x, Matrix &wDense);
	Matrix& operator()(Matrix &x);
	Tensor& operator()(Tensor &x);
	vector<Vector>& operator()(vector<Vector> &x);

	DenseLayer(BinaryReader &dis, bool use_bias = true);
	void init(BinaryReader &dis, bool use_bias = true);
};


struct Embedding {
	unordered_map<word, int> char2id;
	Matrix wEmbedding;
	Matrix& call(const String &word, Matrix &wordEmbedding);

	Matrix& operator()(String &word, Matrix &wordEmbedding, size_t max_length);

	Matrix& operator()(const String &word, Matrix &wordEmbedding);

	Matrix& operator()(const String &word, Matrix &wordEmbedding,
			Matrix &wEmbedding);

	Matrix& operator()(const VectorI &word, Matrix &wordEmbedding);

	Matrix& operator()(const VectorI &word, Matrix &wordEmbedding,
			Matrix &wEmbedding);

	Tensor& operator()(const vector<VectorI> &word, Tensor &y);
	Tensor& operator()(const vector<VectorI> &word);

	void initialize(BinaryReader &dis, bool dic);

	Embedding(BinaryReader &dis);
	Embedding(unordered_map<word, int> &char2id, Matrix &wEmbedding);
	Embedding(BinaryReader &dis, bool dic);
};

struct RNN {
	typedef ::object<RNN> object;

	virtual ~RNN() {
	}

	virtual Vector& call(const Matrix &x, Vector &ret) {
		return ret;
	}

	virtual Vector& call(const Matrix &x, Vector &ret,
			vector<vector<double>> &arr) {
		return ret;
	}

	virtual Vector& call_reverse(const Matrix &x, Vector &ret) {
		return ret;
	}

	virtual Vector& call_reverse(const Matrix &x, Vector &ret,
			vector<vector<double>> &arr) {
		return ret;
	}

	virtual Matrix& call_return_sequences(const Matrix &x, Matrix &ret) {
		return ret;
	}

	virtual Matrix& call_return_sequences_reverse(const Matrix &x,
			Matrix &ret) {
		return ret;
	}

	virtual vector<vector<vector<double>>> & weight(
			vector<vector<vector<double>>> &arr) {
		return arr;
	}
};


enum merge_mode {
	sum, mul, ave, concat
};

struct Bidirectional {
	RNN::object forward, backward;

	merge_mode mode;

	Matrix& call_return_sequences(const Matrix &x, Matrix &ret);

	Vector& call(const Matrix &x, Vector &ret);
	Vector& call(const Matrix &x, Vector &ret, vector<vector<double>> &arr);
//private:
//	Bidirectional(RNN *forward, RNN *backward, merge_mode mode);
};

/**
 * implimentation of Gated Recurrent Unit
 */

struct BidirectionalGRU: Bidirectional {

	BidirectionalGRU(BinaryReader &dis, merge_mode mode);
};

struct BidirectionalLSTM : Bidirectional {
	BidirectionalLSTM(BinaryReader &dis, merge_mode mode);
};

/**
 * implimentation of Gated Recurrent Unit
 */

struct GRU: RNN {

	VectorActivator sigmoid;
	VectorActivator tanh;
	VectorActivator softmax;

	Matrix Wxu;
	Matrix Whu;
	Vector bu;

	Matrix Wxr;
	Matrix Whr;
	Vector br;

	Matrix Wxh;
	Matrix Whh;
	Vector bh;

	Vector& call(const Matrix &x, Vector &h);
	Vector& call_reverse(const Matrix &x, Vector &h);
	Vector& call(const Matrix &x, Vector &h, vector<vector<double>> &arr);
	Vector& call_reverse(const Matrix &x, Vector &h,
			vector<vector<double>> &arr);

	Matrix& call_return_sequences(const Matrix &x, Matrix &ret);
	Matrix& call_return_sequences_reverse(const Matrix &x, Matrix &ret);

	Vector& activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h);
	Vector& activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h,
			vector<vector<double>> &arr);

	vector<vector<vector<double>>> &weight(vector<vector<vector<double>>> &arr);

	GRU(BinaryReader &dis);
};

struct LSTM: RNN {

	Matrix Wxi;
	Matrix Whi;
	Matrix Wci;
	Vector bi;

	Matrix Wxf;
	Matrix Whf;
	Matrix Wcf;
	Vector bf;

	Matrix Wxc;
	Matrix Whc;
	Vector bc;

	Matrix Wxo;
	Matrix Who;
	Matrix Wco;
	Vector bo;

	Matrix Why;
	Vector by;

	VectorActivator sigmoid;
	VectorActivator tanh;

	LSTM(Matrix Wxi, Matrix Wxf, Matrix Wxc, Matrix Wxo, Matrix Whi, Matrix Whf,
			Matrix Whc, Matrix Who, Vector bi, Vector bf, Vector bc, Vector bo);
	Vector& call(const Matrix &x, Vector &h);
	Vector& activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h,
			Vector &c);

	LSTM(BinaryReader &dis);
	Matrix& call_return_sequences(const Matrix &x, Matrix &arr);
	Matrix& call_return_sequences_reverse(const Matrix &x, Matrix &arr);
	Vector& call_reverse(const Matrix &x, Vector &h);
};
