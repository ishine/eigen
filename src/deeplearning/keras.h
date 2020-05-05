#pragma once
#include "matrix.h"
#include<vector>
using std::vector;

struct CRF {
	Vector bias;
	Matrix G;
	Matrix kernel;
	Vector left_boundary;
	Vector right_boundary;
	Activation activation;

	CRF(Matrix kernel, Matrix G, Vector bias, Vector left_boundary,
			Vector right_boundary);

	Matrix& viterbi_one_hot(const Matrix &X, Matrix &oneHot);

	VectorI& operator ()(const Matrix &X, VectorI &best_paths) const;
	CRF(KerasReader &dis);
};

struct Conv1D {
	Tensor w;
	Vector bias;
	Activation activate = { Activator::relu };

	Conv1D(KerasReader &dis, bool bias = true);

	static int initial_offset(int xshape, int yshape, int wshape, int sshape);

//	#stride=(1,1)
	Matrix& operator()(const Matrix &x, Matrix &y, int s = 1) const;
	Matrix operator()(const Matrix &x, int s = 1) const;
};

struct Conv1DSame {
	Tensor w;
	Vector bias;
	Activation activate = { Activator::relu };

	Conv1DSame(KerasReader &dis);

//	#stride=(1,1)
	Matrix& operator()(const Matrix &x, Matrix &y) const;
	Matrix operator()(const Matrix &x) const;
};

struct DenseLayer {
	/**
	 *
	 */

	Matrix weight;
	Vector bias;
	Activation activation = { Activator::tanh };

	Vector& operator()(const Vector &x, Vector &ret) const;
	Vector& operator()(Vector &x) const;

	Matrix& operator()(const Matrix &x, Matrix &wDense) const;
	Matrix& operator()(Matrix &x) const;
	Tensor& operator()(Tensor &x) const;
	vector<Vector>& operator()(vector<Vector> &x) const;

	DenseLayer(KerasReader &dis, Activator activator = Activator::tanh);
	void init(KerasReader &dis);
	DenseLayer(TorchReader &dis, Activator activator = Activator::tanh);

};

struct Embedding {
	Matrix wEmbedding;

	Matrix& operator()(VectorI &word, Matrix &wordEmbedding,
			size_t max_length) const;

	Matrix& operator()(const VectorI &word, Matrix &wordEmbedding) const;

	Matrix& operator()(const VectorI &word, Matrix &wordEmbedding,
			Matrix &wEmbedding) const;

	Tensor& operator()(const vector<VectorI> &word, Tensor &y) const;
	Tensor operator()(const vector<VectorI> &word) const;
	Matrix operator()(const VectorI &word) const;

	void initialize(KerasReader &dis);
	void initialize(TorchReader &dis);

	Embedding(KerasReader &dis);
	Embedding(TorchReader &dis);
};

struct RNN {
	typedef ::object<RNN> object;

//	Activation recurrent_activation = { Activator::hard_sigmoid };
//	since tensorflow 1.15, recurrent_activation=sigmoid
	Activation recurrent_activation = { Activator::sigmoid };
	Activation activation = { Activator::tanh };

	virtual ~RNN() {
	}

	virtual Vector& call(const Matrix &x, Vector &ret) const = 0;

//	Vector& call(const Matrix &x, Vector &ret,
//			vector<vector<double>> &arr) const;
//
	virtual Vector& call_reverse(const Matrix &x, Vector &ret) const = 0;

//	virtual Vector& call_reverse(const Matrix &x, Vector &ret,
//			vector<vector<double>> &arr) const = 0;

	virtual Matrix& call_return_sequences(const Matrix &x,
			Matrix &ret) const = 0;

	virtual Matrix& call_return_sequences_reverse(const Matrix &x,
			Matrix &ret) const = 0;

	virtual vector<vector<vector<double>>>& weight(
			vector<vector<vector<double>>> &arr) {
		return arr;
	}
};

struct Bidirectional {
	RNN::object forward, backward;
	enum merge_mode {
		sum, mul, ave, concat
	};

	merge_mode mode;

	Matrix& operator()(const Matrix &x, Matrix &ret) const;

	Vector& operator()(const Matrix &x, Vector &ret) const;
	Vector& operator()(const Matrix &x, Vector &ret,
			vector<vector<double>> &arr) const;
//private:
//	Bidirectional(RNN *forward, RNN *backward, merge_mode mode);
};

/**
 * implimentation of Gated Recurrent Unit
 */

struct BidirectionalGRU: Bidirectional {

	BidirectionalGRU(KerasReader &dis, merge_mode mode);
};

struct BidirectionalLSTM: Bidirectional {
	BidirectionalLSTM(KerasReader &dis, merge_mode mode);
};

/**
 * implimentation of Gated Recurrent Unit
 */

struct GRU: RNN {
	Matrix Wxu;
	Matrix Whu;
	Vector bu;

	Matrix Wxr;
	Matrix Whr;
	Vector br;

	Matrix Wxh;
	Matrix Whh;
	Vector bh;

	Vector& call(const Matrix &x, Vector &h) const;
	Vector& call_reverse(const Matrix &x, Vector &h) const;
	Vector& call(const Matrix &x, Vector &h, vector<vector<double>> &arr) const;
	Vector& call_reverse(const Matrix &x, Vector &h,
			vector<vector<double>> &arr) const;

	Matrix& call_return_sequences(const Matrix &x, Matrix &ret) const;
	Matrix& call_return_sequences_reverse(const Matrix &x, Matrix &ret) const;

	Vector& activate(const Eigen::Block<const Matrix, 1, -1, 1> &x,
			Vector &h) const;
	Vector& activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h,
			vector<vector<double>> &arr) const;

	vector<vector<vector<double>>>& weight(vector<vector<vector<double>>> &arr);

	GRU(KerasReader &dis);
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

	LSTM(Matrix Wxi, Matrix Wxf, Matrix Wxc, Matrix Wxo, Matrix Whi, Matrix Whf,
			Matrix Whc, Matrix Who, Vector bi, Vector bf, Vector bc, Vector bo);
	Vector& call(const Matrix &x, Vector &h) const;
	Vector& activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h,
			Vector &c) const;

	LSTM(KerasReader &dis);
	Matrix& call_return_sequences(const Matrix &x, Matrix &arr) const;
	Matrix& call_return_sequences_reverse(const Matrix &x, Matrix &arr) const;
	Vector& call_reverse(const Matrix &x, Vector &h) const;
};

struct Bilinear {
	Bilinear(TorchReader &dis, Activation activation = { Activator::linear });
	Bilinear(KerasReader &dis, Activation activation = { Activator::linear });
	Tensor weight;
	Vector bias;
	Activation activation;
	Tensor operator ()(const Tensor &x, const Tensor &y);
	Vector operator ()(const Vector &x, const Vector &y);
	Matrix operator ()(const Matrix &x, const Matrix &y);
};

