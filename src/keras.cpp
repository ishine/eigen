#include <keras.h>

CRF::CRF(Matrix kernel, Matrix G, Vector bias, Vector left_boundary,
		Vector right_boundary) {
	this->bias = bias;

	this->G = G;

	this->kernel = kernel;

	this->left_boundary = left_boundary;

	this->right_boundary = right_boundary;
}

Matrix& CRF::viterbi_one_hot(const Matrix &X, Matrix &oneHot) {
	VectorI label;
	label = call(X, label);
	int n = bias.cols();
	Matrix eye = Matrix::Identity(n, n);
	int m = label.size();
	oneHot.resize(m, n);
	for (int i = 0; i < m; ++i) {
		oneHot.row(i) = eye.row(label[i]);
	}
	return oneHot;
}

VectorI& CRF::call(const Matrix &X, VectorI &best_paths) {
	//add a row vector to a matrix
	Matrix x = X * kernel;
	x.rowwise() += bias;

	x.row(0) += left_boundary;

	int length = x.rows();
	x.row(length - 1) += right_boundary;

	int i = 0;
	Vector min_energy = x.row(i++);

	vector<vector<int>> argmin_tables(length);

	while (i < length) {
		Matrix energy = G;
		energy.rowwise() += min_energy;

		min_energy = min(energy, min_energy, argmin_tables[i - 1]);
		min_energy += x.row(i++);
	}

	int argmin;
	min_energy.minCoeff(&argmin);

	assert(i == length);

	best_paths.resize(length);
	best_paths[--i] = argmin;

	for (--i; i >= 0; --i) {
		argmin = argmin_tables[i][argmin];
		best_paths[i] = argmin;
	}
	return best_paths;
}

CRF::CRF(BinaryReader &dis) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	dis.read(kernel);
	dis.read(G);
	dis.read(bias);
	dis.read(left_boundary);
	dis.read(right_boundary);
}

Conv1D::Conv1D(const Tensor &w, const Vector &bias, MatrixActivator activate) :
		w(w), bias(bias), activate(activate) {
}

Conv1D::Conv1D(BinaryReader &dis, bool bias, MatrixActivator activate) :
		activate(activate) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	dis.read(w);

	if (bias)
		dis.read(this->bias);
}

int Conv1D::initial_offset(int xshape, int yshape, int wshape, int sshape) {
	if (yshape > 1) {
		int l = xshape + (wshape - sshape) * (yshape - 1);
		if (yshape * wshape < l)
			l = yshape * wshape;
		return wshape
				- (2 * wshape + l - (l + wshape - 1) / wshape * wshape + 1) / 2;
	} else
		return -((xshape - wshape) / 2);
}

//	#stride=(1,1)
Matrix& Conv1D::conv_same(const Matrix &x, Matrix &y, int s) {
	int yshape0 = (x.rows() + s - 1) / s;
	y = Matrix::Zero(yshape0, x.cols());

	int d0 = initial_offset(x.rows(), y.rows(), w.size(), s);
	for (int i = 0; i < yshape0; ++i) {
		int _i = s * i - d0;
		int di_end = std::min((int) w.size(), (int) x.rows() - _i);
		for (int di = std::max(0, -_i); di < di_end; ++di) {
			y.row(i) += x.row(_i + di) * w[di];
		}

		if (bias.data())
			y.row(i) += bias;
	}

	if (activate) {
		activate(y);
	}

	return y;
}

Vector& DenseLayer::operator()(const Vector &x, Vector &ret) {
	ret = x * wDense + bDense;
	return ret;
}

Vector& DenseLayer::operator()(Vector &x) {

	x *= wDense;
//		cout << "bDense.data() = " << bDense.data() << endl;
	if (bDense.data())
		x += bDense;
	return x;
}

Matrix& DenseLayer::operator()(Matrix &x, Matrix &wDense) {
	wDense = this->wDense;
	return operator ()(x);
}

Matrix& DenseLayer::operator()(Matrix &x) {
	x *= wDense;
	if (bDense.data())
		x += bDense;
	return x;
}

vector<Vector>& DenseLayer::operator()(vector<Vector> &x) {
	x *= wDense;
	if (bDense.data())
		x += bDense;
	return x;
}

Tensor& DenseLayer::operator()(Tensor &x) {
	x *= wDense;
	if (bDense.data())
		x += bDense;
	return x;
}

DenseLayer::DenseLayer(BinaryReader &dis, bool use_bias) {
	dis.read(wDense);

	if (use_bias)
		dis.read(bDense);
}

Matrix& Embedding::operator()(String &words, Matrix &wordEmbedding,
		size_t max_length) {
	if (max_length && words.length() > max_length) {
		words = words.substr(0, max_length);
	}
	return operator()(words, wordEmbedding);
}

Matrix& Embedding::operator()(const String &words, Matrix &wordEmbedding) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;

	int length = words.size();
//	cout << "length = " << length << endl;

	wordEmbedding.resize(length, wEmbedding.cols());

	for (int j = 0; j < wordEmbedding.rows(); ++j) {
		int index = 1;
		word ch = words[j];
		if (char2id.count(ch))
			index = char2id[ch];

		wordEmbedding.row(j) = wEmbedding.row(index);
	}
	return wordEmbedding;
}

Matrix& Embedding::call(const String &words, Matrix &wordEmbedding) {

	int length = words.size();
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	cout << "length = " << length << endl;

	wordEmbedding.resize(length, wEmbedding.cols());

	for (int j = 0; j < wordEmbedding.rows(); ++j) {
		int index = 1;
		word ch = words[j];

		if (char2id.count(ch))
			index = char2id[ch];

		wordEmbedding.row(j) = this->wEmbedding.row(index);
	}

	return wordEmbedding;
}

Matrix& Embedding::operator()(const VectorI &words, Matrix &wordEmbedding) {

	int length = words.size();
	wordEmbedding.resize(length, wEmbedding.cols());

	for (int j = 0; j < wordEmbedding.rows(); ++j) {
		int index = words[j];
		wordEmbedding.row(j) = wEmbedding.row(index);
	}
	return wordEmbedding;
}

Tensor& Embedding::operator()(const vector<VectorI> &words) {
	static Tensor wordEmbedding;
	operator ()(words, wordEmbedding);
	return wordEmbedding;
}

Tensor& Embedding::operator()(const vector<VectorI> &words, Tensor &wordEmbedding) {

	int batch_size = words.size();
	wordEmbedding.resize(batch_size);

	for (int k = 0; k < batch_size; ++k) {
		this->operator ()(words[k], wordEmbedding[k]);
	}
	return wordEmbedding;
}

Matrix& Embedding::operator()(const VectorI &words, Matrix &wordEmbedding,
		Matrix &wEmbedding) {
	wEmbedding = this->wEmbedding;
	return this->operator ()(words, wordEmbedding);
}

Matrix& Embedding::operator()(const String &words, Matrix &wordEmbedding,
		Matrix &wEmbedding) {
	wEmbedding = this->wEmbedding;
	return this->operator ()(words, wordEmbedding);
}

void Embedding::initialize(BinaryReader &dis, bool dic) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;

	if (dic) {
		dis.read(char2id);
	}

	dis.read(wEmbedding);
}
//
Embedding::Embedding(BinaryReader &dis, bool dic) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	initialize(dis, dic);
}

Embedding::Embedding(unordered_map<word, int> &char2id, Matrix &wEmbedding) :
		char2id(char2id), wEmbedding(wEmbedding) {
}

Embedding::Embedding(BinaryReader &dis) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
//	cout << "in " << __func__ << endl;

	initialize(dis, true);
}

LSTM::LSTM(Matrix Wxi, Matrix Wxf, Matrix Wxc, Matrix Wxo, Matrix Whi,
		Matrix Whf, Matrix Whc, Matrix Who, Vector bi, Vector bf, Vector bc,
		Vector bo) {
	this->Wxi = Wxi;
	this->Wxf = Wxf;
	this->Wxc = Wxc;
	this->Wxo = Wxo;

	this->Whi = Whi;
	this->Whf = Whf;
	this->Whc = Whc;
	this->Who = Who;

	this->bi = bi;
	this->bf = bf;
	this->bc = bc;
	this->bo = bo;

	this->sigmoid = ::hard_sigmoid;
	this->tanh = ::tanh;
}

Vector& LSTM::call(const Matrix &x, Vector &h) {
	Vector c;
	h = c = Vector::Zero(x.cols());

	for (int t = 0; t < x.rows(); ++t) {
		activate(x.row(t), h, c);
	}

	return h;
}

Vector& LSTM::activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h,
		Vector &c) {
	Vector i = x * Wxi + h * Whi + bi;
	Vector f = x * Wxf + h * Whf + bf;
	Vector _c = x * Wxc + h * Whc + bc;

	_c = sigmoid(f).cwiseProduct(c) + sigmoid(i).cwiseProduct(tanh(_c));

	Vector o = x * Wxo + h * Who + bo;

	c = _c;
	h = sigmoid(o).cwiseProduct(tanh(_c));
	return h;
}

LSTM::LSTM(BinaryReader &dis) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	dis.read(Wxi);
	dis.read(Wxf);
	dis.read(Wxc);
	dis.read(Wxo);

	dis.read(Whi);
	dis.read(Whf);
	dis.read(Whc);
	dis.read(Who);

	dis.read(bi);
	dis.read(bf);
	dis.read(bc);
	dis.read(bo);
	sigmoid = ::hard_sigmoid;
	tanh = ::tanh;
}

Matrix& LSTM::call_return_sequences(const Matrix &x, Matrix &arr) {
	arr.resize(x.rows(), x.cols());
	Vector h, c;
	h = c = Vector::Zero(x.cols());

	for (int t = 0, length = x.rows(); t < length; ++t) {
		arr.row(t) = activate(x.row(t), h, c);
	}

	return arr;
}

Matrix& LSTM::call_return_sequences_reverse(const Matrix &x, Matrix &arr) {
	arr.resize(x.rows(), x.cols());
	Vector h, c;
	h = c = Vector::Zero(x.cols());

	for (int t = x.rows() - 1; t >= 0; --t) {
		arr.row(t) = activate(x.row(t), h, c);
	}

	return arr;
}

Vector& LSTM::call_reverse(const Matrix &x, Vector &h) {
	Vector c;
	h = c = Vector::Zero(x.cols());

	for (int t = x.rows() - 1; t >= 0; --t) {
		activate(x.row(t), h, c);
	}

	return h;
}

Matrix& Bidirectional::call_return_sequences(const Matrix &x, Matrix &ret) {
	Matrix forward;
	this->forward->call_return_sequences(x, forward);
	Matrix backward;
	this->backward->call_return_sequences_reverse(x, backward);

	switch (mode) {
	case sum:
		ret = forward + backward;
		break;
	case ave:
		ret = (forward + backward) / 2;
		break;
	case mul:
		ret = forward.array() * backward.array();
		break;
	case concat:
		ret << forward, backward;
		break;
	}
	return ret;
}

Vector& Bidirectional::call(const Matrix &x, Vector &ret) {
	Vector forward;
	this->forward->call(x, forward);
	Vector backward;
	this->backward->call_reverse(x, backward);

	switch (mode) {
	case sum:
		ret = forward + backward;
		break;
	case ave:
		ret = (forward + backward) / 2;
		break;
	case mul:
		ret = forward.cwiseProduct(backward);
		break;
	case concat:
		ret.resize(forward.cols() * 2);
		ret << forward, backward;
		break;
	}
	return ret;
}

Vector& Bidirectional::call(const Matrix &x, Vector &ret,
		vector<vector<double>> &arr) {
	Vector forward;
	forward = this->forward->call(x, forward, arr);
	arr.push_back(convert2vector(forward));

	Vector backward;
	backward = this->backward->call_reverse(x, backward, arr);
	arr.push_back(convert2vector(backward));

	switch (mode) {
	case sum:
		ret = forward + backward;
		break;
	case ave:
		ret = (forward + backward) / 2;
		break;
	case mul:
		ret = forward.cwiseProduct(backward);
		break;
	case concat:
		ret.resize(forward.cols() * 2);
		ret << forward, backward;
		break;
	}
	return ret;
}

BidirectionalGRU::BidirectionalGRU(BinaryReader &dis, merge_mode mode) {
//enforce the construction order of forward and backward! never to use the member initializer list of the super class!
	this->forward = new GRU(dis);
	this->backward = new GRU(dis);
	this->mode = mode;
}

BidirectionalLSTM::BidirectionalLSTM(BinaryReader &dis, merge_mode mode) {
	//enforce the construction order of forward and backward! never to use the member initializer list of the super class!
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	this->forward = new LSTM(dis);
	this->backward = new LSTM(dis);
	this->mode = mode;
}

Vector& GRU::call(const Matrix &x, Vector &h) {
	h = Vector::Zero(x.cols());
	for (int t = 0; t < x.rows(); ++t) {
		h = activate(x.row(t), h);
	}
	return h;
}

Vector& GRU::call_reverse(const Matrix &x, Vector &h) {
	h = Vector::Zero(x.cols());
	for (int t = x.rows() - 1; t >= 0; --t) {
		h = activate(x.row(t), h);
	}
	return h;
}

Vector& GRU::call(const Matrix &x, Vector &h, vector<vector<double>> &arr) {
	h = Vector::Zero(x.cols());
//	arr.push_back(convert2vector(this->Wxu, 0));
//	arr.push_back(convert2vector(this->Wxr, 0));
//	arr.push_back(convert2vector(this->Wxh, 0));
//	arr.push_back(convert2vector(this->Whu, 0));
//	arr.push_back(convert2vector(this->Whr, 0));
//	arr.push_back(convert2vector(this->Whh, 0));
//	arr.push_back(convert2vector(this->bu));
//	arr.push_back(convert2vector(this->br));
//	arr.push_back(convert2vector(this->bh));

	for (int t = 0; t < x.rows(); ++t) {
//		arr.push_back(convert2vector(h));
		h = activate(x.row(t), h);
	}
	return h;
}

Vector& GRU::call_reverse(const Matrix &x, Vector &h,
		vector<vector<double>> &arr) {
	h = Vector::Zero(x.cols());
//	arr.push_back(convert2vector(this->Wxu, 0));
//	arr.push_back(convert2vector(this->Wxr, 0));
//	arr.push_back(convert2vector(this->Wxh, 0));
//	arr.push_back(convert2vector(this->Whu, 0));
//	arr.push_back(convert2vector(this->Whr, 0));
//	arr.push_back(convert2vector(this->Whh, 0));
//
//	arr.push_back(convert2vector(this->bu));
//	arr.push_back(convert2vector(this->br));
//	arr.push_back(convert2vector(this->bh));

	for (int t = x.rows() - 1; t >= 0; --t) {
//		arr.push_back(convert2vector(h));
		h = activate(x.row(t), h);
	}

	return h;
}

Matrix& GRU::call_return_sequences(const Matrix &x, Matrix &ret) {
	Vector h = Vector::Zero(x.cols());
	return ret;
}

Matrix& GRU::call_return_sequences_reverse(const Matrix &x, Matrix &ret) {
	Vector h = Vector::Zero(x.cols());
	return ret;
}

Vector& GRU::activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h,
		vector<vector<double>> &arr) {
	Vector tmp = x * Wxr;
	arr.push_back(convert2vector(tmp)); //3

	tmp = h * Whr;
	arr.push_back(convert2vector(tmp)); //4

	arr.push_back(convert2vector(br)); //5

	Vector r = x * Wxr + h * Whr + br;
	arr.push_back(convert2vector(r));
	r = sigmoid(r);
//	arr.push_back(convert2vector(r));

	Vector u = x * Wxu + h * Whu + bu;
	u = sigmoid(u);
	arr.push_back(convert2vector(u));

	Vector gh = x * Wxh + r.cwiseProduct(h) * Whh + bh;
	gh = tanh(gh);
	arr.push_back(convert2vector(gh));

	h = (Vector::Ones(u.cols()) - u).cwiseProduct(gh) + u.cwiseProduct(h);
	return h;
}

Vector& GRU::activate(const Eigen::Block<const Matrix, 1, -1, 1> &x,
		Vector &h) {
	Vector r = x * Wxr + h * Whr + br;
	r = sigmoid(r);

	Vector u = x * Wxu + h * Whu + bu;
	u = sigmoid(u);

	Vector gh = x * Wxh + r.cwiseProduct(h) * Whh + bh;
	gh = tanh(gh);

	h = (Vector::Ones(u.cols()) - u).cwiseProduct(gh) + u.cwiseProduct(h);
	return h;
}

GRU::GRU(BinaryReader &dis) {
	dis.read(Wxu);
	dis.read(Wxr);
	dis.read(Wxh);

	dis.read(Whu);
	dis.read(Whr);
	dis.read(Whh);

	dis.read(bu);
	dis.read(br);
	dis.read(bh);

	this->sigmoid = ::hard_sigmoid;
	this->tanh = ::tanh;
	this->softmax = ::softmax;

}

vector<vector<vector<double>>>& GRU::weight(
		vector<vector<vector<double>>> &arr) {
	arr.push_back(convert2vector(this->Wxu));
	arr.push_back(convert2vector(this->Wxr));
	arr.push_back(convert2vector(this->Wxh));

	arr.push_back(convert2vector(this->Whu));
	arr.push_back(convert2vector(this->Whr));
	arr.push_back(convert2vector(this->Whh));

	return arr;
}

