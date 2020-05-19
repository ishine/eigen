#include "keras.h"

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
	label = (*this)(X, label);
	int n = bias.cols();
	Matrix eye = Matrix::Identity(n, n);
	int m = label.size();
	oneHot.resize(m, n);
	for (int i = 0; i < m; ++i) {
		oneHot.row(i) = eye.row(label[i]);
	}
	return oneHot;
}

VectorI& CRF::operator()(const Matrix &X, VectorI &best_paths) const {
	//add a row vector to a matrix
	Matrix x = X * kernel;
	add(x, bias);

	x.row(0) += left_boundary;

	int length = x.rows();
	x.row(length - 1) += right_boundary;

	int i = 0;
	Vector min_energy = x.row(i++);

	vector<vector<int>> argmin_tables(length);

	while (i < length) {
		Matrix energy = G;
		add(energy, min_energy);

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

CRF::CRF(KerasReader &dis) {
	__cout(__PRETTY_FUNCTION__)
	dis >> kernel;
	dis >> G;
	dis >> bias;
	dis >> left_boundary;
	dis >> right_boundary;
}

Conv1D::Conv1D(KerasReader &dis, bool bias) {
	__cout(__PRETTY_FUNCTION__)
	dis >> w;

	if (bias)
		dis >> this->bias;
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

Matrix Conv1D::operator()(const Matrix &x, int s) const {
	Matrix y;
	return (*this)(x, y, s);
}

//	#stride=(1,1)
Matrix& Conv1D::operator()(const Matrix &x, Matrix &y, int s) const {
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

	return activate(y);
}

Conv1DSame::Conv1DSame(KerasReader &dis) {
	__cout(__PRETTY_FUNCTION__)
	dis >> w;

	dis >> this->bias;
}

Matrix Conv1DSame::operator()(const Matrix &x) const {
	Matrix y;
	return (*this)(x, y);
}

//	#stride=(1,1)
Matrix& Conv1DSame::operator()(const Matrix &x, Matrix &y) const {
	int yshape0 = x.rows();
	y = Matrix::Zero(yshape0, x.cols());

	int d0 = (w.size() - 1) / 2;
//#pragma omp parallel for num_threads(cpu_count)
	for (int i = 0; i < yshape0; ++i) {
		int _i = i - d0;
		int di_end = std::min((int) w.size(), (int) x.rows() - _i);
		for (int di = std::max(0, -_i); di < di_end; ++di) {
			y.row(i) += x.row(_i + di) * w[di];
		}

		y.row(i) += bias;
	}

	return activate(y);
}

Vector& DenseLayer::operator()(const Vector &x, Vector &ret) const {
	ret = x * weight + bias;
	return activation(ret);
}

Vector& DenseLayer::operator()(Vector &x) const {

	x *= weight;
	x += bias;
	return activation(x);
}

Matrix& DenseLayer::operator()(const Matrix &x, Matrix &ret) const {
	ret = x * weight;
	add(ret, bias);
	return activation(ret);
}

Matrix& DenseLayer::operator()(Matrix &x) const {
	x *= weight;
	add(x, bias);
	return activation(x);
}

vector<Vector>& DenseLayer::operator()(vector<Vector> &x) const {
	x *= weight;

	x += bias;
	return activation(x);
}

Tensor& DenseLayer::operator()(Tensor &x) const {
	x *= weight;

	x += bias;
	return activation(x);
}

DenseLayer::DenseLayer(KerasReader &dis, Activator activation) :
		activation( { activation }) {
	__cout(__PRETTY_FUNCTION__)
	dis >> weight;

	dis >> bias;
}

DenseLayer::DenseLayer(TorchReader &dis, Activator activation) :
		weight(dis.read_matrix().transpose()), bias(dis.read_vector()),

		activation( { activation }) {
	__cout(__PRETTY_FUNCTION__)
}

Matrix& Embedding::operator()(const VectorI &words,
		Matrix &wordEmbedding) const {
	int length = words.size();

	wordEmbedding.resize(length, wEmbedding.cols());

	for (int j = 0; j < length; ++j) {
		wordEmbedding.row(j) = wEmbedding.row(words[j]);
	}
	return wordEmbedding;
}

Matrix& Embedding::operator()(const vector<int> &words,
		Matrix &wordEmbedding) const {
	int length = words.size();

	wordEmbedding.resize(length, wEmbedding.cols());

	for (int j = 0; j < length; ++j) {
		wordEmbedding.row(j) = wEmbedding.row(words[j]);
	}
	return wordEmbedding;
}

Tensor Embedding::operator()(const vector<VectorI> &words) const {
	Tensor wordEmbedding;
	operator ()(words, wordEmbedding);
	return wordEmbedding;
}

Matrix Embedding::operator()(const VectorI &words) const {
	Matrix wordEmbedding;
	return operator ()(words, wordEmbedding);
}

Matrix Embedding::operator()(const vector<int> &words) const {
	Matrix wordEmbedding;
	return operator ()(words, wordEmbedding);
}

Tensor& Embedding::operator()(const vector<VectorI> &words,
		Tensor &wordEmbedding) const {

	int batch_size = words.size();
	wordEmbedding.resize(batch_size);

	for (int k = 0; k < batch_size; ++k) {
		this->operator ()(words[k], wordEmbedding[k]);
	}
	return wordEmbedding;
}

Matrix& Embedding::operator()(const VectorI &words, Matrix &wordEmbedding,
		Matrix &wEmbedding) const {
	wEmbedding = this->wEmbedding;
	return this->operator ()(words, wordEmbedding);
}

Matrix& Embedding::operator()(VectorI &word, Matrix &wordEmbedding,
		size_t max_length) const {
	word.resize(max_length);
	return (*this)(word, wordEmbedding);

}

void Embedding::initialize(KerasReader &dis) {
	__cout(__PRETTY_FUNCTION__)

	dis >> wEmbedding;
}

void Embedding::initialize(TorchReader &dis) {
	__cout(__PRETTY_FUNCTION__)
	dis >> wEmbedding;
}

Embedding::Embedding(KerasReader &dis) {
	__cout(__PRETTY_FUNCTION__)
	initialize(dis);
}

Embedding::Embedding(TorchReader &dis) {
	__cout(__PRETTY_FUNCTION__)
	initialize(dis);
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

//	this->recurrent_activation = ::hard_sigmoid;
//	this->activation = ::activation;
}

Vector& LSTM::call(const Matrix &x, Vector &h) const {
	Vector c;
	h = c = Vector::Zero(x.cols());

	for (int t = 0; t < x.rows(); ++t) {
		activate(x.row(t), h, c);
	}

	return h;
}

Vector& LSTM::activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h,
		Vector &c) const {
	Vector i = x * Wxi + h * Whi + bi;
	Vector f = x * Wxf + h * Whf + bf;
	Vector _c = x * Wxc + h * Whc + bc;

	_c = recurrent_activation(f).cwiseProduct(c)
			+ recurrent_activation(i).cwiseProduct(activation(_c));

	Vector o = x * Wxo + h * Who + bo;

	c = _c;
	h = recurrent_activation(o).cwiseProduct(activation(_c));
	return h;
}

LSTM::LSTM(KerasReader &dis) {
	__cout(__PRETTY_FUNCTION__)
	Matrix Wx;
	dis >> Wx;
	Wxi.resize(Wx.rows(), Wx.cols() / 4);
	Wxf.resize(Wx.rows(), Wx.cols() / 4);
	Wxc.resize(Wx.rows(), Wx.cols() / 4);
	Wxo.resize(Wx.rows(), Wx.cols() / 4);
	Wx >> Wxi, Wxf, Wxc, Wxo;

	Matrix Wh;
	dis >> Wh;
	Whi.resize(Wh.rows(), Wh.cols() / 4);
	Whf.resize(Wh.rows(), Wh.cols() / 4);
	Whc.resize(Wh.rows(), Wh.cols() / 4);
	Who.resize(Wh.rows(), Wh.cols() / 4);
	Wh >> Whi, Whf, Whc, Who;

	Vector b;
	dis >> b;
	bi.resize(b.size() / 4);
	bf.resize(b.size() / 4);
	bc.resize(b.size() / 4);
	bo.resize(b.size() / 4);
	b >> bi, bf, bc, bo;

//	recurrent_activation = ::hard_sigmoid;
//	activation = ::activation;
}

Matrix& LSTM::call_return_sequences(const Matrix &x, Matrix &arr) const {
	arr.resize(x.rows(), x.cols());
	Vector h, c;
	h = c = Vector::Zero(x.cols());

	for (int t = 0, length = x.rows(); t < length; ++t) {
		arr.row(t) = activate(x.row(t), h, c);
	}

	return arr;
}

Matrix& LSTM::call_return_sequences_reverse(const Matrix &x,
		Matrix &arr) const {
	arr.resize(x.rows(), x.cols());
	Vector h, c;
	h = c = Vector::Zero(x.cols());

	for (int t = x.rows() - 1; t >= 0; --t) {
		arr.row(t) = activate(x.row(t), h, c);
	}

	return arr;
}

Vector& LSTM::call_reverse(const Matrix &x, Vector &h) const {
	Vector c;
	h = c = Vector::Zero(x.cols());

	for (int t = x.rows() - 1; t >= 0; --t) {
		activate(x.row(t), h, c);
	}

	return h;
}

Matrix& Bidirectional::operator ()(const Matrix &x, Matrix &ret) const {
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

Vector& Bidirectional::operator()(const Matrix &x, Vector &ret) const {
	Vector forward;
	this->forward->call(x, forward);

//	printf("forward.shape = (%lld * %lld)\n", forward.rows(), forward.cols());

	Vector backward;
	this->backward->call_reverse(x, backward);

//	printf("backward.shape = (%lld * %lld)\n", backward.rows(), backward.cols());

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

Vector& Bidirectional::operator()(const Matrix &x, Vector &ret,
		vector<vector<double>> &arr) const {
	Vector forward;
//	forward = this->forward->call(x, forward, arr);
	arr.push_back(convert2vector(forward));

	Vector backward;
//	backward = this->backward->call_reverse(x, backward, arr);
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

BidirectionalGRU::BidirectionalGRU(KerasReader &dis, merge_mode mode) {
//enforce the construction order of forward and backward! never to use the member initializer list of the super class!
	this->forward = new GRU(dis);
	this->backward = new GRU(dis);
	this->mode = mode;
}

BidirectionalLSTM::BidirectionalLSTM(KerasReader &dis, merge_mode mode) {
	//enforce the construction order of forward and backward! never to use the member initializer list of the super class!
	__cout(__PRETTY_FUNCTION__)
	this->forward = new LSTM(dis);
	this->backward = new LSTM(dis);
	this->mode = mode;
}

Vector& GRU::call(const Matrix &x, Vector &h) const {
	h = Vector::Zero(x.cols());
	for (int t = 0; t < x.rows(); ++t) {
		h = activate(x.row(t), h);
	}
	return h;
}

Vector& GRU::call_reverse(const Matrix &x, Vector &h) const {
	h = Vector::Zero(x.cols());
	for (int t = x.rows() - 1; t >= 0; --t) {
		h = activate(x.row(t), h);
	}
	return h;
}

Vector& GRU::call(const Matrix &x, Vector &h,
		vector<vector<double>> &arr) const {
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
		vector<vector<double>> &arr) const {
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

Matrix& GRU::call_return_sequences(const Matrix &x, Matrix &arr) const {

	arr.resize(x.rows(), x.cols());
	Vector h;
	h = Vector::Zero(x.cols());

	for (int t = 0, length = x.rows(); t < length; ++t) {
		arr.row(t) = activate(x.row(t), h);
	}

	return arr;
}

Matrix& GRU::call_return_sequences_reverse(const Matrix &x, Matrix &ret) const {
	Vector h = Vector::Zero(x.cols());
	return ret;
}

Vector& GRU::activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h,
		vector<vector<double>> &arr) const {
	Vector tmp = x * Wxr;
	arr.push_back(convert2vector(tmp)); //3

	tmp = h * Whr;
	arr.push_back(convert2vector(tmp)); //4

	arr.push_back(convert2vector(br)); //5

	Vector r = x * Wxr + h * Whr + br;
	arr.push_back(convert2vector(r));
	r = recurrent_activation(r);
//	arr.push_back(convert2vector(r));

	Vector u = x * Wxu + h * Whu + bu;
	u = recurrent_activation(u);
	arr.push_back(convert2vector(u));

	Vector gh = x * Wxh + r.cwiseProduct(h) * Whh + bh;
	gh = activation(gh);
	arr.push_back(convert2vector(gh));

	h = (Vector::Ones(u.cols()) - u).cwiseProduct(gh) + u.cwiseProduct(h);
	return h;
}

Vector& GRU::activate(const Eigen::Block<const Matrix, 1, -1, 1> &x,
		Vector &h) const {
	Vector r = x * Wxr + h * Whr + br;
	r = recurrent_activation(r);

	Vector u = x * Wxu + h * Whu + bu;
	u = recurrent_activation(u);

	Vector gh = x * Wxh + r.cwiseProduct(h) * Whh + bh;
	gh = activation(gh);

	h = (Vector::Ones(u.cols()) - u).cwiseProduct(gh) + u.cwiseProduct(h);
	return h;
}

GRU::GRU(KerasReader &dis) {
	__cout(__PRETTY_FUNCTION__)

	Matrix Wx;
	dis >> Wx;

	Wxu.resize(Wx.rows(), Wx.cols() / 3);
	Wxr.resize(Wx.rows(), Wx.cols() / 3);
	Wxh.resize(Wx.rows(), Wx.cols() / 3);
	Wx >> Wxu, Wxr, Wxh;

	Matrix Wh;
	dis >> Wh;
	Whu.resize(Wh.rows(), Wh.cols() / 3);
	Whr.resize(Wh.rows(), Wh.cols() / 3);
	Whh.resize(Wh.rows(), Wh.cols() / 3);
	Wh >> Whu, Whr, Whh;

	Vector b;
	dis >> b;
	bu.resize(b.size() / 3);
	br.resize(b.size() / 3);
	bh.resize(b.size() / 3);
	b >> bu, br, bh;

//	this->recurrent_activation = ::hard_sigmoid;
//	this->activation = ::activation;
//	this->softmax = ::softmax;

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

Bilinear::Bilinear(TorchReader &dis, Activation activation) :
		weight(dis.read_tensor()),

		bias(dis.read_vector()),

		activation(activation) {
	__cout(__PRETTY_FUNCTION__)
}

Bilinear::Bilinear(KerasReader &dis, Activation activation) :
		weight(dis.read_tensor()),

		bias(dis.read_vector()),

		activation(activation) {
	__cout(__PRETTY_FUNCTION__)
}

Tensor Bilinear::operator ()(const Tensor &x, const Tensor &y) {
	__cout(__PRETTY_FUNCTION__)

	int z_dimension = weight.size();

//	cout << "z_dimension = " << z_dimension << endl;

	print_shape(bias)
//	print_tensor(x);
//	print_tensor(y);
//	print_tensor(weight);

	int n = x.size();
	int m = x[0].rows();

	assert(y.size() == x.size() && x[0].rows() == y[0].rows());

	Tensor z = ndarray(n, m, z_dimension);
	print_tensor(z);
	for (int k = 0; k < z_dimension; ++k) {
		Matrix &weight = this->weight[k];
		double bias = this->bias(k);
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				z[i](j, k) = x[i].row(j) * weight * y[i].row(j).transpose()
						+ bias;
			}
		}
	}

	return activation(z);
}

Matrix Bilinear::operator ()(const Matrix &x, const Matrix &y) {
	__cout(__PRETTY_FUNCTION__)

	int z_dimension = weight.size();

//	cout << "z_dimension = " << z_dimension << endl;

	print_shape(bias)
	print_tensor(weight);

	int n = x.rows();

	assert(y.rows() == x.rows());

	Matrix z = ndarray(n, z_dimension);

	for (int k = 0; k < z_dimension; ++k) {
		Matrix &weight = this->weight[k];
		double bias = this->bias(k);
		for (int i = 0; i < n; ++i) {
			z(i, k) = x.row(i) * weight * y.row(i).transpose() + bias;
		}
	}

	return activation(z);
}

Vector Bilinear::operator ()(const Vector &x, const Vector &y) {
	__cout(__PRETTY_FUNCTION__)

	int z_dimension = weight.size();

//	cout << "z_dimension = " << z_dimension << endl;

	print_shape(bias)
	print_tensor(weight);

	Vector z = ndarray(z_dimension);
	for (int k = 0; k < z_dimension; ++k) {
		Matrix &weight = this->weight[k];
		double bias = this->bias(k);
		z(k) = x * weight * y.transpose() + bias;
	}

	return activation(z);
}
