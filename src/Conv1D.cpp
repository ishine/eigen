#include "Conv1D.h"

Conv1D::Conv1D(const vector<Matrix> &w, const Vector &bias, MatrixActivator activate) :
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
		int di_end = std::min((int)w.size(), (int)x.rows() - _i);
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

