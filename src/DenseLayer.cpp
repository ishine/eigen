#include "Utility.h"
#include "DenseLayer.h"

Vector &DenseLayer::operator()(const Vector &x, Vector &ret) {
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

DenseLayer::DenseLayer(BinaryReader &dis, bool bias) {
	dis.read(wDense);

	if (bias)
		dis.read(bDense);
}

