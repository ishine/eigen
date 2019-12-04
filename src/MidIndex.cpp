#include <MidIndex.h>

VectorI& MidIndex::operator ()(const MatrixI& x, VectorI &res) {
	res.resize(x.rows());
	for (int i = 0; i < x.rows(); ++i) {
		for (int j = 0; j < x.cols(); ++j) {
			if (x(i, j) == SEP) {
				res(i) = j + 1;
				break;
			}
		}
	}
	return res;
}


MidIndex::MidIndex(int SEP) {
	this->SEP = SEP;
	cout << "in " << __PRETTY_FUNCTION__ << endl;
}
