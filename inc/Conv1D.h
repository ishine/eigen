#include "Utility.h"
#include "matrix.h"

struct Conv1D {

	Conv1D(const vector<Matrix> &w, const Vector &bias, MatrixActivator activate = relu);

	vector<Matrix> w;
	Vector bias;
	MatrixActivator activate;

	Conv1D(BinaryReader &dis, bool bias = true,
			MatrixActivator activate = relu);

	static int initial_offset(int xshape, int yshape, int wshape, int sshape);

//	#stride=(1,1)
	Matrix& conv_same(const Matrix &x, Matrix &y, int s = 1);
};
