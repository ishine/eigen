#pragma once

#include <Utility.h>

struct MidIndex {
	MidIndex(int SEP = 102);
	int SEP;
	VectorI& operator()(const MatrixI& token, VectorI &res);
};
