#pragma once

#include <Utility.h>

struct SegmentInput {
	SegmentInput();
	MatrixI& operator()(const MatrixI& token, const VectorI &res, MatrixI& inputSegment);
};
