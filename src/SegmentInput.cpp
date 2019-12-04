#include <SegmentInput.h>
#include "lagacy.h"

MatrixI& SegmentInput::operator ()(const MatrixI& inputToken,
		const VectorI &inputMid, MatrixI& inputSegment) {
	int batch_size = inputToken.rows();
	int length = inputToken.cols();
	inputSegment.resize(batch_size, length);

	for (int i = 0; i < batch_size; ++i) {
		int mid = inputMid(i);
		stosd(&inputSegment(i, 0), 0, mid);
		stosd(&inputSegment(i, mid), 1, length - mid);
	}
	return inputSegment;
}
