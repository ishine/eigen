#pragma once

#include <Utility.h>

struct AttentionMask {
	AttentionMask(bool diagnal_attention = false);
	bool diagnal_attention;
	vector<MatrixI>& operator()(const MatrixI& segment_ids,
			vector<MatrixI>& mask);
};
