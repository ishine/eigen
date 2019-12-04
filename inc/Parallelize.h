#pragma once

#include <Utility.h>

struct Parallelize {
	Parallelize(int num_attention_heads);
	int num_attention_heads;
	vector<MatrixI>& operator()(vector<MatrixI> &mask);
};
