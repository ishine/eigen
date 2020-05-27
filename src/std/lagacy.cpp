#include "lagacy.h"
#include <math.h>
const double PI = std::atan(1.0) * 4;

const double sqrt_PI = sqrt(2 / PI);

extern "C" {

dword nCPP = 9;
dword nCPP1 = 8;
dword EFLAGS = 789;

double gelu(double x) {
	double cdf = 0.5 * (1.0 + tanh((sqrt_PI * (x + 0.044715 * pow(x, 3)))));
	return x * cdf;
}

}

