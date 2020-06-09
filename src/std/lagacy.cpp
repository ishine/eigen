#include "lagacy.h"
#include <math.h>
const double PI = std::atan(1.0) * 4;

const double sqrt_PI = sqrt(2 / PI);

//extern "C" {

dword nCPP = 9;
dword nCPP1 = 8;
dword EFLAGS = 789;

double gelu(double x) {
	double cdf = 0.5 * (1.0 + tanh((sqrt_PI * (x + 0.044715 * pow(x, 3)))));
	return x * cdf;
}

double hard_sigmoid(double x) {
	if (x < -2.5)
		return 0;
	if (x > 2.5)
		return 1;
	return 0.2 * x + 0.5;
}

double relu(double x) {
	return x < 0 ? 0 : x;
}

void stosd(void *edi, dword eax, qword rcx) {
	while (rcx) {
		*(int*) edi = eax;
		--rcx;
	}
}

void *movsq(void *_edi, const void *_esi, qword rcx) {
	qword* edi = (qword*) _edi;
	qword* esi = (qword*) _esi;
	while (rcx) {
		*edi = *esi;
		--rcx;
		++edi;
		++esi;
	}
	return edi;
}

//}

