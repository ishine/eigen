using qword = long long int;
extern "C" qword  asm6args(qword rcx, qword rdx, qword r8, qword r9,
		qword fifthArg, qword sixthArg);

extern "C" int gcd_int(int arg1, int arg2, int rcx, int rdx);

qword add_cpp(qword rcx, qword rdx, qword r8, qword r9,
		qword fifthArg, qword sixthArg){
	return rcx + rdx + r8 + r9 + fifthArg + sixthArg;
}

#include <iostream>
using namespace std;
int main(){
	cout << "asm6args(1, 2, 3, 4, 5, 6) = " << asm6args(1, 2, 3, 4, 5, 6) << endl;
	cout << "add_cpp(1, 2, 3, 4, 5, 6) = " << add_cpp(1, 2, 3, 4, 5, 6) << endl;
	cout << "gcd_int(2, 6) = " << gcd_int(0, 0, 2, 6) << endl;
}
