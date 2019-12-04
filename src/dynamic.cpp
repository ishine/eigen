#include <iostream>
using std::cout;
using std::endl;

void print() {
	static int status = 0;
	cout << "this is a program! id = " << status++ << endl;
}
