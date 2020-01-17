#pragma once

#include "Interval.h"

struct Emit: Interval {
	String value;

	Emit(int start, int end, String value) : Interval(start, end){
		this->value = value;
	}

	String toString() {
		return Interval::toString() + u"=" + this->value;
	}

};
