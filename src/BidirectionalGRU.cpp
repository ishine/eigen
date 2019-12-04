#include "BidirectionalGRU.h"

BidirectionalGRU::BidirectionalGRU(BinaryReader &dis, merge_mode mode) {
//enforce the construction order of forward and backward! never to use the member initializer list of the super class!
	this->forward = new GRU(dis);
	this->backward = new GRU(dis);
	this->mode = mode;
}

