#include "BidirectionalLSTM.h"

BidirectionalLSTM::BidirectionalLSTM(BinaryReader &dis, merge_mode mode) {
	//enforce the construction order of forward and backward! never to use the member initializer list of the super class!
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	this->forward = new LSTM(dis);
	this->backward = new LSTM(dis);
	this->mode = mode;
}
