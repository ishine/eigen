#include "Bidirectional.h"
#include "LSTM.h"
#include "Utility.h"

struct BidirectionalLSTM : Bidirectional {	
	BidirectionalLSTM(BinaryReader &dis, merge_mode mode);
};
