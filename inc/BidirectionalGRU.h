/**
 * implimentation of Gated Recurrent Unit
 * 
 * @author Cosmos
 *
 */
#include "Bidirectional.h"
#include "GRU.h"
#include "Utility.h"
struct BidirectionalGRU: Bidirectional {

	BidirectionalGRU(BinaryReader &dis, merge_mode mode);
};
