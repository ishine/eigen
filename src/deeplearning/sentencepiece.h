#pragma once
#include "../sentencepiece/sentencepiece_processor.h"
/**
 * this header is written to avoid segmentation fault in execution!
 */
sentencepiece::SentencePieceProcessor& en_tokenizer();
