#include "sentencepiece.h"
#include "../deeplearning/utility.h"

sentencepiece::SentencePieceProcessor& en_tokenizer() {
	static sentencepiece::SentencePieceProcessor sp(
			modelsDirectory() + "en/bert/albert_base/30k-clean.model");
	return sp;
}
