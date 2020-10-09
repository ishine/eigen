#include "sentencepiece.h"
#include "../deeplearning/utility.h"

sentencepiece::SentencePieceProcessor& en_tokenizer() {
	static sentencepiece::SentencePieceProcessor sp([] {
		return os_access(modelsDirectory() + "en/bert/albert_base/30k-clean.model") ?
		modelsDirectory() + "en/bert/albert_base/30k-clean.model":
		modelsDirectory() + "EN/bert/albert_base/30k-clean.model";
	}());
	return sp;
}

#include "../sentencepiece/sentencepiece.pb.h"

void test_sentencepiece_keras(const std::string &workDirectory) {
	sentencepiece::SentencePieceProcessor &processor = en_tokenizer();

	std::vector<std::string> pieces;
	processor.Encode("This is a test.", &pieces);
	for (const std::string &token : pieces) {
		std::cout << token << std::endl;
	}

	std::vector<int> ids;
	processor.Encode("This is a test.", &ids);
	for (const int id : ids) {
		std::cout << id << std::endl;
	}

	pieces.assign( { "▁This", "▁is", "▁a", "▁", "te", "st", "." });

	// sequence of pieces
	std::string text;
	processor.Decode(pieces, &text);
	std::cout << text << std::endl;

	ids.assign( { 451, 26, 20, 3, 158, 128, 12 });  // sequence of ids
	processor.Decode(ids, &text);
	std::cout << text << std::endl;

//	std::vector<std::string> pieces;
//	processor.SampleEncode("This is a test.", &pieces, -1, 0.2);
//	std::cout << pieces << std::endl;

//	std::vector<int> ids;
//	processor.SampleEncode("This is a test.", &ids, -1, 0.2);
//	std::cout << ids << std::endl;

	sentencepiece::SentencePieceText spt;

	// Encode
	processor.Encode("This is a test.", &spt);

	std::cout << spt.text() << std::endl;   // This is the same as the input.
	for (const auto &piece : spt.pieces()) {
		std::cout << piece.begin() << std::endl;   // beginning of byte offset
		std::cout << piece.end() << std::endl;     // end of byte offset
		std::cout << piece.piece() << std::endl;   // internal representation.
		std::cout << piece.surface() << std::endl; // external representation. spt.text().substr(begin, end - begin) == surface().
		std::cout << piece.id() << std::endl;      // vocab id
	}

	// Decode
	processor.Decode( { 10, 20, 30 }, &spt);
	std::cout << spt.text() << std::endl; // This is the same as the decoded string.
	for (const auto &piece : spt.pieces()) {
		std::cout << piece.begin() << std::endl;   // beginning of byte offset
		std::cout << piece.end() << std::endl;     // end of byte offset
		std::cout << piece.piece() << std::endl;   // internal representation.
		std::cout << piece.surface() << std::endl; // external representation. spt.text().substr(begin, end - begin) == surface().
		std::cout << piece.id() << std::endl;      // vocab id
	}

	std::cout << processor.GetPieceSize();   // returns the size of vocabs.
	std::cout << processor.PieceToId("foo");  // returns the vocab id of "foo"
	std::cout << processor.IdToPiece(10); // returns the string representation of id 10.
	std::cout << processor.IsUnknown(0); // returns true if the given id is an unknown token. e.g., <unk>
	std::cout << processor.IsControl(10); // returns true if the given id is a control token. e.g., <s>, </s>
}

// You can also load a model from std::ifstream.
// std::ifstream in("//path/to/model.model");
// auto status = processor.Load(in);
