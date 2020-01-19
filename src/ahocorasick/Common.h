#include <map>
#include <vector>
using std::vector;
using String = std::u16string;

#include "Trie.h"

namespace ahocorasick {
extern std::map<String, String> dictionaryMap;
extern String text;
extern bool debug;
extern String wordsToBeDeleted;

vector<String> loadDictionary(const string &path =
		"../corpus/ahocorasick/en/dictionary.txt");

vector<String> loadDictionary(const string&, int limit);

String loadText(const string &path = "../corpus/ahocorasick/en/text.txt");

int countAhoCorasickDoubleArrayTrie();

Trie naiveUpdate();

Trie naiveConstruct();

int countNaiveConstruct();

Trie naiveDelete();

void initialize();
}
