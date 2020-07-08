#include "../std/utility.h"
#include "AhoCorasickDoubleArrayTrie.h"

struct CWSTagger {

	vector<String> segment(const String &text);

	vector<vector<String>> segment(const vector<String> &predict_text);

	vector<vector<String>> segment_paralleled(const vector<String> &predict_text);

	vector<vector<vector<String>>> segment_paralleled(const vector<vector<String>> &predict_text);

	CWSTagger(const string &vocab);

	AhoCorasickDoubleArrayTrie<double> dat;

};
