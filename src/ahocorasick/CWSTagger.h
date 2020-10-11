#include "../std/utility.h"
#include "AhoCorasickDoubleArrayTrie.h"

struct CWSTagger {

	vector<String> segment(const String &text);

	vector<vector<String>> segment(const vector<String> &text);

	vector<vector<vector<String>>> segment(const vector<vector<String>> &text);

	vector<vector<String>> segment_paralleled(
			const vector<String> &predict_text);

	vector<vector<vector<String>>> segment_paralleled(
			const vector<vector<String>> &predict_text);

	CWSTagger(const string &vocab);

	AhoCorasickDoubleArrayTrie<char16_t, double> dat;

	void weightAdjustment(const std::map<String, double> &map);

	static std::map<String, double> to_map(const string &vocab);

	static CWSTagger& instance();
};
