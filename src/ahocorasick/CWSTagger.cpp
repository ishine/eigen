#include "CWSTagger.h"

std::map<String, double> CWSTagger::to_map(const string &vocab) {
	std::map<String, double> treeMap;

	for (String &tuple : Text(vocab).readlines()) {
		int index = tuple.find_first_of(u'\t');
		String text = tuple.substr(0, index);
		String value = tuple.substr(index + 1);
		assert(text.size() >= 2);

		treeMap[text] = sqrt(
				atoi(Text::unicode2utf(value).data()) * text.size());
	}
	return treeMap;
}

CWSTagger::CWSTagger(const string &vocab) :
		dat(std::map<String, double>()) {
	auto map = to_map(vocab);
	dat = AhoCorasickDoubleArrayTrie<char16_t, double>(map);
	for (auto &entry : map) {
		String text = entry.first;
		vector<AhoCorasickDoubleArrayTrie<char16_t, double>::HitIndexed> array =
				dat.parseTextIndexed(text);
		if (array.size() > 1) {
			AhoCorasickDoubleArrayTrie<char16_t, double>::HitIndexed last = array.back();
			array.pop_back();

			double score = last.value;
			for (auto &hit : array) {
				if (hit.value >= score) {
					score = hit.value + 0.01;
				}
			}
			if (score != last.value) {
				last.setValue(score);
			}
		}
	}
}
