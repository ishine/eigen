#include "CWSTagger.h"
#include "../deeplearning/utility.h"
#include "Knapsack.h"

std::map<String, double> CWSTagger::to_map(const string &vocab) {
	std::map<String, double> treeMap;

	for (String &tuple : Text(vocab).readlines()) {
		int index = tuple.find_first_of(u'\t');
		String text = tuple.substr(0, index);
		String value = tuple.substr(index + 1);
		assert(text.size() >= 2);

//		treeMap[text] = sqrt(
//				atoi(Text::unicode2utf(value).data()) * text.size());
		treeMap[text] = atof(Text::unicode2utf(value).data());
	}
	return treeMap;
}

CWSTagger::CWSTagger(const string &vocab) :
		dat(to_map(vocab)) {
}

void CWSTagger::weightAdjustment(const std::map<String, double> &map) {
	for (auto &entry : map) {
		auto &text = entry.first;
		auto array = dat.parseTextIndexed(text);
		if (array.size() > 1) {
			auto last = array.back();
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

	dat.root = nullptr;
}

CWSTagger& CWSTagger::instance() {
	__cout(__PRETTY_FUNCTION__)
	static CWSTagger inst(modelsDirectory() + "cn/cws/vocab.csv");
	return inst;
}

vector<String> CWSTagger::segment(const String &text) {
	return Knapsack(text, dat.parseText(text)).cut();
}

vector<vector<String>> CWSTagger::segment(const vector<String> &text) {
	int size = text.size();
	vector<vector<String>> arr(size);
	for (int i = 0; i < size; ++i) {
		arr[i] = segment(text[i]);
	}

	return arr;
}

vector<vector<vector<String>>> CWSTagger::segment(
		const vector<vector<String>> &text) {
	int size = text.size();
	vector<vector<vector<String>> > arr(size);
	for (int i = 0; i < size; ++i) {
		arr[i] = segment(text[i]);
	}

	return arr;
}
