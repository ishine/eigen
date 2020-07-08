#include "AhoCorasickDoubleArrayTrie.h"
#include <random> // std::default_random_engine

void testLoop() {
	const bool debug = true;
	std::map<String, String> dictionaryMap;
	string path = "../jetty/src/test/resources/ahocorasick/dictionary-small.txt";

	vector<String> dictionary;

	Text(path) >> dictionary;

//	dictionary.resize(180);
	for (String &word : dictionary) {
		dictionaryMap[word] = u"[" + word + u"]";
	}

	cout << "dictionary.size() = " << dictionaryMap.size() << endl;
	String text =
			Text("../jetty/src/test/resources/ahocorasick/dictionary-small.txt").toString()
					+ Text(
							"../jetty/src/test/resources/ahocorasick/text-small.txt").toString();

	seed_rand();
	shuffle(dictionary.begin(), dictionary.end(),
			std::default_random_engine(rand()));

	cout << "dictionary.size() = " << dictionary.size() << endl;

//	if (debug) {
//		for (String &word : dictionary) {
//			cout << word << endl;
//		}
//	}

	long start = clock();
	AhoCorasickDoubleArrayTrie<String> dat(dictionaryMap);
	cout << "time cost = " << (clock() - start) / CLOCKS_PER_SEC << endl;
	cout << "space cost = " << dat.node.size() << endl;

	if (debug) {
//		cout << dat << endl;
	}

	dat.checkValidity();
	vector<AhoCorasickDoubleArrayTrie<String>::Hit> arr = dat.parseText(text);
	start = clock();
	AhoCorasickDoubleArrayTrie<String> _dat(dictionaryMap);
	cout << "time cost = " << (clock() - start) / CLOCKS_PER_SEC << endl;
	cout << "space cost = " << _dat.node.size() << endl;

	_dat.checkValidity();
	String debugWord;// = u"水";

	for (String &word : subList(dictionary, 0,
			std::min(100, (int) dictionary.size()))) {
		if (!debugWord.empty() && debugWord != word)
			continue;
//			if (debug)
		cout << "removing word: " << word << endl;

		_dat.remove(word);
		_dat.remove(word);
		if (debug) {
//			cout << _dat << endl;
		}
		_dat.checkValidity();
		_dat.put(word, u"[" + word + u"]");
		_dat.put(word, u"[" + word + u"]");
		if (debug) {
//			cout << _dat << endl;
		}
		_dat.checkValidity();
		assert(dat == _dat);
		vector<AhoCorasickDoubleArrayTrie<String>::Hit> _arr = _dat.parseText(
				text);
		assert(arr == _arr);
	}

	cout << "test successfully!" << endl;
}
