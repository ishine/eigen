#include "test.h"

namespace ahocorasick {

vector<String> loadDictionary(const string &path) {
	vector<String> dictionary;
	for (const auto &s : Text(path)) {
		dictionary.push_back(s);
	}

	return dictionary;
}

vector<String> loadDictionary(const string &path, int limit) {
	vector<String> dictionary(limit);
	Text(path) >> dictionary;
//	cout << "dictionary.size() = " << dictionary.size() << endl;
	return dictionary;
}

String loadText(const string &path) {
	return Text(path).toString();
}

//int countAhoCorasickDoubleArrayTrie() {
//
//// Build a AhoCorasickDoubleArrayTrie implemented by hankcs
//	AhoCorasickDoubleArrayTrie<String> ahoCorasickDoubleArrayTrie =
//			new AhoCorasickDoubleArrayTrie<String>();
//
//	ahoCorasickDoubleArrayTrie.build(dictionaryMap);
//
//	vector<String> result;
//
//	ahoCorasickDoubleArrayTrie.parseText(text, new AhoCorasickDoubleArrayTrie.IHit<String>() {
//				void hit(int begin, int end, String value) {
////				System.out.printf("%s = %s\n", text.substring(begin, end), value);
//					result.add(value);
//				}
//			});
//	return result.size();
//}

std::map<String, String> dictionaryMap;
String text;
bool debug = false;
//	String wordsToBeDeleted = "dictatorial";

void initialize() {
	for (String &word : std::sample(
			loadDictionary("../corpus/ahocorasick/en/dictionary.txt"), 10000)) {
		dictionaryMap[word] = word;
	}
//			dictionaryMap.remove(wordsToBeDeleted);

	cout << "dictionary.size() = " << dictionaryMap.size() << endl;
	text = loadText("text.txt");
	if (dictionaryMap.size() <= 10) {
		debug = true;
	}

}

Trie naiveUpdate() {

	Trie ahoCorasickNaive;

	for (auto &p : dictionaryMap) {
		ahoCorasickNaive.update(p.first, p.second);
		if (debug)
			cout << ahoCorasickNaive.rootState;
	}
	printf("construction finished\n");
	return ahoCorasickNaive;
}

Trie naiveConstruct() {
	Trie ahoCorasickNaive;

	ahoCorasickNaive.build(dictionaryMap);

	return ahoCorasickNaive;
}

int countNaiveConstruct() {

	Trie ahoCorasickNaive;

	ahoCorasickNaive.build(dictionaryMap);
	if (debug) {
		printf("building ahocorasic all at once:\n");
		cout << ahoCorasickNaive.rootState;
	}

	vector<String> result;
	for (Emit emit : ahoCorasickNaive.parseText(text)) {
//			int begin = emit.getStart();
//			int end = emit.getEnd();
		String value = emit.value;

//			System.out.printf("%s = %s\n", text.substring(begin, end), value);
		result.push_back(value);
	}
//		System.out.println(ahoCorasickNaive.rootState);

	return result.size();
}

Trie naiveDelete(const String &wordsToBeDeleted) {
	Trie ahoCorasickNaive;

	ahoCorasickNaive.build(dictionaryMap);

	if (debug) {
		printf("before deletion:\n");
		cout << ahoCorasickNaive.rootState;
	}

	ahoCorasickNaive.remove(wordsToBeDeleted);
	if (debug)
		cout << ahoCorasickNaive.rootState;

	printf("construction finished\n");
	return ahoCorasickNaive;
}

void testUpdate() {
	Trie trieConstruction = naiveConstruct();
	Trie trieUpdate = naiveUpdate();

	assert(*trieConstruction.rootState == *trieUpdate.rootState);
	assert(
			trieConstruction.parseText(text).size()
					== trieUpdate.parseText(text).size());
}

void testDelete() {

	vector<String> keywords;
	for (auto &p : dictionaryMap) {
		keywords.push_back(p.first);
	}

	for (String &wordsToBeDeleted : keywords) {
		cout << "testing word: " << wordsToBeDeleted << endl;

		Trie trieDelete = naiveDelete(wordsToBeDeleted);

		dictionaryMap.erase(wordsToBeDeleted);
		Trie trieConstruct = naiveConstruct();
		dictionaryMap[wordsToBeDeleted] = wordsToBeDeleted;

		assert(*trieConstruct.rootState == *trieDelete.rootState);

		assert(
				trieConstruct.parseText(text).size()
						== trieDelete.parseText(text).size());

	}

}

}
