#include "../std/utility.h"

#include "AhoCorasickDoubleArrayTrie.h"

using Hit = AhoCorasickDoubleArrayTrie<char16_t, double>::Hit;

struct Knapsack {

	static bool equals(const vector<String> &seg_pred,
			const vector<String> &seg_gold);

	void print();

	void preprocess(vector<Hit> &wordList);

	Knapsack(const String &text, const vector<Hit> &wordList);

	void add(const Hit &part);

	void update(int index, const Hit &part);

	bool checkConformity(const vector<String> &seg_gold);

	Hit remove(int index);

	String text;
	vector<bool> occupied;
	vector<Hit> &wordList;
	vector<Hit> partUsed;
	vector<Hit*> spareParts;
	Hit *prevPart = nullptr;
	Hit *sparePart = nullptr;

	bool is_occupied(const Hit &currPart);

	int numOfBlocksCovered(const Hit &part);

	double scoreOfBlocksCovered(int sum);

	void update_consecutive(int num, const Hit &part);

	vector<String> cut();

	String toString();

	vector<String> convert2segmentFromPartUsed();

	vector<String> convert2segment();

	void intersects(Hit &currPart);
	void non_intersects(Hit &currPart);
	void intersects();
};
