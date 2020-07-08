#pragma once
#include "utility.h"

struct TextTreeNode {
	vector<TextTreeNode*> x, y;

	~TextTreeNode() {
		for (auto p : x) {
			delete p;
		}
		for (auto p : y) {
			delete p;
		}
	}
	// objects hold a formatted label string and the level,column
	// coordinates for a shadow tree node
	String value; // formatted node value
	int i, j;

	TextTreeNode(const String &value = u"");

	static int max_width(vector<TextTreeNode*> &list);
	int max_width();
	void hierarchize();

	static void hierarchize(vector<TextTreeNode*> &list, int level, int &column);

	void hierarchize(int level, int &column);
	// the font type should be simsun;
	String toString();

	String toString(int max_width, bool shrink = false);

	void shrink(TextTreeNode *parent);
	void shrink(vector<TextTreeNode*> &x);
	vector<TextTreeNode*> rightSiblings(TextTreeNode *kinder);

	bool hasLeftKinder(TextTreeNode *kinder);
	TextTreeNode *left_hand();
	TextTreeNode *right_hand();
	TextTreeNode *first_kinder();
	TextTreeNode *last_kinder() ;
	TextTreeNode *left_most();
	static int shift(TextTreeNode *prev, TextTreeNode *curr);
	static void offset(const vector<TextTreeNode*> &x, int dj);
	void offset(int dj) ;
};
