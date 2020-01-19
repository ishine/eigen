/**
 *
 * Based on the Aho-Corasick white paper, Bell technologies:
 * ftp://163.13.200.222/assistant/bearhero/prog/%A8%E4%A5%A6/ac_bm.pdf
 * 
 * @author Robert Bor
 */
#include "TrieConfig.h"
#include "State.h"
//#include "IntervalTree.h"
#include "Emit.h"
#include "Token.h"

#include <queue>

struct Trie {

	TrieConfig trieConfig;

	object<State> rootState;

	Trie(TrieConfig trieConfig);

	Trie();

	Trie* caseInsensitive();

	Trie* removeOverlaps();

	Trie* onlyWholeWords();

	void addKeyword(String keyword, String value);
	void update(String keyword, String value);
	void remove(String keyword);
	void build(std::map<String, String> &map);

	vector<Token> tokenize(String text);

	Token createFragment(Emit emit, String text, int lastCollectedPosition);
	vector<Emit> parseText(String text);
	void removePartialMatches(String searchText, vector<Emit> collectedEmits);

	static object<State> getState(object<State> currentState,
			char16_t transition);
	void constructFailureStates();

	void updateFailureStates(vector<State::Transition> &queue, String keyword);
	void deleteFailureStates(State *parent, char16_t character, String keyword,
			int numOfDeletion);

	void storeEmits(int position, State *currentState,
			vector<Emit> &collectedEmits);
};
