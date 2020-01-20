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

	Trie(const TrieConfig &trieConfig);

	Trie();

	void clear();

	Trie* caseInsensitive();

	Trie* removeOverlaps();

	Trie* onlyWholeWords();

	void addKeyword(const String &keyword, const String &value);
	void update(const String &keyword, const String &value);
	void remove(const String &keyword);
	void build(std::map<String, String> &map);

	vector<Token> tokenize(const String &text);

	Token createFragment(const Emit &emit, const String &text,
			int lastCollectedPosition);

	vector<Emit> parseText(const String &text);

	void removePartialMatches(const String &searchText,
			vector<Emit> &collectedEmits);

	static State* getState(State *currentState, char16_t transition);

	void constructFailureStates();

	void updateFailureStates(vector<State::Transition> &queue, String keyword);

	void deleteFailureStates(State *parent, char16_t character, String keyword,
			int numOfDeletion);

	void storeEmits(int position, State *currentState,
			vector<Emit> &collectedEmits);
};
