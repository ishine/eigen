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

class Trie {

	TrieConfig trieConfig;

	object<State> rootState;

	Trie(TrieConfig trieConfig) {
		this->trieConfig = trieConfig;
		this->rootState = new State();
	}

	Trie() :
			Trie(TrieConfig()) {
	}

	Trie* caseInsensitive() {
		this->trieConfig.setCaseInsensitive(true);
		return this;
	}

	Trie* removeOverlaps() {
		this->trieConfig.setAllowOverlaps(false);
		return this;
	}

	Trie* onlyWholeWords() {
		this->trieConfig.setOnlyWholeWords(true);
		return this;
	}

	void addKeyword(String keyword, String value) {
		if (keyword.size() == 0) {
			return;
		}
		State *currentState = this->rootState;
		for (char16_t character : keyword) {
			currentState = currentState->addState(character);
		}

		currentState->addEmit(State::Tuple(keyword.size(), value));
	}

	void update(String keyword, String value) {
		if (keyword.size() == 0) {
			return;
		}

		if (keyword == u"genesis") {
			cout << "update " << keyword << "= " << value << endl;
		}

		vector<State::Transition> start;

		State *currentState = this->rootState;
		for (auto character : keyword) {
			currentState = currentState->updateState(character, start);
		}
		currentState->updateEmit(State::Tuple(keyword.size(), value));

		updateFailureStates(start, keyword);
	}

	void remove(String keyword) {
		if (keyword.size() == 0) {
			return;
		}

//		if (keyword.equals("意思"))
//			System.out.printf("delete %s\n", keyword);

		State *currentState = this->rootState;
		std::stack<State*> parent;
		for (auto character : keyword) {
			parent.push(currentState);
			currentState = currentState->nextStateIgnoreRootState(character);
			if (currentState == nullptr)
				return;
		}

		currentState->deleteEmit(keyword.size());

		char16_t character = 0;
		int numOfDeletion = 0;
		for (int i = keyword.size() - 1; i >= 0; --i) {
			if (!currentState->success.empty())
				break;

			bool tobebroken = false;
			for (auto &tuple : currentState->emits) {
				if (tuple.char_length == i + 1) {
					tobebroken = true;
					break;
				}
			}
			if (tobebroken) {
				break;
			}

			character = keyword[i];
			currentState = parent.top();
			parent.pop();

			currentState->success.erase(character);
			++numOfDeletion;
		}

		deleteFailureStates(currentState, character, keyword, numOfDeletion);
	}

	void build(std::map<String, String> &map) {
		for (auto p = map.begin(); p != map.end(); ++p) {
			this->addKeyword(p->first, p->second);
		}
		this->constructFailureStates();
	}

//	vector<Token> tokenize(String text) {
//
//		vector<Token> tokens;
//
//		vector<Emit> collectedEmits = parseText(text);
//		int lastCollectedPosition = -1;
//		for (Emit emit : collectedEmits) {
//			if (emit.getStart() - lastCollectedPosition > 1) {
//				tokens.push_back(createFragment(emit, text, lastCollectedPosition));
//			}
//			tokens.push_back(createMatch(emit, text));
//			lastCollectedPosition = emit.getEnd();
//		}
//		if (text.size() - lastCollectedPosition > 1) {
//			tokens.push_back(createFragment(nullptr, text, lastCollectedPosition));
//		}
//
//		return tokens;
//	}

//	Token createFragment(Emit emit, String text, int lastCollectedPosition) {
//		return new FragmentToken(
//				text.substr(lastCollectedPosition + 1,
//						emit == nullptr ? text.size() : emit.getStart()));
//	}
//
//	Token createMatch(Emit emit, String text) {
//		return new MatchToken(text.substr(emit.getStart(), emit.getEnd() + 1),
//				emit);
//	}

	vector<Emit> parseText(String text) {

		int position = 0;
		State *currentState = this->rootState;
		vector<Emit> collectedEmits;
		for (char16_t character : text) {
//			if (trieConfig.isCaseInsensitive()) {
//				character = Character.toLowerCase(character);
//			}
			currentState = getState(currentState, character);
			storeEmits(++position, currentState, collectedEmits);
		}

		if (trieConfig.isOnlyWholeWords()) {
//			removePartialMatches(text, collectedEmits);
		}

		if (!trieConfig.isAllowOverlaps()) {
//			IntervalTree intervalTree = IntervalTree(collectedEmits);
//			intervalTree.removeOverlaps(collectedEmits);
		}

		return collectedEmits;
	}

//	void removePartialMatches(String searchText, vector<Emit> collectedEmits) {
//		long size = searchText.size();
//		vector<Emit> removeEmits;
//		for (Emit emit : collectedEmits) {
//			if ((emit.getStart() == 0
//					|| !Character.isAlphabetic(
//							searchText.charAt(emit.getStart() - 1)))
//					&& (emit.getEnd() + 1 == size
//							|| !Character.isAlphabetic(
//									searchText.charAt(emit.getEnd() + 1)))) {
//				continue;
//			}
//			removeEmits.add(emit);
//		}
//
//		for (Emit removeEmit : removeEmits) {
//			collectedEmits.remove(removeEmit);
//		}
//	}

	static object<State> getState(object<State> currentState,
			char16_t transition) {
		for (;;) {
			object<State> state = currentState->nextState(transition);
			if (state != nullptr)
				return state;
			currentState = currentState->failure;
		}
	}

	void constructFailureStates() {
		std::queue<State*> queue;

// First, set the fail state of all depth 1 states to the root state
		for (State *depthOneState : rootState->getStates()) {
			depthOneState->failure = rootState;
			queue.push(depthOneState);
		}

// Second, determine the fail state for all depth > 1 state
		while (!queue.empty()) {
			State *currentState = queue.front();
			queue.pop();

			for (auto p = currentState->success.begin();
					p != currentState->success.end(); ++p) {
				State *targetState = p->second;
				queue.push(targetState);

				State *newFailureState = State::newFailureState(currentState,
						p->first);
				targetState->failure = newFailureState;
				targetState->addEmit(newFailureState->emits);
			}
		}
	}

	void updateFailureStates(vector<State::Transition> &queue, String keyword) {
		for (State::Transition &transit : queue) {
			transit.set_failure();
		}

		State::Transition &keywordHead = queue[0];

		State *rootState = this->rootState;
		vector<State*> list;
		if (keywordHead.parent->depth == 0) {
			list = keywordHead.parent->locate_state(keywordHead.character);

		} else {
			int mid = keyword.size() - (queue.size() - 1);
			String _keyword = keyword.substr(mid - 1);
			keyword = keyword.substr(0, mid);
			list = rootState->locate_state(keyword);

			for (size_t i = 0; i < keyword.size() - 1; ++i) {
				rootState = rootState->success.at(keyword[i]);
			}
			keyword = _keyword;
		}

		State::constructFailureStates(list, rootState, keyword);
	}

	void deleteFailureStates(State *parent, char16_t character, String keyword,
			int numOfDeletion) {
		int char_length = keyword.size();
		State *rootState = this->rootState;
		vector<State*> list;
		if (parent->depth == 0) {
			list = parent->locate_state(character);
		} else {
			size_t mid = keyword.size() - numOfDeletion;
			String _keyword = keyword.substr(mid - 1);
			if (keyword.empty() || keyword.size() < mid)
				return;

			keyword = keyword.substr(0, mid);
			list = rootState->locate_state(keyword);

			keyword = _keyword;
		}

		State::deleteFailureStates(list, keyword, char_length);
	}

	void storeEmits(int position, State *currentState,
			vector<Emit> &collectedEmits) {
		for (State::Tuple &emit : currentState->emits) {
			collectedEmits.push_back(
					Emit(position - emit.char_length, position, emit.value));
		}
	}

};
