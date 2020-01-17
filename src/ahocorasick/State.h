/**
 * <p>
 * A state has various important tasks it must attend to:
 * </p>
 *
 * <ul>
 * <li>success; when a character points to another state, it must return that
 * state</li>
 * <li>failure; when a character has no matching state, the algorithm must be
 * able to fall back on a state with less depth</li>
 * <li>emits; when this state is passed and keywords have been matched, the
 * matches must be 'emitted' so that they can be used later on.</li>
 * </ul>
 *
 * <p>
 * The root state is special in the sense that it has no failure state; it
 * cannot fail. If it 'fails' it will still parse the next character and start
 * from the root node. This ensures that the algorithm always runs. All other
 * states always have a fail state.
 * </p>
 *
 * @author Robert Bor
 */
#include "../deeplearning/utility.h"
#include <map>
using std::map;
#include <queue>

struct LNodeShadow {
	vector<object<LNodeShadow>> x, y;

	// objects hold a formatted label string and the level,column
	// coordinates for a shadow tree node
	String value; // formatted node value
	int i, j;

	LNodeShadow() {
	}

	LNodeShadow(String value) {
		this->value = value;
	}

	static int max_width(vector<object<LNodeShadow>> &list) {
		int length = 0;
		for (LNodeShadow *x : list) {
			int width = x->max_width();
			if (width > length) {
				length = width;
			}
		}
		return length;
	}

	int max_width() {
		int width = byte_length(value);
		if (x.size()) {
			int width_x = max_width(x);
			if (width_x > width)
				width = width_x;
		}
		if (y.size()) {
			int width_y = max_width(y);
			if (width_y > width)
				width = width_y;
		}
		return width;
	}

	void hierarchize() {
		int column = 0;
		hierarchize(0, column);
	}

	static void hierarchize(vector<object<LNodeShadow>> &list, int level,
			int &column) {
		for (LNodeShadow *x : list) {
			x->hierarchize(level, column);
		}
	}

	// static int size(vector<LNodeShadow> list){
	// int size = 0;
	// for (LNodeShadow x : list){
	// size += x.size();
	// }
	// return size;
	// }
	//
	// int size(){
	// int size = 1;
	// if (x != nullptr)
	// size += size(x);
	// if (y != nullptr)
	// size += size(y);
	// return size;
	// }
	//
	// static int sizeInbeween(vector<LNodeShadow> list){
	// int size = list.size();
	// if (size == 1)
	// return size;
	// int i = 0;
	// size += size(list.get(i).y);
	//
	// for (++i; i < list.size() - 1; ++i){
	// size += list.get(i).size();
	// }
	// size += size(list.get(i).x);
	// return size;
	// }
	//
	void hierarchize(int level, int &column) {
		if (x.size())
			hierarchize(x, level + 1, column);
		// allocate node for left child at next level in tree; attach node
		i = level;
		j = column++; // update column to next cell in the table

		if (y.size())
			hierarchize(y, level + 1, column);
	}

	// the font type should be simsun;
	String toString() {
		return toString(max_width());
	}

	String toString(int max_width) {
		const static auto lineSeparator = u'\n';

		String cout = u"";
		int currLevel = 0;
		int currCol = 0;

		// build the shadow tree
		hierarchize();
		// const int colWidth = Math.max(max_width, max_width()) + 1;
		const int colWidth = max_width;

		// use during the level order scan of the shadow tree
		LNodeShadow *currNode;
		//
		// store siblings of each nodeShadow object in a queue so that they
		// are visited in order at the next level of the tree
		std::queue<LNodeShadow*> q;
		//
		// insert the root in the queue and set current level to 0
		q.push(this);
		//
		// continue the iterative process until the queue
		// is empty
		while (q.size() != 0) {
			// delete front node from queue and make it the
			// current node
			currNode = q.front();
			q.pop();

			if (currNode->i > currLevel) {
				// if level changes, output a newline
				currLevel = currNode->i;
				currCol = 0;
				cout += lineSeparator;
			}

			char16_t ch;
			if (currNode->x.size()) {
//				assert(currNode->x.length > 0);
				for (LNodeShadow *t : currNode->x)
					q.push(t);

				LNodeShadow *head = currNode->x[0];
				// the string is right-aligned / right-justified, that's why
				// there a series of leading ' ';
				int dif = colWidth - byte_length(head->value);// for leading ' 's
				cout += String((head->j - currCol) * colWidth + dif, u' ');
				cout += String((currNode->j - head->j) * colWidth - dif, u'_');

				ch = u'_';
			} else {
				cout += String((currNode->j - currCol) * colWidth, u' ');

				ch = u' ';
			}

			// for leading white spaces;
			cout += String(colWidth - byte_length(currNode->value), ch)
					+ currNode->value;

			currCol = currNode->j;
			if (currNode->y.size()) {
				for (LNodeShadow *t : currNode->y)
					q.push(t);

				LNodeShadow *last = currNode->y[currNode->y.size() - 1];
				cout += String((last->j - currCol) * colWidth, u'_');

				currCol = last->j;
			}

			++currCol;
		}
		cout += lineSeparator;

		return cout;
	}
};

struct State {

	/** effective the size of the keyword */
	const size_t depth;

	/**
	 * referred to in the white paper as the 'goto' structure. From a state it is
	 * possible to go to other states, depending on the character passed.
	 */
	std::map<char16_t, object<State>> success;

	/** if no matching states are found, the failure state will be returned */
	State *failure = nullptr;

	/**
	 * whenever this state is reached, it will emit the matches keywords for future
	 * reference
	 */

	struct Tuple {
		Tuple(int keyword_length, String value) {
			this->char_length = keyword_length;
			this->value = value;
		}

		int char_length;
		String value;
	};

	bool equals(const State &obj) const {
		if (depth != obj.depth)
			return false;

		if (success != obj.success)
			return false;

		if (failure == nullptr) {
			if (obj.failure != nullptr)
				return false;
		} else {
			if (obj.failure == nullptr)
				return false;
			if (failure->depth != obj.failure->depth)
				return false;
		}

		if (emits.size() != obj.emits.size())
			return false;

		return true;
	}

	vector<Tuple> emits;

	LNodeShadow* toShadowTree() {
		LNodeShadow *newNode = new LNodeShadow(u"");
		size_t x_length = success.size() / 2;
		size_t y_length = success.size() - x_length;
		vector<char16_t> list;

		for (auto p = success.begin(); p != success.end(); ++p) {
			list.push_back(p->first);
		}
		// tree node
				if (x_length > 0) {
					newNode->x.resize(x_length);
					for (size_t i = 0; i < x_length; ++i) {
						char16_t word = list[i];
						State *state = success[word];
						LNodeShadow *node = state->toShadowTree();

						newNode->x[i] = node;
						node->value += word;
						if (state->failure != nullptr && state->failure->depth != 0) {
							node->value += ::toString(state->failure->depth);
						}

						switch (state->emits.size()) {
						case 0:
							break;
							case 1:
								node->value += u'+';
								break;
								default:
									node->value += u'*';
									node->value += ::toString(state->emits.size());
									break;
						}
					}
				}
				// allocate node for left child at next level in tree;

		if (y_length > 0) {
			newNode->y.resize(y_length);
			for (size_t i = x_length; i < success.size(); ++i) {
				char16_t word = list[i];
				State *state = success[word];
				LNodeShadow *node = state->toShadowTree();

				newNode->y[i - x_length] = node;
				node->value += word;
				if (state->failure != nullptr && state->failure->depth != 0) {
					node->value += ::toString(state->failure->depth);
				}

				switch (state->emits.size()) {
					case 0:
					break;
					case 1:
					node->value += u'+';
					break;
					default:
					node->value += u'*';
					node->value += ::toString(state->emits.size());
					break;
				}
			}
		}
		return newNode;
	}

		String toString() {
			LNodeShadow *root = this->toShadowTree();
			return root->toString();
		}

		State() :
		State(0) {
		}

		State(int depth) :
		depth(depth) {
		}

		State* nextState(char16_t character) {
			State *nextState = success[character];
			if (nextState == nullptr && depth == 0)
			return this;

			return nextState;
		}

		object<State>& nextStateIgnoreRootState(char16_t character) {
			return success[character];
		}

		State* addState(char16_t character) {
			State *nextState = nextStateIgnoreRootState(character);
			if (nextState == nullptr) {
				nextState = new State(this->depth + 1);
				this->success[character] = nextState;
			}
			return nextState;
		}

		void locate_state(char16_t ch, vector<State*> &list) {
			for (auto p = success.begin(); p != success.end(); ++p) {
				auto &state = p->second;
				if (p->first == ch && state->depth > 1) {
					list.push_back(this);
				}
				state->locate_state(ch, list);
			}
		}

		void locate_state(const String &prefix, const String &keyword,
		vector<State*> &list) {
			for (auto p = success.begin(); p != success.end(); ++p) {
				auto &state = p->second;
				auto ch = p->first;

				String newPrefix = prefix + ch;

				if (newPrefix == keyword && state->depth > keyword.size()) {
					list.push_back(this);
				}

				for (;;) {
					if (keyword.find(newPrefix) == 0 && newPrefix.size() < keyword.size()) {
						state->locate_state(newPrefix, keyword, list);
						break;
					}

					if (newPrefix.empty())
					break;
					newPrefix = newPrefix.substr(1);
				}
			}
		}

		vector<State*> locate_state(char16_t ch) {
			vector<State*> list;
			locate_state(ch, list);
			return list;
		}

		vector<State*> locate_state(String keyword) {
			vector<State*> list;
			this->locate_state(u"", keyword, list);
			return list;
		}

		struct Transition {
			char16_t character;
			State *parent;

			Transition(char16_t character, State *state) {
				this->character = character;
				this->parent = state;
			}

			State* node() {
				return parent->nextStateIgnoreRootState(character);
			}

			void set_failure() {
				if (parent->depth == 0) {
					parent->nextStateIgnoreRootState(character)->failure = parent;
				} else {
					State *targetState = parent->nextStateIgnoreRootState(character);
					State *newFailure = newFailureState(parent, character);
					targetState->failure = newFailure;
					targetState->addEmit(newFailure->emits);
				}
			}
		};

		static void constructFailureStates(vector<State*> list, State *rootState,
		String keyword) {
			char16_t character = keyword[0];
			for (auto parent : list) {
				State *state = parent->success[character];
				state->constructFailureStates(parent, rootState, keyword);
			}
		}

		static void deleteFailureStates(vector<State*> list, String keyword,
		int char_length) {
			char16_t character = keyword[0];
			for (auto parent : list) {
				State *state = parent->success[character];
				state->deleteFailureStates(parent, keyword, char_length);
			}
		}

		static State* newFailureState(State *currentState, char16_t transition) {
			State *state;
			do {
				currentState = currentState->failure;
				state = currentState->nextState(transition);
			}while (state == nullptr);

			return state;
		}

		State* updateState(char16_t character, vector<Transition> &queue) {
			State *nextState = nextStateIgnoreRootState(character);
			if (nextState == nullptr) {
				nextState = new State(depth + 1);
				success[character] = nextState;
				queue.push_back(Transition(character, this));
			}
			return nextState;
		}

		State* updateState(char16_t character) {
			State *nextState = nextStateIgnoreRootState(character);
			if (nextState == nullptr) {
				nextState = new State(depth + 1);
				success[character] = nextState;
			}
			return nextState;
		}

		void addEmit(const Tuple &tuple) {
			this->emits.push_back(tuple);
		}

		void updateEmit(const Tuple &tuple) {
			for (auto &t : emits) {
				if (t.char_length == tuple.char_length) {
					t.value = tuple.value;
					return;
				}
			}

			emits.push_back(tuple);
		}

		void deleteEmit(int char_length) {
			for (auto t = emits.begin(); t != emits.end(); ++t) {
				if (t->char_length == char_length) {
					emits.erase(t);
					break;
				}
			}
		}

		void addEmit(vector<Tuple> &emits) {
			for (const Tuple &emit : emits) {
				addEmit(emit);
			}
		}

		void updateEmit(const vector<Tuple> &emits) {
			for (Tuple emit : emits) {
				updateEmit(emit);
			}
		}

		vector<State*> getStates() {
			vector<State*> list;
			for (auto p = this->success.begin(); p != success.end(); ++p) {
				list.push_back(p->second);
			}
			return list;
		}

		vector<char16_t> getTransitions() {
			vector<char16_t> list;
			for (auto p = this->success.begin(); p != success.end(); ++p) {
				list.push_back(p->first);
			}
			return list;
		}

		bool update_failure(State *parent, char16_t ch) {
			State *newFailureState = State::newFailureState(parent, ch);
			if (failure == newFailureState) {
				return false;
			}

			failure = newFailureState;
			updateEmit(newFailureState->emits);
			return true;
		}

		bool delete_failure(object<State> parent, char16_t ch) {
			object<State> newFailureState = State::newFailureState(parent, ch);
			if (failure == newFailureState) {
				return false;
			}

			failure = newFailureState;
			return true;
		}

		State* update_failure(State *parent, char16_t ch, State *keywordNode) {
			State *newFailureState = State::newFailureState(parent, ch);
			if (failure == newFailureState) {
				if (keywordNode != nullptr)
				updateEmit(keywordNode->emits);
				return failure;
			}

			failure = newFailureState;
			updateEmit(newFailureState->emits);
			return nullptr;
		}

		void constructFailureStates(State *parent, State *rootState, String keyword) {
			char16_t character = keyword[0];
			rootState = rootState->nextStateIgnoreRootState(character);

			keyword = keyword.substr(1);

			bool failure = true;
			if (!update_failure(parent, character)) {
				parent = update_failure(parent, character, rootState);
				failure = false;
			}

// Second, determine the fail state for all depth > 1 state

			if (!keyword.empty()) {
				State *state = success.at(keyword[0]);
				if (state != nullptr) {
					if (failure)
					state->constructFailureStates(this, rootState, keyword);
					else {
						state->constructFailureStates_(this, rootState, keyword);
					}
				}
			}
		}

		void constructFailureStates_(State *parent, State *rootState, String keyword) {
			char16_t character = keyword[0];
			rootState = rootState->nextStateIgnoreRootState(character);

			keyword = keyword.substr(1);

			if (failure->depth <= rootState->depth) {
				failure = rootState;
			}
			updateEmit(rootState->emits);

// Second, determine the fail state for all depth > 1 state

			if (!keyword.empty()) {
				State *state = success.at(keyword[0]);
				if (state != nullptr) {
					state->constructFailureStates_(this, rootState, keyword);
				}
			}
		}

		void deleteFailureStates(State *parent, String keyword, int char_length) {
			char16_t character = keyword[0];

			keyword = keyword.substr(1);

			delete_failure(parent, character);

			if (keyword.empty()) {
				deleteEmit(char_length);
			} else {
				State *state = success.at(keyword[0]);
				if (state != nullptr) {
					state->deleteFailureStates(this, keyword, char_length);
				}
			}
		}
	}
;
