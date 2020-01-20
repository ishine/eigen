#include "State.h"
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

LNodeShadow::LNodeShadow(const String &value) :
		value(value) {
}

int LNodeShadow::max_width(vector<object<LNodeShadow>> &list) {
	int length = 0;
	for (LNodeShadow *x : list) {
		int width = x->max_width();
		if (width > length) {
			length = width;
		}
	}
	return length;
}

int LNodeShadow::max_width() {
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

void LNodeShadow::hierarchize() {
	int column = 0;
	hierarchize(0, column);
}

void LNodeShadow::hierarchize(vector<object<LNodeShadow>> &list, int level,
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
void LNodeShadow::hierarchize(int level, int &column) {
	if (x.size())
		hierarchize(x, level + 1, column);
	// allocate node for left child at next level in tree; attach node
	i = level;
	j = column++; // update column to next cell in the table

	if (y.size())
		hierarchize(y, level + 1, column);
}

// the font type should be simsun;
String LNodeShadow::toString() {
	return toString(max_width());
}

String LNodeShadow::toString(int max_width) {
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
			int dif = colWidth - byte_length(head->value);	// for leading ' 's
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

bool State::operator ==(const State &obj) const {
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

LNodeShadow* State::toShadowTree() {
	LNodeShadow *newNode = new LNodeShadow(u"");
	size_t x_length = success.size() / 2;
	size_t y_length = success.size() - x_length;
	vector<char16_t> list;

	for (auto &p : success) {
		list.push_back(p.first);
	}
	// tree node
	if (x_length > 0) {
		newNode->x.resize(x_length);
		for (size_t i = 0; i < x_length; ++i) {
			auto word = list[i];
			State *state = success.at(word);
			LNodeShadow *node = state->toShadowTree();

			newNode->x[i] = node;
			node->value += word;
			if (state->failure != nullptr && state->failure->depth != 0) {
				node->value += std::toString(state->failure->depth);
			}

			switch (state->emits.size()) {
				case 0:
				break;
				case 1:
				node->value += u'+';
				break;
				default:
				node->value += u'*';
				node->value += std::toString(state->emits.size());
				break;
			}
		}
	}
	// allocate node for left child at next level in tree;

	if (y_length > 0) {
		newNode->y.resize(y_length);
		for (size_t i = x_length; i < success.size(); ++i) {
			auto word = list[i];
			State *state = success.at(word);
			LNodeShadow *node = state->toShadowTree();

			newNode->y[i - x_length] = node;
			node->value += word;
			if (state->failure != nullptr && state->failure->depth != 0) {
				node->value += std::toString(state->failure->depth);
			}

			switch (state->emits.size()) {
				case 0:
				break;
				case 1:
				node->value += u'+';
				break;
				default:
				node->value += u'*';
				node->value += std::toString(state->emits.size());
				break;
			}
		}
	}
	return newNode;
}

String State::toString() {
	LNodeShadow *root = this->toShadowTree();
	return root->toString();
}

State::State(int depth) :
		depth(depth) {
}

State* State::nextState(char16_t character) {
	try {
		return success.at(character);
	} catch (std::out_of_range&) {
		if (depth == 0)
			return this;
		return nullptr;
	}
}

State* State::addState(char16_t character) {
	try {
		return success.at(character);
	} catch (std::out_of_range&) {
		return success[character] = new State(depth + 1);
	}
}

void State::locate_state(char16_t ch, vector<State*> &list) {
	for (auto &p : success) {
		State *state = p.second;
		if (p.first == ch && state->depth > 1) {
			list.push_back(this);
		}
		state->locate_state(ch, list);
	}
}

void State::locate_state(const String &prefix, const String &keyword,
		vector<State*> &list) {
	for (auto &p : success) {
		State *state = p.second;
		auto ch = p.first;

		String newPrefix = prefix + ch;

		if (newPrefix == keyword && state->depth > keyword.size()) {
			list.push_back(this);
		}

		for (;;) {
			if (keyword.find(newPrefix) == 0
					&& newPrefix.size() < keyword.size()) {
				state->locate_state(newPrefix, keyword, list);
				break;
			}

			if (newPrefix.empty())
				break;
			newPrefix = newPrefix.substr(1);
		}
	}
}

vector<State*> State::locate_state(char16_t ch) {
	vector<State*> list;
	list.clear();
	locate_state(ch, list);
	return list;
}

vector<State*> State::locate_state(const String &keyword) {
	vector<State*> list;
	list.clear();
	this->locate_state(u"", keyword, list);
	return list;
}

State::Transition::Transition(char16_t character, State *state) {
	this->character = character;
	this->parent = state;
}

State* State::Transition::node() {
	return parent->success.at(character);
}

void State::Transition::set_failure() {
	if (parent->depth == 0) {
		parent->success.at(character)->failure = parent;
	} else {
		State *targetState = parent->success.at(character);
		State *newFailure = newFailureState(parent, character);
		targetState->failure = newFailure;
		targetState->addEmit(newFailure->emits);
	}
}

void State::constructFailureStates(vector<State*> &list, State *rootState,
		const String &keyword) {
	auto character = keyword[0];
	for (auto parent : list) {
		State *state = parent->success.at(character);
		state->constructFailureStates(parent, rootState, keyword);
	}
}

void State::deleteFailureStates(vector<State*> &list, const String &keyword,
		int char_length) {
	auto character = keyword[0];
	for (auto parent : list) {
		State *state = parent->success.at(character);
		state->deleteFailureStates(parent, keyword, char_length);
	}
}

State* State::newFailureState(State *currentState, char16_t transition) {
	State *state;
	do {
		currentState = currentState->failure;
		state = currentState->nextState(transition);
	} while (state == nullptr);

	return state;
}

State* State::updateState(char16_t character, vector<Transition> &queue) {
	try {
		return success.at(character);
	} catch (std::out_of_range&) {
		queue.push_back(Transition(character, this));
		return success[character] = new State(depth + 1);
	}
}

void State::addEmit(const Tuple &tuple) {
	this->emits.push_back(tuple);
}

void State::updateEmit(const Tuple &tuple) {
	for (auto &t : emits) {
		if (t.char_length == tuple.char_length) {
			t.value = tuple.value;
			return;
		}
	}

	emits.push_back(tuple);
}

void State::deleteEmit(size_t char_length) {
	for (auto t = emits.begin(); t != emits.end(); ++t) {
		if (t->char_length == char_length) {
			emits.erase(t);
			break;
		}
	}
}

void State::addEmit(vector<Tuple> &emits) {
	for (const auto &emit : emits) {
		addEmit(emit);
	}
}

void State::updateEmit(const vector<Tuple> &emits) {
	for (const auto &emit : emits) {
		updateEmit(emit);
	}
}

bool State::update_failure(State *parent, char16_t ch) {
	State *newFailureState = State::newFailureState(parent, ch);
	if (failure == newFailureState) {
		return false;
	}

	failure = newFailureState;
	updateEmit(newFailureState->emits);
	return true;
}

bool State::delete_failure(State *parent, char16_t ch) {
	State *newFailureState = State::newFailureState(parent, ch);
	if (failure == newFailureState) {
		return false;
	}

	failure = newFailureState;
	return true;
}

State* State::update_failure(State *parent, char16_t ch, State *keywordNode) {
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

void State::constructFailureStates(State *parent, State *rootState,
		const String &previous_keyword) {
	auto character = previous_keyword[0];
	try {
		rootState = rootState->success.at(character);
	} catch (std::out_of_range&) {
		rootState = nullptr;
	}

	String keyword = previous_keyword.substr(1);

	bool failure = true;
	if (!update_failure(parent, character)) {
		parent = update_failure(parent, character, rootState);
		failure = false;
	}

// Second, determine the fail state for all depth > 1 state

	if (!keyword.empty()) {
		try {
			State *state = success.at(keyword[0]);
			if (failure)
				state->constructFailureStates(this, rootState, keyword);
			else {
				state->constructFailureStates_(this, rootState, keyword);
			}
		} catch (std::out_of_range&) {
		}
	}
}

void State::constructFailureStates_(State *parent, State *rootState,
		const String &previous_keyword) {
	auto character = previous_keyword[0];
	try {
		rootState = rootState->success.at(character);
	} catch (std::out_of_range&) {
		rootState = nullptr;
	}

	String keyword = previous_keyword.substr(1);

	if (failure->depth <= rootState->depth) {
		failure = rootState;
	}
	updateEmit(rootState->emits);

// Second, determine the fail state for all depth > 1 state

	if (!keyword.empty()) {
		try {
			State *state = success.at(keyword[0]);
			state->constructFailureStates_(this, rootState, keyword);
		} catch (std::out_of_range&) {
		}
	}
}

void State::deleteFailureStates(State *parent, const String &previous_keyword,
		int char_length) {
	auto character = previous_keyword[0];

	String keyword = previous_keyword.substr(1);

	delete_failure(parent, character);

	if (keyword.empty()) {
		deleteEmit(char_length);
	} else {
		try {
			State *state = success.at(keyword[0]);
			state->deleteFailureStates(this, keyword, char_length);
		} catch (std::out_of_range&) {
		}
	}
}
