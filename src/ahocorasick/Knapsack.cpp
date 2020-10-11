#include "Knapsack.h"

bool Knapsack::equals(const vector<String> &seg_pred,
		const vector<String> &seg_gold) {
	size_t i = 0, j = 0;
	for (; i < seg_pred.size() && j < seg_gold.size();) {
		if (seg_pred[i] == seg_gold[j]) {
			++i;
			++j;
			continue;
		}
		if (startsWith(seg_pred[i], seg_gold[j])) {
			String seg_pred_next = seg_pred[i++];
			do {
				seg_pred_next = seg_pred_next.substr(seg_gold[j++].size());
				if (seg_pred_next.empty())
					break;

				if (!startsWith(seg_pred_next, seg_gold[j]))
					return false;

			} while (true);

		} else if (startsWith(seg_gold[j], seg_pred[i])) {
			String seg_gold_next = seg_gold[j++];
			do {
				seg_gold_next = seg_gold_next.substr(seg_pred[i++].size());
				if (seg_gold_next.empty())
					break;

				if (!startsWith(seg_gold_next, seg_pred[i]))
					return false;

			} while (true);
		} else
			return false;
	}
	return true;
}

void Knapsack::print() {
	for (auto &hit : wordList) {
		String substr = text.substr(hit.begin, hit.end - hit.begin);
		cout << substr << " = " << hit.value << endl;
	}
	cout << endl;
}

void Knapsack::preprocess(vector<Hit> &wordList) {
	for (size_t i = 1; i < wordList.size(); ++i) {

		Hit curr = wordList[i];

		Hit prev = wordList[i - 1];

//			System.out.println("processing prev = " + text.substr(prev.begin, prev.end));
//			System.out.println("processing curr = " + text.substr(curr.begin, curr.end));

		while (prev.end <= curr.end && prev.begin >= curr.begin) {
			wordList.erase(wordList.begin() + --i);
//				print(text, wordList);
			if (i - 1 < 0)
				break;
			prev = wordList[i - 1];
		}
	}
}

Knapsack::Knapsack(const String &text, const vector<Hit> &wordList) :
		text(text), occupied(text.size()), wordList((vector<Hit>&) wordList) {
	spareParts.reserve(2);
}

void Knapsack::add(const Hit &part) {
	partUsed.push_back(part);
//		log.info("adding {}", part.substr(text));
	for (int i = part.begin; i < part.end; ++i) {
		occupied[i] = true;
	}
}

void Knapsack::update(int index, const Hit &part) {
	Hit &prev = partUsed[index];
	//		log.info("adding {}", part.substr(text));
	for (int i = prev.begin; i < prev.end; ++i) {
		occupied[i] = false;
	}

	partUsed[index] = part;

	for (int i = part.begin; i < part.end; ++i) {
		occupied[i] = true;
	}
}

bool Knapsack::checkConformity(const vector<String> &seg_gold) {
	const vector<String> &seg_pred = convert2segmentFromPartUsed();
	for (size_t i = 0; i < seg_pred.size(); ++i) {
		if (seg_pred[i] != seg_gold[i]) {
			cout << "inconsistency detected:" << endl;
			cout << seg_pred << endl;
			cout << seg_gold << endl;
			return false;
		}
	}
	return true;
}

Hit Knapsack::remove(int index) {
	Hit part = partUsed[index];
//		log.info("removing {}", part.substr(text));
	for (int i = part.begin; i < part.end; ++i) {
		occupied[i] = false;
	}
	partUsed.erase(partUsed.begin() + index);
	return part;
}

bool Knapsack::is_occupied(const Hit &currPart) {
	for (int i = currPart.begin; i < currPart.end; ++i) {
		if (occupied[i])
			return true;
	}
	return false;
}

int Knapsack::numOfBlocksCovered(const Hit &part) {
	int sum = 1;
	for (int i = partUsed.size() - 2; i >= 0; --i) {
		if (!partUsed[i].intersects(part))
			break;
		++sum;
	}
	return sum;
}

double Knapsack::scoreOfBlocksCovered(int sum) {
	double score = 0;
	for (int i = partUsed.size() - 1, j = 0; j < sum; --i, ++j) {
		score += partUsed[i].value;
	}
	return score;
}

void Knapsack::update_consecutive(int num, const Hit &part) {
	vector<Hit> list;
	for (int cnt = 0; cnt < num; ++cnt) {
		list.push_back(remove(partUsed.size() - 1));
	}

	add(part);
}

vector<String> Knapsack::cut() {
//	this->print();

	for (auto &currPart : wordList) {
		if (is_occupied(currPart)) {
			spareParts.push_back(&currPart);
			continue;
		}

		if (spareParts.empty()) {
			add(currPart);
			prevPart = &currPart;
			continue;
		}

		sparePart = spareParts.back();

		if (sparePart->end < prevPart->end) {
			spareParts.clear();
			add(currPart);
			prevPart = &currPart;
			continue;
		}

		if (sparePart->intersects(*prevPart)) {
			intersects(currPart);
		} else {
			if (sparePart->value > currPart.value) {
				update(partUsed.size() - 1, *sparePart);
				prevPart = sparePart;
			}
			spareParts.clear();
		}
	}

	if (!spareParts.empty()) {
		sparePart = spareParts.back();

		if (sparePart->end >= prevPart->end
				&& sparePart->intersects(*prevPart)) {
			intersects();
		}

	}

	return convert2segment();
}

void Knapsack::intersects(Hit &currPart) {
	if (sparePart->intersects(currPart)) {
		int numOfBlocksCovered = this->numOfBlocksCovered(*sparePart);
		if (numOfBlocksCovered > 1) {
			double score = scoreOfBlocksCovered(numOfBlocksCovered);
			if (sparePart->value > currPart.value + score) {
				update_consecutive(numOfBlocksCovered, *sparePart);
				prevPart = sparePart;
				spareParts.clear();
			} else {
				add(currPart);
				prevPart = &currPart;
			}
		} else {
			if (sparePart->value > currPart.value + prevPart->value) {
				update(partUsed.size() - 1, *sparePart);
				prevPart = sparePart;
				spareParts.pop_back();
			} else {
				if (spareParts.size() >= 3) {
					spareParts.pop_back();
					sparePart = spareParts.back();
					intersects(currPart);
					return;
				}

				if (spareParts.size() == 2) {
					auto _sparePart = spareParts[0];
					if (!_sparePart->intersects(currPart)) {
						if (_sparePart->value > prevPart->value) {
							spareParts.pop_back();
							sparePart = _sparePart;
							non_intersects(currPart);
							return;
						}
					}
				}

				add(currPart);
				prevPart = &currPart;
			}
		}
	} else {
		non_intersects(currPart);
	}
}

void Knapsack::non_intersects(Hit &currPart) {
	int numOfBlocksCovered = this->numOfBlocksCovered(*sparePart);
	if (numOfBlocksCovered > 1) {
		double score = scoreOfBlocksCovered(numOfBlocksCovered);

		if (sparePart->value > score) {
			update_consecutive(numOfBlocksCovered, *sparePart);
		}
		spareParts.clear();
	} else {
		if (sparePart->value > prevPart->value) {
			update(partUsed.size() - 1, *sparePart);
			prevPart = sparePart;
		}

		spareParts.pop_back();
		if (!spareParts.empty()) {
			auto old_sparePart = sparePart;
			sparePart = spareParts.back();
			if (sparePart->intersects(*old_sparePart)) {
				non_intersects(currPart);
				return;
			}

			if (sparePart->end >= prevPart->end
					&& sparePart->begin != prevPart->begin
					&& !sparePart->intersects(currPart)
					&& sparePart->intersects(*prevPart)
					&& this->numOfBlocksCovered(*sparePart) == 1
					&& sparePart->value > prevPart->value) {
				update(partUsed.size() - 2, *sparePart);
			}

			spareParts.clear();
		}
	}
	add(currPart);
	prevPart = &currPart;
}

void Knapsack::intersects() {
	int numOfBlocksCovered = this->numOfBlocksCovered(*sparePart);

	if (numOfBlocksCovered > 1) {
		double score = scoreOfBlocksCovered(numOfBlocksCovered);

		if (sparePart->value > score) {
			update_consecutive(numOfBlocksCovered, *sparePart);
			prevPart = sparePart;
		}

		spareParts.pop_back();
		if (spareParts.empty())
			return;
		sparePart = spareParts.back();

		if (sparePart->end >= prevPart->end && sparePart->intersects(*prevPart))
			intersects();
		return;

	} else {
		if (sparePart->value > prevPart->value) {
			update(partUsed.size() - 1, *sparePart);
		}

		spareParts.pop_back();
		if (!spareParts.empty()) {
			auto _sparePart = spareParts.back();
			if (!_sparePart->intersects(*sparePart)) {
				sparePart = _sparePart;
				if (sparePart->end >= prevPart->end
						&& sparePart->begin != prevPart->begin
						&& sparePart->intersects(*prevPart)
						&& this->numOfBlocksCovered(*sparePart) == 1
						&& sparePart->value > prevPart->value) {
					update(partUsed.size() - 2, *sparePart);
				}
			}
			spareParts.clear();
		}
	}
//		add(currPart);
//		prevPart = &currPart;
}

String Knapsack::toString() {
	return join(u" ", cut());
}

vector<String> Knapsack::convert2segmentFromPartUsed() {
	vector < String > arr;
	static Hit initial { 0, 0, 0 };
	Hit *prev = &initial;
	for (auto &part : partUsed) {
		if (prev->end != part.begin) {
			for (int i = prev->end; i < part.begin; ++i) {
				arr.push_back(String(1, text[i]));
			}
		}

		arr.push_back(part.substr(text));
		prev = &part;
	}

	return arr;
}

vector<String> Knapsack::convert2segment() {
	vector < String > arr;
	static Hit initial { 0, 0, 0 };
	Hit *prev = &initial;
	for (auto &part : partUsed) {
		if (prev->end != part.begin) {
			for (int i = prev->end; i < part.begin; ++i) {
				auto ch = text[i];
				if (!iswspace(ch))
					arr.push_back(String(1, ch));
			}
		}

		arr.push_back(part.substr(text));
		prev = &part;
	}

	if (prev->end != (int) text.size()) {
		for (size_t i = prev->end; i < text.size(); ++i) {
			auto ch = text[i];
			if (!iswspace(ch))
				arr.push_back(String(1, ch));
		}
	}

	return arr;

}

