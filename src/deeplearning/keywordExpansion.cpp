#include "keywordExpansion.h"
#include "../json/json.h"
#include "matrix.h"

Json::Value readFromStream(const string &json_file);

Weight& Weight::operator +=(const MapGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	for (auto &tuple : rhs.map) {
		auto &key = tuple.first;
		auto gradient = tuple.second;

#if USING_P_GIVEN_X
		if (key == "p_given_x")
			p_given_x += *gradient;
#else
		if (key == "x_given_p")
			x_given_p += *gradient;
#endif

		else if (key == "x_probability")
			x_probability += *gradient;
		else if (key == "epsilon")
			oov_probability += *gradient;
		else {
			__log(__PRETTY_FUNCTION__);
			throw;
		}
	}
	return *this;
}

Weight& Weight::operator -=(const MapGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	for (auto &tuple : rhs.map) {
		auto &key = tuple.first;
		auto gradient = tuple.second;
#if USING_P_GIVEN_X
		if (key == "p_given_x")
			p_given_x -= *gradient;
#else
		if (key == "x_given_p")
			x_given_p -= *gradient;
#endif

		else if (key == "x_probability")
			x_probability -= *gradient;
		else if (key == "epsilon")
			oov_probability -= *gradient;
		else {
			__log(__PRETTY_FUNCTION__);
			throw;
		}
	}
	return *this;
}

std::map<string, Weight>& operator +=(std::map<string, Weight> &lhs,
		const MapGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	for (auto &tuple : rhs.map) {
		lhs[tuple.first] += *tuple.second;
	}

	return lhs;
}

std::map<string, Weight>& operator -=(std::map<string, Weight> &lhs,
		const MapGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	for (auto &tuple : rhs.map) {
		lhs[tuple.first] -= *tuple.second;
	}

	return lhs;
}

std::map<string, Weight>& operator +=(std::map<string, Weight> &lhs,
		const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const MapGradient*>(&rhs);
	if (gradient)
		return lhs += *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return lhs;
}

std::map<string, Weight>& operator -=(std::map<string, Weight> &lhs,
		const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const MapGradient*>(&rhs);
	if (gradient)
		return lhs -= *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return lhs;
}

Weight& Weight::operator +=(const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const MapGradient*>(&rhs);
	if (gradient)
		return *this += *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return *this;
}

Weight& Weight::operator -=(const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const MapGradient*>(&rhs);
	if (gradient)
		return *this -= *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return *this;
}

vector<HyperParameter>& operator +=(vector<HyperParameter> &lhs,
		const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const VectorGradient*>(&rhs);
	if (gradient)
		return lhs += *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return lhs;
}

vector<HyperParameter>& operator -=(vector<HyperParameter> &lhs,
		const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const VectorGradient*>(&rhs);
	if (gradient)
		return lhs -= *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return lhs;
}

vector<HyperParameter>& operator +=(vector<HyperParameter> &lhs,
		const VectorGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	for (size_t key = 0; key < rhs.vec.size(); ++key) {
		lhs[key] += *rhs.vec[key];
	}
	return lhs;
}

vector<HyperParameter>& operator -=(vector<HyperParameter> &lhs,
		const VectorGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	for (size_t key = 0; key < rhs.vec.size(); ++key) {
		lhs[key] -= *rhs.vec[key];
	}
	return lhs;
}

HyperParameter& HyperParameter::operator +=(const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const MapGradient*>(&rhs);
	if (gradient)
		return *this += *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return *this;
}

HyperParameter& HyperParameter::operator -=(const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const MapGradient*>(&rhs);
	if (gradient)
		return *this -= *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return *this;
}

HyperParameter& HyperParameter::operator +=(const MapGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	for (auto &tuple : rhs.map) {
		(*this)[tuple.first] += *tuple.second;
	}
	return *this;
}

HyperParameter& HyperParameter::operator -=(const MapGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	for (auto &tuple : rhs.map) {
		(*this)[tuple.first] -= *tuple.second;
	}
	return *this;
}

const string KeywordExpansionManager::fields[] = { "abst", "title" };

bool KeywordExpansionManager::does_weight_exists() {
	for (auto &field : fields) {
		if (os_access(
				"../pytext/data/" + section + "/" + string(field)
						+ "/x_probability.bin"))
			continue;
		return false;
	}
	return true;
}

bool KeywordExpansionManager::does_vocab_count_exists() {
	for (auto &field : fields) {
		if (os_access(
				"../pytext/data/" + section + "/" + string(field)
						+ "/vocab.txt"))
			continue;
		return false;
	}
	return true;
}

KeywordExpansionManager::KeywordExpansionManager(const string &section,
		double oov_probability) :
		section(section) {
	__log(__PRETTY_FUNCTION__);

	if (this->does_weight_exists()) {
		this->load_weights();
	} else {
		this->init_weights();
	}
	for (auto &field : this->fields) {
		this->weight[field].oov_probability = oov_probability;
	}
}

void KeywordExpansionManager::load_weights() {
	__log(__PRETTY_FUNCTION__);
//	__log(getcwd());
	__timer_begin();
	for (auto &field : fields) {
		std::ifstream vocabFile(
				"../pytext/data/" + section + "/" + string(field)
						+ "/vocab.txt");

#if USING_P_GIVEN_X
		BinaryFile p_given_x_file(
				"../pytext/data/" + section + "/" + string(field)
						+ "/p_given_x.bin");
#else
		BinaryFile x_given_p_file(
				"../pytext/data/" + section + "/"+ string(field) + "/x_given_p.bin");
#endif

		string line;

		std::set < string > x_words;

		string prev_p_word;
		while (std::getline(vocabFile, line)) {
			auto array = split(line);
			auto &p_word = array[0];

			ensure_true(prev_p_word < p_word);
			prev_p_word = p_word;

			int length = array.size() - 1;

			vector<double> floats(length);
#if USING_P_GIVEN_X
			p_given_x_file >> floats;
#else
			x_given_p_file >> floats;
#endif

			string prev_x_word;
			for (int i = 0; i < length; ++i) {
				auto &x_word = array[i + 1];

				ensure_true(prev_x_word < x_word);
				prev_x_word = x_word;

#if USING_P_GIVEN_X
				this->weight[field].p_given_x[p_word][x_word] = floats[i];
#else
				this->weight[field].x_given_p[p_word][x_word] = floats[i];
#endif

				x_words.insert(x_word);
			}
		}

		ensure_true(p_given_x_file.getsize() == 0);

		{
			BinaryFile x_probability_file(
					"../pytext/data/" + section + "/" + string(field)
							+ "/x_probability.bin");

			vector<double> floats(x_words.size());
			x_probability_file >> floats;
			int i = 0;
			for (auto &x_word : x_words) {
				this->weight[field].x_probability[x_word] = floats[i++];
			}

			ensure_true(x_probability_file.getsize() == 0);
		}
	}

	__timer_end();
}

void KeywordExpansionManager::build_vocab_count() {
	__log(__PRETTY_FUNCTION__);
	std::map<string, std::map<string, std::map<string, int>>> count;
	for (auto &inst : load_data()) {
		inst.accumulate(count);
	}

	for (auto &tuple : count) {
		auto &field = tuple.first;
		auto &count_field = tuple.second;

		std::ofstream vocab_file(
				"../pytext/data/" + section + "/" + string(field)
						+ "/vocab.txt");

		BinaryFile count_file(
				"../pytext/data/" + section + "/" + string(field)
						+ "/count.bin");

		for (auto &pair : count_field) {
			auto &p_word = pair.first;
			auto &count_field_p_word = pair.second;

			vocab_file << p_word;
			for (auto &pair : count_field_p_word) {
				vocab_file << ' ' << pair.first;
			}
			vocab_file << endl;
			count_file << values(count_field_p_word);
		}
	}
}

void KeywordExpansionManager::init_weights() {
	__log(__PRETTY_FUNCTION__);
	__timer_begin();

	if (!does_vocab_count_exists()) {
		build_vocab_count();
	}

#if USING_P_GIVEN_X
	std::map<string, std::map<string, std::map<string, double> > > p_given_x;
#endif

	for (auto &field : fields) {
//		__print(field);
		std::ifstream vocabFile(
				"../pytext/data/" + section + "/" + string(field)
						+ "/vocab.txt");
		BinaryFile countFile(
				"../pytext/data/" + section + "/" + string(field)
						+ "/count.bin");

		int numOfIntegers = countFile.getsize() / 4;
		int numOfIntegersCnt = 0;
		string line;

		string prev_p_word;
		while (std::getline(vocabFile, line)) {
			auto array = split(line);
			auto &p_word = array[0];

			ensure_true(prev_p_word < p_word);
			prev_p_word = p_word;

			int length = array.size() - 1;

			vector<int> counts(length);
			countFile >> counts;
			numOfIntegersCnt += length;

			string prev_x_word;
			for (int i = 0; i < length; ++i) {
				auto &x_word = array[i + 1];

				ensure_true(prev_x_word < x_word);
				prev_x_word = x_word;

#if USING_P_GIVEN_X
				p_given_x[field][x_word][p_word] = counts[i];
#else
				weight[field].x_given_p[p_word][x_word] = counts[i];
#endif

//				cout << "p_probability[" << field << "][" << x_word << "]["
//						<< p_word << "] = " << counts[i] << endl;
			}
		}

		ensure_true(countFile.getsize() == 0);
		ensure_true(numOfIntegersCnt == numOfIntegers);
	}

#if USING_P_GIVEN_X

	for (auto &field : fields) {
//		__print(field);

		for (auto &tuple : p_given_x[field]) {
			auto &x_word = tuple.first;
			auto x_sum = sum(values(tuple.second));
//			__print(x_sum);

			this->weight[field].x_probability[x_word] = x_sum;
			for (auto &pair : tuple.second) {
				auto &p_word = pair.first;
//#determine P(p_word | x_word)
//#determine logP(p_word | x_word)
				pair.second = std::log(pair.second / x_sum);

				ensure_false(std::isnan(pair.second));
				ensure_false(std::isinf(pair.second));

//#convert to weight: self.weight[field][p_word][x_word] = self.p_probability[field][x_word][p_word]
				this->weight[field].p_given_x[p_word][x_word] = pair.second;

//				cout << "weight[" << field << "].p_probability[" << p_word
//						<< "][" << x_word << "] = " << pair.second << endl;

			}
		}

		auto x_sum = sum(values(this->weight[field].x_probability));
//		__print(x_sum);
		for (auto &pair : this->weight[field].x_probability) {
			//#determine P(x_word)
			//#determine logP(x_word)
			pair.second = std::log(pair.second / x_sum);

			ensure_false(std::isnan(pair.second));
			ensure_false(std::isinf(pair.second));

//			cout << "weight[" << field << "].x_probability[" << pair.first
//					<< "] = " << pair.second << endl;
		}
	}
#else


	for (auto &field : fields) {
//		__print(field);

		for (auto &tuple : weight[field].x_given_p) {
//			auto &p_word = tuple.first;
			auto p_sum = sum(values(tuple.second));
//			__print(p_sum);

//			this->weight[field].p_probability[p_word] = p_sum;
			for (auto &pair : tuple.second) {
//#determine P(x_word | p_word)
//#determine logP(x_word | p_word)
				pair.second = std::log(pair.second / p_sum);

				ensure_false(std::isnan(pair.second));
				ensure_false(std::isinf(pair.second));
//				cout << "weight[" << field << "].p_probability[" << p_word
//						<< "][" << x_word << "] = " << pair.second << endl;

			}
		}

//		auto p_sum = sum(values(this->weight[field].p_probability));
//		__print(p_sum);
//		for (auto &pair : this->weight[field].p_probability) {
//			//#determine P(x_word)
//			pair.second /= p_sum;
//			//#determine logP(x_word)
//			pair.second = std::log(pair.second);
//
//			ensure_false(std::isnan(pair.second));
//			ensure_false(std::isinf(pair.second));
//
////			cout << "weight[" << field << "].x_probability[" << pair.first
////					<< "] = " << pair.second << endl;
//		}

	}
#endif

	this->save_weights();
	__timer_end();
}

void KeywordExpansionManager::save_weights() const {
	__log(__PRETTY_FUNCTION__);
	__timer_begin();

	for (auto &tuple : this->weight) {
		auto &field = tuple.first;
		auto &weight = tuple.second;

#if USING_P_GIVEN_X
		BinaryFile p_given_x_file(
				"../pytext/data/" + section + "/" + string(field)
						+ "/p_given_x.bin");
		for (auto &pair : weight.p_given_x) {
			p_given_x_file << values(pair.second);
		}

#else
		BinaryFile x_given_p_file(
				"../pytext/data/" + section + "/"+ string(field) + "/x_given_p.bin");
		for (auto &pair : weight.x_given_p) {
			x_given_p_file << values(pair.second);
		}

#endif
		BinaryFile x_probability_file(
				"../pytext/data/" + section + "/" + string(field)
						+ "/x_probability.bin");
		x_probability_file << values(weight.x_probability);
	}
	__timer_end();
}

double Weight::sigma(const std::vector<string> &p_words, const string &x_word,
		int n) const {
#ifdef USING_P_GIVEN_X
//	argmax[i]P(x[i] | p) = argmax[i](logP(x[i]) + ∑[j:n]logP(p[j] | x[i]))
	vector<double> p_given_x;
	p_given_x.reserve(p_words.size());
	for (auto &p_word : p_words) {
		p_given_x.push_back(this->p_given_x.at(p_word).at(x_word));
	}
	return sum(p_given_x) + this->x_probability.at(x_word)
			+ (n - p_given_x.size()) * this->oov_probability;
#else
	//	argmax[i]P(x[i] | p) = argmax[i](-(n-1)logP(x[i]) + ∑[j:n]logP(x[i] | p[j]))
	vector<double> x_given_p;
	x_given_p.reserve(p_words.size());
	for (auto &p_word : p_words) {
		x_given_p.push_back(this->x_given_p.at(p_word).at(x_word));
	}

	return sum(x_given_p) - (n - 1) * this->x_probability.at(x_word) + (n - x_given_p.size()) * this->oov_probability;
#endif
}

void KeywordExpansionManager::sigma_gradient(Gradient &gradient,
		const string &field, const std::vector<string> &p_words,
		const string &x_word, double delta) const {
//	__log(__PRETTY_FUNCTION__);
//		__log(gradient);
//      delta = y_true - (sum(weight[field].p_probability[p_word][x_word] for p_word in p_words) + weight[field].x_probability[x_word])
//      Loss = delta ** 2 / 2

	vector<double> v;
	v.reserve(p_words.size());
	for (auto &p_word : p_words) {
		gradient[field]["p_given_x"][p_word][x_word] -= delta;
	}

	gradient[field]["x_probability"][x_word] -= delta;
}

double KeywordExpansionManager::accuracy_per_instance(
		const ExpansionInstance &inst, Gradient *gradient) const {

//	__log(__PRETTY_FUNCTION__);

	auto &field = inst.field;
	auto &inputSet = inst.inputSet;
	auto &goldSet = inst.goldSet;
	if (inputSet.empty() or goldSet.empty() or not this->weight.count(field))
		return 1;

	int max_extend_length = std::ceil(inputSet.size() * 1.5);

	int goldSet_length = goldSet.size();

//	__log(field);
//	__log(len(inputSet));
//	__log(inputSet);
//	__log(goldSet);

	std::map<string, std::vector<string>> x2p;

	for (auto &p_word : inputSet) {
		ensure_true(this->weight.at(field).p_given_x.count(p_word));

		for (auto &tuple : this->weight.at(field).p_given_x.at(p_word)) {
			auto &x_word = tuple.first;
//			if (inputSet.count(x_word))
//				continue;

			x2p[x_word].push_back(p_word);
		}
	}

	std::map<string, double> mappingDict;
	for (auto &tuple : x2p) {
		auto &x_word = tuple.first;
		mappingDict[x_word] = this->weight.at(field).sigma(tuple.second, x_word,
				inputSet.size());
	}

	auto sortedItems = items(mappingDict);

	sort(sortedItems,
			[](const std::pair<string, double> &lhs,
					const std::pair<string, double> &rhs) {
				return lhs.second > rhs.second;
			});

	std::set < string > predSet;
	for (auto &e : subList(sortedItems, 0, max_extend_length)) {
		predSet.insert(e.first);
	}

	if (gradient) {
		auto unwantedSet = predSet - goldSet;
		auto missingSet = goldSet - predSet;

		int numOfMoves = std::min(unwantedSet.size(), missingSet.size());
		if (numOfMoves) {
			vector<std::pair<string, int>> missingItems;
			for (int index = max_extend_length; index < sortedItems.size();
					++index) {

				auto &x_word = sortedItems[index].first;
				if (missingSet.count(x_word)) {
					missingItems.push_back( { x_word, index });
					if (missingItems.size() >= numOfMoves)
						break;
				}
			}

			for (auto &tuple : missingItems) {
				auto &x_word = tuple.first;
				int index = tuple.second;

				ensure_true(mappingDict.count(x_word));

				ensure_ge(index, max_extend_length);
//			__log(this->weight[field].hyper);
//			cout << "mappingDict[" << x_word << "] = " << mappingDict[x_word]
//					<< endl;
//			cout << "index of " << x_word
//					<< " is lagging behind, needs adjustment!" << endl;

//			double delta = std::max(this->epsilon,
//					sortedItems[index - 1].second - mappingDict[x_word]);
				double delta = this->epsilon;
				delta /= goldSet_length;
				ensure_gt(delta, 0);
				sigma_gradient(*gradient, field, x2p[x_word], x_word, delta);
			}

			vector<std::pair<string, int>> unwantedItems;
			for (int index = std::min(max_extend_length,
					(int) sortedItems.size()) - 1; index >= 0; --index) {
				auto &x_word = sortedItems[index].first;

				if (not (unwantedSet.count(x_word)))
					continue;

				unwantedItems.push_back( { x_word, index });
				if (unwantedItems.size() >= numOfMoves)
					break;
			}

			for (auto &tuple : unwantedItems) {
				auto &x_word = tuple.first;
				int index = tuple.second;

				assert_lt(index, max_extend_length);
//			ensure_eq(sortedItems[index].second.second, mappingDict[x_word]);

				ensure_not(goldSet.count(x_word));

//			cout << "x_word " << x_word
//					<< " is not in goldSet, needs adjustment!" << endl;

//			double delta = std::min(-this->epsilon,
//					sortedItems[index + 1].second.second - mappingDict[x_word]);
				double delta = -this->epsilon;
				delta /= goldSet_length;
				ensure_lt(delta, 0);
				sigma_gradient(*gradient, field, x2p[x_word], x_word, delta);
			}
		}
	}

	return double((predSet & goldSet).size()) / goldSet_length;
}

std::set<string> KeywordExpansionManager::predict(const string &field,
		std::set<string> &inputSet) {
	std::set < string > predSet;
	return predSet;
}

void KeywordExpansionManager::training(int epoch, int batch_size,
		double learning_rate, double epsilon, int num_threads, bool shuffling) {
//	__log(__PRETTY_FUNCTION__);
	this->epsilon = epsilon;
	this->batch_size = batch_size;
	auto training_data = load_data();

	History bestHistory;
	__print(training_data.size());

	AdamOptimizer optimizer(
			(training_data.size() + this->batch_size - 1) / this->batch_size,
			learning_rate, 0);

	for (int i = 0; i < epoch; ++i) {
		if (shuffling)
			shuffle(training_data);
		auto newHistory = this->total_accuracy(training_data, &optimizer);
		cout << "total accuracy = " << newHistory;
		if (bestHistory.empty()) {
			bestHistory = newHistory;
			continue;
		}

		if (newHistory.total_accuracy() >= bestHistory.total_accuracy()) {
			cout << ",\tsaving best weights" << endl;
			bestHistory = newHistory;
			this->save_weights();
		}
	}
}

void KeywordExpansionManager::evaluate(int batch_size, int num_threads,
		bool shuffle) {
	__log(__PRETTY_FUNCTION__);
	this->batch_size = batch_size;
	auto training_data = load_data();

	__print(training_data.size());

	if (shuffle)
		::shuffle(training_data);

	auto history = this->total_accuracy(training_data);
	cout << "total accuracy = " << history;
}

vector<ExpansionInstance> KeywordExpansionManager::load_data() {
//	__timer_begin();

	auto json_training_data = readFromStream(
			"../pytext/data/" + section + "/training_corpus.json");

	vector<ExpansionInstance> training_data;
	training_data.reserve(json_training_data.size());
	for (auto &tuple : json_training_data) {
		std::set<string> predSet, goldSet;
		for (auto &e : tuple[1]) {
			predSet.insert(e.asString());
//			cout << e.asString() << endl;
//			predSet.insert(std::toString(e.asString()));
		}

		for (auto &e : tuple[2]) {
			goldSet.insert(e.asString());
//			goldSet.insert(std::toString(e.asString()));
//			cout << std::toString(e.asString()) << endl;
		}

		training_data.push_back( { tuple[0].asString(), predSet, goldSet });
	}

//	__timer_end();
//	if (limit)
//		training_data.resize(limit);
	return training_data;
}

History KeywordExpansionManager::total_accuracy(
		const vector<ExpansionInstance> &training_data,
		AdamOptimizer *optimizer) {
//	__log(__PRETTY_FUNCTION__);
	History accuracy;
	for (int i = 0, size = training_data.size(); i < size; i += batch_size) {
		double start = clock();
		History acc;
		if (optimizer) {
			MapGradient gradient;
			acc = batch_accuracy(subList(training_data, i, i + batch_size),
					&gradient);
			gradient /= acc.count;

			for (auto &field : this->fields)
				optimizer->get_updates(gradient[field], weight[field]);

		} else {
			acc = batch_accuracy(subList(training_data, i, i + batch_size));
		}

		accuracy += acc;

		cout << "batch " << i << ", " << accuracy << " \ttime cost = "
				<< (clock() - start) / CLOCKS_PER_SEC << endl;

	}

	return accuracy;
}

History KeywordExpansionManager::batch_accuracy(
		const ConstSubList<ExpansionInstance> &training_data,
		MapGradient *gradientSum) {
	cout.setf(std::ios::showpoint);

	int size = training_data.size();

	vector<History> histories(this->num_threads);
	vector<MapGradient*> gradients(this->num_threads);
	if (gradientSum) {
		for (int i = 0; i < this->num_threads; ++i) {
			gradients[i] = new MapGradient();
		}
	}

#pragma omp parallel for num_threads(this->num_threads)
	for (int i = 0; i < size; ++i) {
		auto &inst = training_data[i];
		int pid = omp_get_thread_num();
		histories[pid].accuracy[inst.field] += accuracy_per_instance(inst,
				gradients[pid]);
		++histories[pid].count[inst.field];
	}

	History accuracy;
	for (auto &history : histories) {
		accuracy += history;
	}

	if (gradientSum) {
		for (auto gradient : gradients) {
			*gradientSum += *gradient;
		}

		del(gradients);
	}

	return accuracy;
}

double AdamOptimizer::learning_rate() {
	double end_learning_rate = this->lr / 10;
	int global_step = std::min(iterations, this->num_train_steps);
	double learning_rate = (this->lr - end_learning_rate)
			* (1 - global_step / this->num_train_steps) + end_learning_rate;

	if (global_step < num_warmup_steps)
		learning_rate *= global_step * 1.0 / num_warmup_steps;
	return learning_rate;
}

void AdamOptimizer::clip_by_global_norm(Gradient &grads, double clip_norm) {
	double global_norm = grads.l2norm();
	grads *= clip_norm / std::max(global_norm, clip_norm);
}

void AdamOptimizer::get_updates(Gradient &grads, Weight &params) {
	clip_by_global_norm(grads);

	++this->iterations;

	for (auto &name : grads.nonzero_gradient()) {
		auto &p = params[name];
		const auto &g = params[name];
		auto &m = this->ms[name];
		auto &v = this->vs[name];

		m = beta_1 * m + (1 - beta_1) * g;
		v = beta_2 * v + (1 - beta_2) * (g * g);

		if (weight_decay_rate)
			p -= (m / (std::sqrt(v) + epsilon) + weight_decay_rate * p)
					* learning_rate();
		else
			p -= m / (std::sqrt(v) + epsilon) * learning_rate();
	}
}

double& Weight::operator [](const string &name) {
	throw;
}

void ExpansionInstance::accumulate(
		std::map<string, std::map<string, std::map<string, int>>> &count) {
	for (auto &p_word : inputSet) {
//		ensure_true((int)p_word.find_first_of(' ') < 0);
		for (auto &x_word : goldSet) {
//			ensure_true((int)x_word.find_first_of(' ') < 0);
			++count[field][p_word][x_word];
		}
	}
}

AdamOptimizer::AdamOptimizer(int num_train_steps, double lr,
		double weight_decay_rate, double warmup_proportion, double beta_1,
		double beta_2, double epsilon) :
		num_train_steps(num_train_steps), lr(lr), weight_decay_rate(
				weight_decay_rate), warmup_proportion(warmup_proportion), beta_1(
				beta_1), beta_2(beta_2), epsilon(epsilon), num_warmup_steps(
				num_train_steps * warmup_proportion) {
}

extern "C" {
void keyword_expansion_training(const char *section, int epoch, int batch_size,
		double learning_rate, double epsilon, int num_threads, bool shuffle) {
	__print(section);
	__print(epoch);
	__print(batch_size);
	__print(learning_rate);
	__print(epsilon);
	__print(num_threads);
	__print(shuffle);

	KeywordExpansionManager(section).training(epoch, batch_size, learning_rate,
			epsilon, num_threads, shuffle);
}

void keyword_expansion_evaluate(const char *section, int batch_size,
		int num_threads, bool shuffle) {
	__print(section);
	__print(batch_size);
	__print(num_threads);
	__print(shuffle);

	KeywordExpansionManager(section).evaluate(batch_size, num_threads, shuffle);
}

void keyword_expansion_save_weights(const char *section) {
	KeywordExpansionManager(section).save_weights();
}
}

