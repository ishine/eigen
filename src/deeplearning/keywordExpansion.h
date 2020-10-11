#pragma once
#include "utility.h"
#include "Gradient.h"

#define USING_P_GIVEN_X 1

struct History {
	std::map<string, double> accuracy;
	std::map<string, int> count;

	bool empty() const {
		for (auto &tuple : count) {
			if (!tuple.second)
				return true;
		}
		return false;
	}

	History() :
			accuracy( { { "title", 0 }, { "abst", 0 } }), count( {
					{ "title", 0 }, { "abst", 0 } }) {
	}

	friend std::ostream& operator <<(std::ostream &cout, const History &h) {
		bool print_comma = false;
		for (auto &tuple : h.accuracy) {
			auto &field = tuple.first;
			if (print_comma)
				cout << ", ";
			else
				print_comma = true;

			cout << "accuracy[" << field << "] = " << std::left << std::setw(8)
					<< std::fixed << std::setprecision(6) << std::setfill('0')
					<< 100 * h.accuracy.at(field) / h.count.at(field);
		}
		return cout;
	}

	double total_accuracy() const {
		double sum = 0;
		for (auto &tuple : accuracy) {
			sum += tuple.second / count.at(tuple.first);
		}
		return sum;
	}

	History& operator +=(const History &rhs) {
		accuracy += rhs.accuracy;
		count += rhs.count;
		return *this;
	}
};

struct HyperParameter {
	double bias, sigma; //, pi;

	HyperParameter& operator *=(double rhs) {
		bias *= rhs;
		sigma *= rhs;
//		pi *= rhs;
		return *this;
	}

	double& operator [](const string &key) {
		if (key == "bias")
			return bias;
		else if (key == "sigma")
			return sigma;
//		else if (key == "pi")
//			return pi;
		else
			throw;
	}

	HyperParameter& operator +=(const Gradient &rhs);
	HyperParameter& operator +=(const MapGradient &rhs);

	HyperParameter& operator -=(const Gradient &rhs);
	HyperParameter& operator -=(const MapGradient &rhs);

	friend BinaryFile& operator >>(BinaryFile &cin, HyperParameter &hyper) {
		cin >> hyper.bias;
		cin >> hyper.sigma;
//		cin >> hyper.pi;
		return cin;
	}

	friend std::ostream& operator <<(std::ostream &cout,
			const HyperParameter &hyper) {
		return cout << hyper.to_map();
	}

	std::map<string, double> to_map() const {
		return { {"bias", bias}, {"sigma", sigma}};
	}
};

struct Weight {
#if USING_P_GIVEN_X
	std::map<string, std::map<string, double>> p_given_x;
#else
	std::map<string, std::map<string, double>> x_given_p;
#endif

	double& operator [](const string &name);

	std::map<string, double> x_probability;
	double oov_probability = -10000;

//	std::map<string, double> p_probability;

	Weight& operator +=(const Gradient &rhs);
	Weight& operator -=(const Gradient &rhs);

	Weight& operator +=(const MapGradient &rhs);
	Weight& operator -=(const MapGradient &rhs);

	double sigma(const std::vector<string> &p_words, const string &x_word,
			int n) const;
};

std::map<string, Weight>& operator +=(std::map<string, Weight> &lhs,
		const Gradient &rhs);
std::map<string, Weight>& operator -=(std::map<string, Weight> &lhs,
		const Gradient &rhs);

std::map<string, Weight>& operator +=(std::map<string, Weight> &lhs,
		const MapGradient &rhs);
std::map<string, Weight>& operator -=(std::map<string, Weight> &lhs,
		const MapGradient &rhs);

vector<HyperParameter>& operator +=(vector<HyperParameter> &lhs,
		const Gradient &rhs);
vector<HyperParameter>& operator +=(vector<HyperParameter> &lhs,
		const VectorGradient &rhs);

vector<HyperParameter>& operator -=(vector<HyperParameter> &lhs,
		const Gradient &rhs);
vector<HyperParameter>& operator -=(vector<HyperParameter> &lhs,
		const VectorGradient &rhs);

struct ExpansionInstance {
	static vector<string> as_vector(const std::set<string> &s) {
		vector < string > v(s.begin(), s.end());
		for (auto &s : v) {
			std::ostringstream cout;
			cout << "'" << s << "'";
			s = cout.str();
		}
		return v;
	}

	string field;
	std::set<string> inputSet, goldSet;
	void accumulate(
			std::map<string, std::map<string, std::map<string, int>>> &count);

	friend std::ostream& operator <<(std::ostream &cout,
			const ExpansionInstance &inst) {

		return cout << "('" << inst.field << "', " << as_vector(inst.inputSet)
				<< ", " << as_vector(inst.goldSet) << ")";
	}

};

struct AdamOptimizer {
	AdamOptimizer(int num_train_steps, double lr = 1e-3,
			double weight_decay_rate = 0.01, double warmup_proportion = 0.1,
			double beta_1 = 0.9, double beta_2 = 0.999, double epsilon = 1e-6);

	const int num_train_steps;
	const double lr;
	const double weight_decay_rate;

	const double warmup_proportion;

	const double beta_1, beta_2;
	const double epsilon;

	const int num_warmup_steps;

	int iterations = 0;
	Weight ms, vs;
	double learning_rate();
	void clip_by_global_norm(Gradient &grads, double clip_norm = 1);
	void get_updates(Gradient &grads, Weight &params);
};

struct KeywordExpansionManager {
	KeywordExpansionManager(const string &section,
			double oov_probability = -15);
	static const string fields[];
	const string section;
	std::map<string, Weight> weight;
	std::map<string, std::map<string, std::map<string, int>>> count;

	int batch_size = 128;
	double epsilon = 1e-4;
	int num_threads = 8;

	void save_weights() const;
	void load_weights();
	void init_weights();

//    # determine the gradient according to the loss incurred
	void sigma_gradient(Gradient &gradient, const string &field,
			const std::vector<string> &p_words, const string &x_word,
			double delta) const;

//#     x -= self.learning_rate * diff(L, x)
	double accuracy_per_instance(const ExpansionInstance &inst,
			Gradient *gradient = nullptr) const;

	std::set<string> predict(const string &field, std::set<string> &inputSet);

	void training(int epoch = 4, int batch_size = 64, double learning_rate =
			1e-3, double epsilon = 1e-4, int num_threads = 8, bool shuffle =
			true);

	vector<ExpansionInstance> load_data();

	History total_accuracy(const vector<ExpansionInstance> &training_data,
			AdamOptimizer *optimizer = nullptr);

	History batch_accuracy(const ConstSubList<ExpansionInstance> &training_data,
			MapGradient *gradientSum = nullptr);

	void evaluate(int batch_size, int num_threads, bool shuffle);

	bool does_weight_exists();
	bool does_vocab_count_exists();

	void build_vocab_count();
};

