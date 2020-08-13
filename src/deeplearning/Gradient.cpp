#include "Gradient.h"
#include "matrix.h"

void VectorGradient::nonzero_gradient(std::vector<std::string> &keys) {

	for (int k = 0; k < vec.size(); ++k) {
		for (auto &keyStr : vec[k]->nonzero_gradient()) {
			ostringstream cout;
			cout << "[" << k << "]" << keyStr;
			keys.push_back(cout.str());
		}
	}
}

void MapGradient::nonzero_gradient(vector<string> &keys) {
	for (auto &tuple : map) {
		auto &k = tuple.first;
		for (auto &keyStr : tuple.second->nonzero_gradient()) {
			keys.push_back("[" + k + "]" + keyStr);
		}
	}
}

void ScalarGradient::nonzero_gradient(vector<string> &keys) {
	if (!*this)
		return;
	ostringstream cout;
	cout << "=" << this->scalar;
	keys.push_back(cout.str());
}

Gradient& NullGradient::operator *=(double rhs) {
//	__log(__PRETTY_FUNCTION__);
	return *this;
}

Gradient& NullGradient::operator /=(double rhs) {
	return *this;
}

Gradient& ScalarGradient::operator *=(double rhs) {
//	__log(__PRETTY_FUNCTION__);
	scalar *= rhs;
	return *this;
}

Gradient& ScalarGradient::operator /=(double rhs) {
	scalar /= rhs;
	return *this;
}

Gradient& VectorGradient::operator *=(double rhs) {
//	__log(__PRETTY_FUNCTION__);
	for (auto gradient : this->vec) {
		*gradient *= rhs;
	}
	return *this;
}

Gradient& VectorGradient::operator /=(double rhs) {
	for (auto gradient : this->vec) {
		*gradient /= rhs;
	}
	return *this;
}

Gradient& MapGradient::operator *=(double rhs) {
//	__log(__PRETTY_FUNCTION__);
	for (auto &tuple : this->map) {
		*tuple.second *= rhs;
	}
	return *this;
}

Gradient& MapGradient::operator /=(double rhs) {
	for (auto &tuple : this->map) {
		*tuple.second /= rhs;
	}
	return *this;
}

Gradient::~Gradient() {
}

Gradient& ScalarGradient::operator +=(Gradient &rhs) {
	auto doubleGradient = dynamic_cast<ScalarGradient*>(&rhs);
	if (doubleGradient) {
		this->scalar += doubleGradient->scalar;
		return *this;
	}
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return *this;
}

Gradient& ScalarGradient::operator -=(Gradient &rhs) {
	auto doubleGradient = dynamic_cast<ScalarGradient*>(&rhs);
	if (doubleGradient) {
		this->scalar -= doubleGradient->scalar;
		return *this;
	}

	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return *this;
}

Gradient& ScalarGradient::operator +=(double rhs) {
	scalar += rhs;
	return *this;
}

Gradient& ScalarGradient::operator -=(double rhs) {
	scalar -= rhs;
	return *this;
}

NullGradient::NullGradient(Gradient *parent) :
		parent(parent) {
}

MapGradient::~MapGradient() {
	for (auto &tuple : this->map) {
		delete tuple.second;
	}
}

std::map<string, std::map<string, double>>& operator +=(
		std::map<string, std::map<string, double>> &lhs,
		const MapGradient &rhs) {
	__cout(__PRETTY_FUNCTION__);
	for (auto &tuple : rhs.map) {
		lhs[tuple.first] += *tuple.second;
	}
	return lhs;
}

std::map<string, std::map<string, double>>& operator -=(
		std::map<string, std::map<string, double>> &lhs,
		const MapGradient &rhs) {
	__cout(__PRETTY_FUNCTION__);
	for (auto &tuple : rhs.map) {
		lhs[tuple.first] -= *tuple.second;
	}
	return lhs;
}

vector<vector<double>>& operator +=(vector<vector<double>> &lhs,
		const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const VectorGradient*>(&rhs);
	if (gradient)
		return lhs += *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return lhs;
}

vector<vector<double>>& operator -=(vector<vector<double>> &lhs,
		const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const VectorGradient*>(&rhs);
	if (gradient)
		return lhs -= *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return lhs;
}

vector<vector<double>>& operator +=(vector<vector<double>> &lhs,
		const VectorGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	int size = rhs.vec.size();
	if (lhs.size() < size)
		lhs.resize(size);
	for (size_t key = 0; key < size; ++key) {
		lhs[key] += *rhs.vec[key];
	}
	return lhs;
}

vector<vector<double>>& operator -=(vector<vector<double>> &lhs,
		const VectorGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	int size = rhs.vec.size();
	if (lhs.size() < size)
		lhs.resize(size);

	for (size_t key = 0; key < size; ++key) {
		lhs[key] -= *rhs.vec[key];
	}
	return lhs;
}

std::map<string, double>& operator +=(std::map<string, double> &lhs,
		const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const MapGradient*>(&rhs);
	if (gradient)
		return lhs += *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return lhs;
}

std::map<string, double>& operator -=(std::map<string, double> &lhs,
		const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const MapGradient*>(&rhs);
	if (gradient)
		return lhs -= *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return lhs;
}

ScalarGradient::operator double() {
	return scalar;
}

NullGradient::operator double() {
	__log(__PRETTY_FUNCTION__);
	return 0;
}

Gradient::operator double() {
	__log(__PRETTY_FUNCTION__);
	throw;
}

std::map<string, double>& operator +=(std::map<string, double> &lhs,
		const MapGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	for (auto &tuple : rhs.map) {
		lhs[tuple.first] += *tuple.second;
	}
	return lhs;
}

std::map<string, double>& operator -=(std::map<string, double> &lhs,
		const MapGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	for (auto &tuple : rhs.map) {
		lhs[tuple.first] -= *tuple.second;
	}
	return lhs;
}

vector<double>& operator +=(vector<double> &lhs, const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const VectorGradient*>(&rhs);
	if (gradient)
		return lhs += *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return lhs;
}

vector<double>& operator -=(vector<double> &lhs, const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const VectorGradient*>(&rhs);
	if (gradient)
		return lhs -= *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return lhs;
}

vector<double>& operator +=(vector<double> &lhs, const VectorGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	int size = rhs.vec.size();
	if (lhs.size() < size)
		lhs.resize(size);

	for (size_t key = 0; key < size; ++key) {
		lhs[key] += *rhs.vec[key];
	}
	return lhs;
}

vector<double>& operator -=(vector<double> &lhs, const VectorGradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	int size = rhs.vec.size();
	if (lhs.size() < size)
		lhs.resize(size);

	for (size_t key = 0; key < size; ++key) {
		lhs[key] -= *rhs.vec[key];
	}
	return lhs;
}

std::map<string, std::map<string, double>>& operator +=(
		std::map<string, std::map<string, double>> &lhs, const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const MapGradient*>(&rhs);
	if (gradient)
		return lhs += *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return lhs;
}

std::map<string, std::map<string, double>>& operator -=(
		std::map<string, std::map<string, double>> &lhs, const Gradient &rhs) {
//	__log(__PRETTY_FUNCTION__);
	auto gradient = dynamic_cast<const MapGradient*>(&rhs);
	if (gradient)
		return lhs -= *gradient;
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return lhs;
}

Gradient& MapGradient::operator [](const string &key) {
//	__log(__PRETTY_FUNCTION__);
	if (this->map.count(key)) {
		return *map[key];
	}
	auto newNode = new NullGradient(this);
	this->map[key] = newNode;
	return *newNode;
}

Gradient& MapGradient::operator +=(Gradient &rhs) {
	auto mapGradient = dynamic_cast<MapGradient*>(&rhs);
	if (mapGradient) {
		for (auto &tuple : mapGradient->map) {
			(*this)[tuple.first] += *tuple.second;
		}
		return *this;
	}
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return *this;
}

Gradient& MapGradient::operator -=(Gradient &rhs) {
	auto mapGradient = dynamic_cast<MapGradient*>(&rhs);
	if (mapGradient) {
		for (auto &tuple : mapGradient->map) {
			(*this)[tuple.first] -= *tuple.second;
		}
		return *this;
	}

	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return *this;
}

Gradient& MapGradient::operator +=(double rhs) {
	__log(__PRETTY_FUNCTION__);
	throw;
}

Gradient& MapGradient::operator -=(double rhs) {
	__log(__PRETTY_FUNCTION__);
	throw;
}

VectorGradient::~VectorGradient() {
	for (auto gradient : this->vec) {
		delete gradient;
	}
}

Gradient& VectorGradient::operator [](const string &key) {
	__log(__PRETTY_FUNCTION__);
	throw;
}

Gradient& ScalarGradient::operator [](const string &key) {
	__log(__PRETTY_FUNCTION__);
	throw;
}

Gradient& ScalarGradient::operator [](int key) {
	__log(__PRETTY_FUNCTION__);
	throw;
}

Gradient& MapGradient::operator [](int key) {
	__log(__PRETTY_FUNCTION__);
	throw;
}

Gradient& VectorGradient::operator [](int key) {
	if (key >= vec.size()) {
		int start = vec.size();
		vec.resize(key + 1);

		for (int i = start; i <= key; ++i) {
			vec[i] = new NullGradient(this);
		}
	}

	return *vec[key];
}

Gradient& VectorGradient::operator +=(Gradient &rhs) {
	auto vectorGradient = dynamic_cast<VectorGradient*>(&rhs);
	if (vectorGradient) {
		int size = vectorGradient->vec.size();
		for (int i = 0; i < size; ++i) {
			(*this)[i] += *vectorGradient->vec[i];
		}
		return *this;
	}

	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return *this;
}

Gradient& VectorGradient::operator -=(Gradient &rhs) {
	auto vectorGradient = dynamic_cast<VectorGradient*>(&rhs);
	if (vectorGradient) {
		int size = vectorGradient->vec.size();
		for (int i = 0; i < size; ++i) {
			(*this)[i] -= *vectorGradient->vec[i];
		}
		return *this;
	}
	ensure_true(dynamic_cast<const NullGradient*>(&rhs));
	return *this;
}

Gradient& VectorGradient::operator +=(double rhs) {
	__log(__PRETTY_FUNCTION__);
	throw;
}

Gradient& VectorGradient::operator -=(double rhs) {
	__log(__PRETTY_FUNCTION__);
	throw;
}

Gradient& NullGradient::operator +=(Gradient &rhs) {
	if (dynamic_cast<NullGradient*>(&rhs)) {
		return *this;
	}

	return parent->reset(this, rhs.clone());
}

Gradient& NullGradient::operator -=(Gradient &rhs) {
	if (dynamic_cast<NullGradient*>(&rhs)) {
		return *this;
	}

	return parent->reset(this, &-rhs);
}

Gradient* NullGradient::clone() const {
	return new NullGradient(parent);
}

Gradient* ScalarGradient::clone() const {
	return new ScalarGradient(this->scalar);
}

Gradient* VectorGradient::clone() const {
	int size = this->vec.size();
	std::vector<Gradient*> dict(size);

	for (int index = 0; index < size; ++index) {
		dict[index] = this->vec[index]->clone();
	}
	return new VectorGradient(dict);
}

Gradient* MapGradient::clone() const {
	std::map<string, Gradient*> dict;
	for (auto &tuple : this->map) {
		dict[tuple.first] = tuple.second->clone();
	}
	return new MapGradient(dict);
}

ScalarGradient::ScalarGradient(double value) :
		scalar(value) {
}

Gradient& NullGradient::operator +=(double rhs) {
	return parent->reset(this, new ScalarGradient { rhs });
}

Gradient& NullGradient::operator -=(double rhs) {
	return parent->reset(this, new ScalarGradient { -rhs });
}

Gradient& NullGradient::operator [](const string &key) {
	std::map<string, Gradient*> dict { { key, this } };
	auto parent = this->parent;
	parent->reference(this) = new MapGradient(dict);
	return *this;
}

Gradient& NullGradient::operator [](int key) {
	vector<Gradient*> vec(key + 1);
	for (int i = 0; i < key; ++i) {
		vec[i] = new NullGradient(nullptr);
	}
	vec[key] = this;
	auto parent = this->parent;
	parent->reference(this) = new VectorGradient(vec);
	return *this;
}

Gradient& Gradient::reset(Gradient *self, Gradient *replacement) {
	reference(self) = replacement;
	delete self;
	return *replacement;
}

Gradient*& Gradient::reference(Gradient *g) {
	__log(__PRETTY_FUNCTION__);
	throw;
}

Gradient*& MapGradient::reference(Gradient *g) {
	for (auto &tuple : map) {
		if (tuple.second == g) {
			return tuple.second;
		}
	}
	__log(__PRETTY_FUNCTION__);
	throw;
}

Gradient*& VectorGradient::reference(Gradient *g) {
	for (auto &v : vec) {
		if (v == g) {
			return v;
		}
	}
	__log(__PRETTY_FUNCTION__);
	throw;
}

void NullGradient::print(std::ostream &cout) const {
	cout << "null";
}

void MapGradient::print(std::ostream &cout) const {
	cout << this->map;
//	cout << "{";
//	bool print_comma = false;
//	for (auto &tuple : this->map) {
//		if (print_comma)
//			cout << ", ";
//		else
//			print_comma = true;
//		cout << tuple.first << " : " << *tuple.second;
//	}
//	cout << "}";
}

void VectorGradient::print(std::ostream &cout) const {
	cout << vec;
//	cout << "[";
//	bool print_comma = false;
//	for (auto gradient : this->vec) {
//		if (print_comma)
//			cout << ", ";
//		else
//			print_comma = true;
//		ensure_true(gradient);
//		cout << *gradient;
//	}
//	cout << "]";
}

void ScalarGradient::print(std::ostream &cout) const {
	cout << this->scalar;
}

std::ostream& operator <<(std::ostream &cout, const Gradient &p) {
	p.print(cout);
	return cout;
}

std::ostream& operator <<(std::ostream &cout, Gradient * const p) {
	return cout << *p;
}

Gradient& NullGradient::operator -() const {
	return *new NullGradient(nullptr);
}

Gradient& ScalarGradient::operator -() const {
	return *new ScalarGradient(-this->scalar);
}

Gradient& VectorGradient::operator -() const {
	int size = this->vec.size();
	std::vector<Gradient*> vec(size);
	for (int index = 0; index < size; ++index) {
		vec[index] = &-*this->vec[index];
	}
	return *new VectorGradient(vec);
}

Gradient& MapGradient::operator -() const {
	std::map<string, Gradient*> dict;
	for (auto &tuple : this->map) {
		dict[tuple.first] = &-*tuple.second;
	}
	return *new MapGradient(dict);
}

void Gradient::set_parent(Gradient*) {
}

void NullGradient::set_parent(Gradient *parent) {
	this->parent = parent;
}

VectorGradient::VectorGradient(std::vector<Gradient*> &dict) :
		vec(dict) {
	for (auto g : dict) {
		g->set_parent(this);
	}
}

MapGradient::MapGradient() {
}

MapGradient::MapGradient(std::map<string, Gradient*> &dict) :
		map(dict) {
	for (auto &tuple : dict) {
		tuple.second->set_parent(this);
	}
}

bool NullGradient::operator !() const {
	return true;
}

bool ScalarGradient::operator !() const {
	return std::abs(this->scalar) < 1e-10;
}

bool VectorGradient::operator !() const {
	for (auto g : this->vec) {
		if (!*g) {
			continue;
		}
		return false;
	}
	return true;
}

bool MapGradient::operator !() const {
	for (auto &tuple : this->map) {
		if (!*tuple.second)
			continue;
		return false;
	}
	return true;
}

double NullGradient::max() const {
	return 0;
}

double ScalarGradient::max() const {
	return this->scalar;
}

double VectorGradient::max() const {
	double max = -oo;
	for (auto g : this->vec) {
		auto max_ = g->max();
		if (max_ > max)
			max = max_;
	}
	return max;
}

double MapGradient::max() const {
	double max = -oo;
	for (auto &tuple : this->map) {
		auto max_ = tuple.second->max();
		if (max_ > max)
			max = max_;
	}
	return max;
}

double NullGradient::min() const {
	return 0;
}

double ScalarGradient::min() const {
	return this->scalar;
}

double VectorGradient::min() const {
	double min = oo;
	for (auto g : this->vec) {
		auto min_ = g->min();
		if (min_ < min)
			min = min_;
	}
	return min;
}

double MapGradient::min() const {
	double min = oo;
	for (auto &tuple : this->map) {
		auto min_ = tuple.second->min();
		if (min_ < min)
			min = min_;
	}
	return min;
}


bool MapGradient::operator ==(const Gradient &rhs) const {
	return rhs == *this;
}

bool VectorGradient::operator ==(const Gradient &rhs) const {
	return rhs == *this;
}

bool ScalarGradient::operator ==(const Gradient &rhs) const {
	return rhs == *this;
}

bool NullGradient::operator ==(const Gradient &rhs) const {
	return rhs == *this;
}

double Gradient::l2norm()const{
	return std::sqrt(this->square());
}

double VectorGradient::square()const{
	double sum = 0;
	for (auto g : this->vec){
		sum += g->square();
	}
	return sum;
}

double MapGradient::square()const{
	double sum = 0;
	for (auto &tuple : this->map){
		sum += tuple.second->square();
	}
	return sum;
}

double ScalarGradient::square()const{
	return this->scalar * this->scalar;
}

double NullGradient::square()const{
	return 0;
}
