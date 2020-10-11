#include "utility.h"

struct Gradient;
struct MapGradient;
struct VectorGradient;
struct ScalarGradient;
struct NullGradient;

std::map<string, std::map<string, double>>& operator -=(
		std::map<string, std::map<string, double>> &lhs, const Gradient &rhs);
std::map<string, std::map<string, double>>& operator -=(
		std::map<string, std::map<string, double>> &lhs,
		const MapGradient &rhs);
std::map<string, double>& operator -=(std::map<string, double> &lhs,
		const Gradient &rhs);
std::map<string, double>& operator -=(std::map<string, double> &lhs,
		const MapGradient &rhs);

vector<vector<double>>& operator -=(vector<vector<double>> &lhs,
		const Gradient &rhs);
vector<vector<double>>& operator -=(vector<vector<double>> &lhs,
		const VectorGradient &rhs);

vector<double>& operator -=(vector<double> &lhs, const Gradient &rhs);
vector<double>& operator -=(vector<double> &lhs, const VectorGradient &rhs);

std::map<string, std::map<string, double>>& operator +=(
		std::map<string, std::map<string, double>> &lhs, const Gradient &rhs);
std::map<string, std::map<string, double>>& operator +=(
		std::map<string, std::map<string, double>> &lhs,
		const MapGradient &rhs);
std::map<string, double>& operator +=(std::map<string, double> &lhs,
		const Gradient &rhs);
std::map<string, double>& operator +=(std::map<string, double> &lhs,
		const MapGradient &rhs);

vector<vector<double>>& operator +=(vector<vector<double>> &lhs,
		const Gradient &rhs);
vector<vector<double>>& operator +=(vector<vector<double>> &lhs,
		const VectorGradient &rhs);

vector<double>& operator +=(vector<double> &lhs, const Gradient &rhs);
vector<double>& operator +=(vector<double> &lhs, const VectorGradient &rhs);

#define virtual_vptrs(virtual, _0_)\
virtual Gradient& operator [](const string &key)_0_;\
virtual Gradient& operator [](int key)_0_;\
virtual Gradient& operator +=(Gradient &rhs)_0_;\
virtual Gradient& operator +=(double rhs)_0_;\
virtual Gradient& operator -=(Gradient &rhs)_0_;\
virtual Gradient& operator -=(double rhs)_0_;\
virtual Gradient& operator /=(double rhs)_0_;\
virtual Gradient& operator *=(double rhs)_0_;\
virtual bool operator ==(const Gradient &rhs) const	_0_;\
virtual void print(std::ostream &cout) const _0_;\
virtual Gradient *clone() const _0_;\
virtual Gradient &operator -() const _0_;\
virtual bool operator ! ()const _0_;\
virtual double max()const _0_;\
virtual double min()const _0_;\
virtual double square()const _0_;\

struct Gradient {
	virtual ~Gradient();

	virtual_vptrs(virtual, =0)
	;
	virtual operator double();
	virtual void set_parent(Gradient *parent);
	virtual Gradient*& reference(Gradient *g);
	Gradient& reset(Gradient *g, Gradient *replacement);

	vector<std::string> nonzero_gradient() {
		vector < string > keys;
		this->nonzero_gradient(keys);
		return keys;
	}

	virtual void nonzero_gradient(vector<string> &keys) {
	}

	bool operator !=(const Gradient &rhs) const {
		return !(*this == rhs);
	}

	virtual bool operator ==(const MapGradient &rhs) const {
		return false;
	}
	virtual bool operator ==(const VectorGradient &rhs) const {
		return false;
	}
	virtual bool operator ==(const ScalarGradient &rhs) const {
		return false;
	}
	virtual bool operator ==(const NullGradient &rhs) const {
		return false;
	}

	double l2norm()const;
};

struct MapGradient: Gradient {
	MapGradient(std::map<string, Gradient*> &dict);
	MapGradient();

	std::map<string, Gradient*> map;
	~MapGradient();

	Gradient*& reference(Gradient *g);

	virtual_vptrs(,)
	;

	using Gradient::nonzero_gradient;
	void nonzero_gradient(vector<string> &keys);

	bool operator ==(const MapGradient &rhs) const {
		if (keys(map) != keys(rhs.map))
			return false;

		for (auto &tuple : map) {
			Gradient *g = tuple.second;
			Gradient *_g = rhs.map.at(tuple.first);
			if (*g != *_g)
				return false;
		}
		return true;
	}

	Gradient& operator /=(const std::map<string, int> &rhs) {
		for (auto &tuple : rhs) {
			(*this)[tuple.first] /= tuple.second;
		}
		return *this;
	}
};

struct VectorGradient: Gradient {

	VectorGradient(vector<Gradient*> &dict);

	std::vector<Gradient*> vec;

	~VectorGradient();

	Gradient*& reference(Gradient *g);

	virtual_vptrs(,)
	;

	void nonzero_gradient(std::vector<std::string> &keys);

	bool operator ==(const VectorGradient &rhs) const {
		if (vec.size() != rhs.vec.size())
			return false;

		for (int i = 0; i < vec.size(); ++i) {
			if (*vec[i] != *rhs.vec[i])
				return false;
		}
		return true;
	}
};

struct ScalarGradient: Gradient {

	ScalarGradient(double value);
	double scalar;

	operator double();

	virtual_vptrs(,)
	;

	void nonzero_gradient(vector<string> &keys);

	bool operator ==(const ScalarGradient &rhs) const {
		return scalar == rhs.scalar;
	}

};

struct NullGradient: Gradient {

	NullGradient(Gradient *parent);
	operator double();
	Gradient *parent;

	virtual_vptrs(,)
	;

	void set_parent(Gradient *parent);

	bool operator ==(const NullGradient &rhs) const {
		return true;
	}
};

std::ostream& operator <<(std::ostream &cout, const Gradient &p);

std::ostream& operator <<(std::ostream &cout, Gradient *const p);
