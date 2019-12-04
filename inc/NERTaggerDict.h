#include "Utility.h"
#include "NERTagger.h"
struct NERTaggerDict {

	static unordered_map<string, NERTagger::object> dict;

	static NERTagger::object &getTagger(const string &service);

	static vector<int> get_repertoire_code(const string &service,
			const String &text);

	static vector<int> &predict(const string &service, const String &text, vector<int> &);
	static vector<vector<vector<double>>> &_predict(const string &service, const String &text, vector<int> &,vector<vector<vector<double>>> &);
};
