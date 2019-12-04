#include "NERTaggerDict.h"

NERTagger::object& NERTaggerDict::getTagger(const string &service) {
	if (!dict.count(service)) {
		cout << "in " << __PRETTY_FUNCTION__ << endl;
		BinaryReader dis(nerBinary(service));
		dict[service] = new NERTagger(dis);
	}

	return dict[service];

}

vector<int> NERTaggerDict::get_repertoire_code(const string &service,
		const String &text) {
	vector<int> repertoire_code;
	return repertoire_code;
}

vector<int>& NERTaggerDict::predict(const string &service, const String &text,
		vector<int> &repertoire_code) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;

	return getTagger(service)->predict(text, repertoire_code);
}

vector<vector<vector<double>>>& NERTaggerDict::_predict(const string &service,
		const String &text, vector<int> &repertoire_code,
		vector<vector<vector<double>>> &arr) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	return getTagger(service)->_predict(text, repertoire_code, arr);
}

extern "C" vector<int>& cpp_ner(const char *service, const word *text,
		vector<int> &repertoire_code) {
	cout << "in " << __FUNCTION__ << endl;
	cout << "service = " << service << endl;
	String words = text;
	cout << "text.size = " << words.size() << endl;
	cout << "repertoire_code.size = " << repertoire_code.size() << endl;
	cout << "repertoire_code = " << repertoire_code << endl;
	assert(words.size() == repertoire_code.size());
	auto &ret = NERTaggerDict::predict(service, words, repertoire_code);
	cout << "repertoire_code = " << repertoire_code << endl;
	cout << "ret = " << ret << endl;
	if (&ret == &repertoire_code) {
		cout << "same memory" << endl;
	} else {
		cout << "not same memory" << endl;
	}
	cout << &repertoire_code << endl;
	return ret;
}

extern "C" vector<vector<vector<double>>>& _cpp_ner(const char *service,
		const word *text, vector<int> &repertoire_code,
		vector<vector<vector<double>>> &arr) {
//	arr.resize(0);
	cout << "in " << __FUNCTION__ << endl;
	cout << "service = " << service << endl;
	String words = text;
	cout << "text.size = " << words.size() << endl;
	cout << "repertoire_code.size = " << repertoire_code.size() << endl;
	cout << "repertoire_code = " << repertoire_code << endl;
	assert(words.size() == repertoire_code.size());

	return NERTaggerDict::_predict(service, words, repertoire_code, arr);
}

extern "C" void cpp_ner_initialize(const char *service) {
	NERTaggerDict::getTagger(service);
}

unordered_map<string, NERTagger::object> NERTaggerDict::dict;
