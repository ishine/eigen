#include <utility.h>
#include <string>

#include<fstream>

using namespace std;

string& workingDirectory() {
#ifdef _WIN32
	static string workingDirectory = "D:/360/solution/";
#else
	static string workingDirectory = "/home/zhoulizhi/solution/";
#endif
	return workingDirectory;
}

//string get_workingDirectory() {
//
//	int index = workingDirectory.find_last_of("/\\");
//
//	workingDirectory = workingDirectory.substr(0, index);
//
//	workingDirectory += "/../";
//
//	cout << "workingDirectory = " << workingDirectory << endl;
//	return workingDirectory;
//}

string& modelsDirectory() {
	static string modelsDirectory = workingDirectory() + "models/";
	return modelsDirectory;
}

string& cnModelsDirectory() {
	static string cnModelsDirectory = modelsDirectory() + "cn/";
	return cnModelsDirectory;
}

string& nerModelsDirectory() {
	static string nerModelsDirectory = cnModelsDirectory() + "ner/";
	return nerModelsDirectory;

}

string& serviceModelsDirectory() {
	static string serviceModelsDirectory = cnModelsDirectory() + "gru_data/";
	return serviceModelsDirectory;
}

string& serviceBinary() {
	static string serviceBinary = serviceModelsDirectory() + "service.bin";
	return serviceBinary;
}

string nerBinary(const string &service) {
	return nerModelsDirectory() + service + ".bin";
}

vector<string> openAttribute(const H5::Group &group, const char *name);

HDF5Reader::HDF5Reader(const string &s_FilePath) :
		hdf5(s_FilePath, H5F_ACC_RDONLY), layer_names(
				openAttribute(hdf5, "layer_names")), layer_index(0), group(
				hdf5.openGroup(layer_names[layer_index])), weight_index(-1) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;

//	this->s_FilePath = s_FilePath;
}

HDF5Reader& HDF5Reader::operator >>(Vector &arr) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	std::pair<vector<int>, vector<double>> tuple;
	*this >> tuple;
	auto &shape = tuple.first;
	auto &weight = tuple.second;
	assert(shape.size() == 1);

	int dimension = shape[0];
	cout << "x = " << dimension << endl;

	arr.resize(dimension);
	assert(arr.cols() == dimension);
	for (int i = 0; i < dimension; ++i) {
		arr[i] = weight[i];
	}
	return *this;
}

//https://bitbucket.hdfgroup.org/projects/HDFFV/repos/hdf5/browse
//https://portal.hdfgroup.org/display/support/HDF5+1.10.5
//https://blog.csdn.net/renyhui/article/details/77735314
//http://web.mit.edu/fwtools_v3.1.0/www/H5.intro.html#Intro-TOC
//http://web.mit.edu/fwtools_v3.1.0/www/cpplus_RM/files.html
//void HDF5Reader::read_hdf5() {
//	vector<int> content;
//	int x;
//	dis.seekg(0, std::ios::end);
//	int size = dis.tellg();
//	cout << "size = " << size << endl;
//	dis.seekg(0, std::ios::beg);
//
//	for (int i = 0; i < size / 4; ++i) {
//		this->dis.read((char*) &x, 4);
//		content.push_back(x);
//	}
//
//	int _size = dis.tellg();
//	assert(_size == size);
//	cout << "finish reading " << endl;
//}

//void* HDF5Reader::read(void *x, int size) {
//	char *arr = (char*) x;
//	this->dis.read(arr, size);
//	for (int i = 0, length = size / 2; i < length; ++i) {
//		std::swap(arr[i], arr[size - 1 - i]);
//	}
//	return x;
//}
//
//int HDF5Reader::read(int &x) {
//	this->read(&x, sizeof(int));
//	return x;
//}

//double HDF5Reader::read(double &x) {
//	this->read(&x, sizeof(double));
//	return x;
//}
//
//float HDF5Reader::read(float &x) {
//	this->read(&x, sizeof(float));
//	return x;
//}
//
//word HDF5Reader::read(word &x) {
//	this->read(&x, sizeof(word));
//	return x;
//}
//

HDF5Reader& HDF5Reader::operator >>(Matrix &arr) {
	cout << "in " << __PRETTY_FUNCTION__ << endl;
	std::pair<vector<int>, vector<double>> tuple;
	*this >> tuple;
	auto &shape = tuple.first;
	auto &weight = tuple.second;
	assert(shape.size() == 2);

	int dimension0 = shape[0];
	int dimension1 = shape[1];
	cout << "x = " << dimension0 << ", " << "y = " << dimension1 << endl;

	arr.resize(dimension0, dimension1);
	int index = 0;
	for (int i0 = 0; i0 < dimension0; ++i0) {
		for (int i1 = 0; i1 < dimension1; ++i1) {
			arr(i0, i1) = weight[index++];
		}
	}
	return *this;
}

//unordered_map<word, int>& HDF5Reader::read(unordered_map<word, int> &char2id) {
//	cout << "in " << __PRETTY_FUNCTION__ << endl;
//	int length;
//	read(length);
//
//	cout << "length = " << length << endl;
////	print_primitive_type_size();
//
//	assert(length >= 0);
//
//	for (int i = 0; i < length; ++i) {
//		word key;
//		read(key);
//		read(char2id[key]);
//
////		cout << "char2id[" << key << "] = " << char2id[key] << endl;
//	}
//	return char2id;
//}

HDF5Reader& HDF5Reader::operator >>(
		std::pair<vector<int>, vector<double>> &tuple) {
	std::pair<vector<int>, vector<double>>& read_keras_model(
			const H5::H5File &file, const H5::Group &group,
			const string &weight_name,
			std::pair<vector<int>, vector<double>> &tuple);

	while (true) {
		if (++weight_index < (int) weight_names.size()) {
			break;
		}

		assert(++layer_index < (int ) layer_names.size());
		group = hdf5.openGroup(layer_names[layer_index]);

		this->weight_names = openAttribute(group, "weight_names");
		weight_index = -1;
	}
	read_keras_model(hdf5, group, weight_names[weight_index], tuple);
	return *this;
}

HDF5Reader& HDF5Reader::operator >>(Tensor &arr) {
	std::pair<vector<int>, vector<double>> tuple;
	*this >> tuple;
	auto &shape = tuple.first;
	auto &weight = tuple.second;
	assert(shape.size() == 3);

	int dimension0 = shape[0];
	int dimension1 = shape[1];
	int dimension2 = shape[2];

	printf("d0 = %d, d1 = %d, d2 = %d\n", dimension0, dimension1, dimension2);
	arr.resize(dimension0);

	int index = 0;
	for (int i0 = 0; i0 < dimension0; ++i0) {
		arr[i0].resize(dimension1, dimension2);
		for (int i1 = 0; i1 < dimension1; ++i1) {
			for (int i2 = 0; i2 < dimension2; ++i2) {
				arr[i0](i1, i2) = weight[index++];
			}
		}
	}

	return *this;
}

void print_primitive_type_size() {
	cout << "sizeof(char) = " << sizeof(char) << endl;
	cout << "sizeof(wchar_t) = " << sizeof(wchar_t) << endl;
	cout << "sizeof(short) = " << sizeof(short) << endl;
	cout << "sizeof(int) = " << sizeof(int) << endl;
	cout << "sizeof(long) = " << sizeof(long) << endl;
	cout << "sizeof(long long) = " << sizeof(long long) << endl;

	cout << "sizeof(unsigned char) = " << sizeof(unsigned char) << endl;
	cout << "sizeof(unsigned wchar_t) = " << sizeof(unsigned wchar_t) << endl;
	cout << "sizeof(unsigned short) = " << sizeof(unsigned short) << endl;
	cout << "sizeof(unsigned int) = " << sizeof(unsigned int) << endl;
	cout << "sizeof(unsigned long) = " << sizeof(unsigned long) << endl;
	cout << "sizeof(unsigned long long) = " << sizeof(unsigned long long)
			<< endl;

	cout << "sizeof(float) = " << sizeof(float) << endl;
	cout << "sizeof(double) = " << sizeof(double) << endl;
	cout << "sizeof(byte) = " << sizeof(byte) << endl;
	cout << "sizeof(word) = " << sizeof(word) << endl;
	cout << "sizeof(dword) = " << sizeof(dword) << endl;
	cout << "sizeof(qword) = " << sizeof(qword) << endl;
	cout << "sizeof(void*) = " << sizeof(void*) << endl;
}

vector<double> convert2vector(const Matrix &m, int row_index) {
	auto start = m.data() + row_index * m.cols();

	vector<double> v(start, start + m.cols());
	return v;
}

vector<vector<double>> convert2vector(const Matrix &m) {
	vector<vector<double>> v(m.rows(), vector<double>(m.cols(), 0.0));
	for (int i = 0; i < m.rows(); ++i) {
		for (int j = 0; j < m.cols(); ++j) {
			v[i][j] = m(i, j);
		}
	}
	return v;
}

vector<double> convert2vector(const Vector &m) {
	auto start = m.data();

	vector<double> v(start, start + m.cols());
	return v;
}

//void HDF5Reader::close() {
//	int size = dis.tellg();
//	dis.seekg(0, std::ios::end);
//	int _size = dis.tellg();
//	assert(_size == size);;
//
//	dis.close();
//}

size_t Text::get_utf8_char_len(char byte) {
// return 0 表示错误
// return 1-6 表示正确值
// 不会 return 其他值

//UTF8 编码格式：
//                              0        1        2        3        4        5
//     U-00000000 - U-0000007F: 0xxxxxxx
//     U-00000080 - U-000007FF: 110xxxxx 10xxxxxx
//     U-00000800 - U-0000FFFF: 1110xxxx 10xxxxxx 10xxxxxx
//     U-00010000 - U-001FFFFF: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
//     U-00200000 - U-03FFFFFF: 111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
//     U-04000000 - U-7FFFFFFF: 1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx

	size_t len = 0;
	unsigned char mask = 0x80;
	while (byte & mask) {
		len++;
		if (len > 6) {
			//cerr << "The mask get len is over 6." << endl;
			return 0;
		}
		mask >>= 1;
	}
	if (0 == len) {
		return 1;
	}
	return len;
}

char Text::get_bits(char ch, int start, int size, int shift) {
	return ((ch >> start) & ((1 << size) - 1)) << shift;
}

string& Text::unicode2utf(const String &wstr) {
	static string s;
	s.clear();
	for (word wc : wstr) {
		s += unicode2utf(wc);
	}
	return s;
}

void Text::test_utf_unicode_conversion() {
	for (word wc = 1; wc > 0; ++wc) {
		auto cstr = unicode2utf(wc);
		auto _wc = utf2unicode(cstr);
		assert(_wc == wc);
		if (_wc != wc) {
			cout << wc << " != " << _wc << endl;
		} else {
//			cout << wc << " == " << _wc << endl;
		}
	}
}

word Text::utf2unicode(const char *pText) {
//	https://blog.csdn.net/qq_38279908/article/details/89329740
//	https://www.cnblogs.com/cfas/p/7931787.html
//  #include <codecvt>        // std::codecvt_utf8
//	return std::wstring_convert<std::codecvt_utf8<wchar_t> >().from_bytes(str);
	word wc;
	char *uchar = (char*) &wc;

	if (!pText[1]) {
		uchar[1] = 0;
		uchar[0] = pText[0];
	} else if (!pText[2]) {
//		U-000007FF: 110xxxxx 10xxxxxx
		uchar[1] = get_bits(pText[0], 2, 4);
		uchar[0] = get_bits(pText[0], 0, 2, 6) + get_bits(pText[1], 0, 6);
	} else /*if (!pText[3])*/{
//		U-0000FFFF: 1110xxxx 10xxxxxx 10xxxxxx
		uchar[1] = get_bits(pText[0], 0, 4, 4) + get_bits(pText[1], 2, 4);
		uchar[0] = get_bits(pText[1], 0, 2, 6) + get_bits(pText[2], 0, 6);
	} /*else if (!pText[4]) {
	 //		U-001FFFFF: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
	 uchar[2] = get_bits(pText[0], 0, 3, 3) + get_bits(pText[1], 4, 2);
	 uchar[1] = get_bits(pText[1], 0, 4, 4) + get_bits(pText[2], 2, 4);
	 uchar[0] = get_bits(pText[2], 0, 2, 6) + get_bits(pText[3], 0, 6);
	 } else if (!pText[5]) {
	 //		U-03FFFFFF: 111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
	 uchar[3] = get_bits(pText[0], 0, 2);
	 uchar[2] = get_bits(pText[1], 0, 6, 2) + get_bits(pText[2], 4, 2);
	 uchar[1] = get_bits(pText[2], 0, 4, 4) + get_bits(pText[3], 2, 4);
	 uchar[0] = get_bits(pText[3], 0, 2, 6) + get_bits(pText[4], 0, 6);
	 } else if (!pText[6]) {
	 //		U-7FFFFFFF: 1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
	 uchar[3] = get_bits(pText[0], 0, 1, 6) + get_bits(pText[1], 0, 6);
	 uchar[2] = get_bits(pText[2], 0, 6, 2) + get_bits(pText[3], 4, 2);
	 uchar[1] = get_bits(pText[3], 0, 4, 4) + get_bits(pText[4], 2, 4);
	 uchar[0] = get_bits(pText[4], 0, 2, 6) + get_bits(pText[5], 0, 6);
	 }
	 */
	return wc;
}

const char* Text::unicode2utf(word wc, char *pText) {
//	https://blog.csdn.net/qq_38279908/article/details/89329740
//	https://www.cnblogs.com/cfas/p/7931787.html
//	return std::wstring_convert<std::codecvt_utf8<wchar_t> >().to_bytes(wstr);
	if (!pText) {
		pText = str;
	}

	char *uchar = (char*) &wc;

	if (!(wc & 0xFFFFFF00)) {
//     U-00000000 - U-0000007F: 0xxxxxxx
		pText[0] = uchar[0];
		pText[1] = 0;
	} else if (!(wc & 0xFFFFF000)) {
//		U-000007FF: 110xxxxx 10xxxxxx
		pText[0] = 0xC0 + get_bits(uchar[1], 0, 4, 2)
				+ get_bits(uchar[0], 6, 2);
		pText[1] = 0x80 + get_bits(uchar[0], 0, 6);
		pText[2] = 0;
	} else if (!(wc & 0xFFFF0000)) {
//		U-0000FFFF: 1110xxxx 10xxxxxx 10xxxxxx
		pText[0] = 0xE0 + get_bits(uchar[1], 4, 4);
		pText[1] = 0x80 + get_bits(uchar[1], 0, 4, 2)
				+ get_bits(uchar[0], 6, 2);
		pText[2] = 0x80 + get_bits(uchar[0], 0, 6);
		pText[3] = 0;
	} else if (!(wc & 0xFF000000)) {
//		U-001FFFFF: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
		pText[0] = 0xF0 + get_bits(uchar[2], 5, 3);
		pText[1] = 0x80 + get_bits(uchar[2], 0, 2, 4)
				+ get_bits(uchar[1], 4, 4);
		pText[2] = 0x80 + get_bits(uchar[1], 0, 4, 2)
				+ get_bits(uchar[0], 6, 2);
		pText[3] = 0x80 + get_bits(uchar[0], 0, 6);
		pText[4] = 0;
	} else if (!(wc & 0xF0000000)) {
//		U-03FFFFFF: 111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
		pText[0] = 0xF8 + get_bits(uchar[3], 0, 2);
		pText[1] = 0x80 + get_bits(uchar[2], 2, 6);
		pText[2] = 0x80 + get_bits(uchar[2], 0, 2, 4)
				+ get_bits(uchar[1], 4, 4);
		pText[3] = 0x80 + get_bits(uchar[1], 0, 4, 2)
				+ get_bits(uchar[0], 6, 2);
		pText[4] = 0x80 + get_bits(uchar[0], 0, 6);
		pText[5] = 0;
	} else {
//		U-7FFFFFFF: 1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
		pText[0] = 0xFC + get_bits(uchar[3], 7, 1);
		pText[1] = 0x80 + get_bits(uchar[3], 0, 6);
		pText[2] = 0x80 + get_bits(uchar[2], 2, 6);
		pText[3] = 0x80 + get_bits(uchar[2], 0, 2, 4)
				+ get_bits(uchar[1], 4, 4);
		pText[4] = 0x80 + get_bits(uchar[1], 0, 4, 2)
				+ get_bits(uchar[0], 6, 2);
		pText[5] = 0x80 + get_bits(uchar[0], 0, 6);
		pText[6] = 0;
	}

	return pText;
}

Text::operator bool() {
	if (file)
		return true;
	return false;
//	return file.operator bool();
}

Text::Text(const string &file) :
		file(file.c_str()) {
}

char Text::str[7];

Text& Text::operator >>(word &v) {
	if (file.get(str[0])) {
		int length = get_utf8_char_len(str[0]);
		file.read(str + 1, length - 1);
		str[length] = 0;

		v = utf2unicode(str);
	}
	return *this;
}

//String L(const char *s) {
//	String wstr;
//	static char str[7];
//	while ((str[0] = *s++)) {
//		int length = Text::get_utf8_char_len(str[0]);
//		for (int i = 1; i < length; ++i)
//			str[i] = *s++;
//
//		str[length] = 0;
//
//		wstr += Text::utf2unicode(str);
//
//	}
//	return wstr;
//}

Text& Text::operator >>(String &v) {
	word wc;
	v.clear();
	while (*this >> wc) {
		if (wc == '\r' || wc == '\n') {
			if (v.size())
				break;
			else
				continue;
		}

		v += wc;
	}
	return *this;
}

Text& Text::operator >>(unordered_map<String, int> &word2id) {
	word2id.clear();
	String s;
	int index = 0;
	while (*this >> s) {
		strip(s);
		word2id[s] = index;
//		if (index <= 200) {
//			cout << s << " = " << index << endl;
//			cout << "s.size() = " << s.size() << endl;
//		}
		++index;
	}
	return *this;
}

//Text& Text::operator >>(string &s) {
//	if (file >> str[0]) {
//		int length = get_utf8_char_len(str[0]);
//		int i;
//		for (i = 1; i < length; ++i) {
//			file >> str[i];
//		}
//		str[i] = 0;
//		s = str;
//	}
//	return *this;
//}
//

VectorI& string2id(const String &s, const unordered_map<String, int> &dict) {
	static VectorI v;
	v.resize(s.size());

	for (size_t i = 0; i < s.size(); ++i) {
		v(i) = dict.at(s.substr(i, 1));
	}
	return v;
}

vector<VectorI>& string2id(const vector<String> &s,
		const unordered_map<String, int> &dict) {
	static vector<VectorI> v;
	int batch_size = s.size();
	v.reserve(batch_size);

	for (int k = 0; k < batch_size; ++k) {
		v[k] = string2id(s[k], dict);
	}
	return v;
}

#ifdef _DEBUG
#include <windows.h>
string& Text::unicode2gbk(const String &wstr) {
	static string result;
	static_assert(sizeof (wchar_t) == sizeof (char16_t));
	int n = WideCharToMultiByte(CP_ACP, 0, (const wchar_t*) wstr.c_str(), -1, 0,
			0, 0, 0);
	result.resize(n - 1);
	::WideCharToMultiByte(CP_ACP, 0, (const wchar_t*) wstr.c_str(), -1,
			(char*) result.c_str(), n, 0, 0);
	return result;
}
#endif

std::ostream& operator <<(std::ostream &cout, const String &unicode) {
#ifdef _DEBUG
	cout << Text::unicode2gbk(unicode);
#else
	cout << Text::unicode2utf(unicode);
#endif

	return cout;
}

vector<String>& split(const String &in) {
	static vector<String> array;
	array.clear();
	String s;
	for (auto ch : in) {
		if (iswspace(ch)) {
			if (!s.empty()) {
				array << s;
				s.clear();
			}
		} else {
			s += ch;
		}
	}

	if (!s.empty()) {
		array << s;
	}
	return array;
}

//#include <locale>         // std::wstring_convert
