#include "utility.h"
#include <string>
#include<cstring> // to include strlen

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
			__log(__PRETTY_FUNCTION__);
			cerr << "illegal char encountered" << (int) byte << endl;
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

char Text::get_bits(char ch, int start, int size, char _ch) {
	return get_bits(ch, start, size, _ch, 8 - size);
}

char Text::get_bits(char ch, int start, int size, char _ch, int _size) {
	return get_bits(_ch, 0, _size, size) + get_bits(ch, start, size);
}

string Text::unicode2utf(const String &wstr) {
	string s;
//	s.clear();
	char str[7];
	for (word wc : wstr) {
		s += unicode2utf(wc, str);
	}
	return s;
}

void Text::test_utf_unicode_conversion() {
	char str[7];
	for (word wc = 1; wc > 0; ++wc) {
		auto cstr = unicode2utf(wc, str);
		auto _wc = utf2unicode(cstr, strlen(cstr));
		assert(_wc == wc);
		if (_wc != wc) {
			cout << wc << " != " << _wc << endl;
		} else {
//			cout << wc << " == " << _wc << endl;
		}
	}
}

int Text::unicode2jchar(int unicode) {
	int res_jchars = 0xdc00d800;
	auto jchars = (word*) &res_jchars;
//	jchars[0] = 0xd800;
//	jchars[1] = 0xdc00;

	unicode -= 65536;

	jchars[0] |= unicode >> 10;
	jchars[1] |= unicode & 0x03ff;

	return res_jchars;
}

int Text::utf2unicode(const char *pText, int length) {
//	https://blog.csdn.net/qq_38279908/article/details/89329740
//	https://www.cnblogs.com/cfas/p/7931787.html
//  #include <codecvt>        // std::codecvt_utf8
//	return std::wstring_convert<std::codecvt_utf8<wchar_t> >().from_bytes(str);
	int wc = 0;
	char *uchar = (char*) &wc;

//	if (length == 0) {
//		length = strlen(pText);
//	}

	switch (length) {
	case 1:
		uchar[1] = 0;
		uchar[0] = pText[0];
		break;
	case 2:
		//		U-000007FF: 110xxxxx 10xxxxxx
		uchar[1] = get_bits(pText[0], 2, 4);
		uchar[0] = get_bits(pText[1], 0, 6, pText[0]);
		break;
	case 3:
		//		U-0000FFFF: 1110xxxx 10xxxxxx 10xxxxxx
		uchar[1] = get_bits(pText[1], 2, 4, pText[0]);
		uchar[0] = get_bits(pText[2], 0, 6, pText[1]);
		break;
	case 4:
		//		U-001FFFFF: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
		uchar[2] = get_bits(pText[1], 4, 2, pText[0], 4);
		uchar[1] = get_bits(pText[2], 2, 4, pText[1]);
		uchar[0] = get_bits(pText[3], 0, 6, pText[2]);
		break;
	case 5:
		//		U-03FFFFFF: 111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
		uchar[3] = get_bits(pText[0], 0, 3);
		uchar[2] = get_bits(pText[2], 4, 2, pText[1]);
		uchar[1] = get_bits(pText[3], 2, 4, pText[2]);
		uchar[0] = get_bits(pText[4], 0, 6, pText[3]);
		break;
	case 6:
		//		U-7FFFFFFF: 1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
		uchar[3] = get_bits(pText[1], 0, 6, pText[0]);
		uchar[2] = get_bits(pText[3], 4, 2, pText[2]);
		uchar[1] = get_bits(pText[4], 2, 4, pText[3]);
		uchar[0] = get_bits(pText[5], 0, 6, pText[4]);
		break;
	default:
		;
	}
	return wc;
}

const char* Text::unicode2utf(word wc, char *pText) {
//	https://blog.csdn.net/qq_38279908/article/details/89329740
//	https://www.cnblogs.com/cfas/p/7931787.html
//	return std::wstring_convert<std::codecvt_utf8<wchar_t> >().to_bytes(wstr);
//	static char str[7];
//	if (!pText) {
//		pText = str;
//	}

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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
	int wc;
	if (*this >> wc) {
		if (wc != 0xfeff) {
			this->file.seekg(0, std::ios::beg);
		}
#pragma GCC diagnostic pop
	}
}

Text::iterator::iterator(Text *text, bool eof) :
		text(text), eof(eof) {
}

bool Text::iterator::operator !=(iterator &end) {
	return this->eof != end.eof;
}

Text::iterator& Text::iterator::operator ++() {
	if (*text >> text->line) {
		eof = false;
	} else {
		eof = text->line.empty();
	}

	return *this;
}

String& Text::iterator::operator *() {
	return text->line;
}

Text::iterator Text::begin() {
	bool eof;
	if (*this >> line) {
		eof = false;
	} else {
		eof = true;
	}

	return iterator(this, eof);
}

Text::iterator Text::end() {
	return iterator(this, true);
}

//char Text::str[7];

Text& Text::operator >>(int &unicode) {
	char str[6];
	if (file.get(str[0])) {
		int length = get_utf8_char_len(str[0]);
		file.read(str + 1, length - 1);
		unicode = utf2unicode(str, length);
	}
	return *this;
}

String Text::toString() {
	vector<String> v;
	*this >> v;
	String tmp;
	for (auto &s : v)
		tmp += s;
	return tmp;
}

vector<String> Text::readlines() {
	vector<String> v;
	*this >> v;
	return v;
}
Text& Text::operator >>(vector<String> &v) {
	String line;
	if (v.size()) {
		for (size_t i = 0; i < v.size(); ++i) {
			if (*this >> line)
				v[i] = line;
			else {
				if (!line.empty())
					v[i] = line;
				break;
			}
		}
	} else {
		while (*this >> line) {
			v.push_back(line);
		}

		if (!line.empty())
			v.push_back(line);

	}
	return *this;
}

String& Text::append_unicode(String &v, int wc) {
	if (wc & 0xffff0000) {
		wc = unicode2jchar(wc);
		auto jchars = (word*) &wc;
		v += jchars[0];
		v += jchars[1];
	} else
		v += wc;
	return v;
}

Text& Text::operator >>(String &v) {
	v.clear();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
	int wc;

	while (*this >> wc) {
		if (wc == '\r' || wc == '\n') {
			if (v.size())
				break;
			else
				continue;
		}
		append_unicode(v, wc);
	}
#pragma GCC diagnostic pop
	return *this;
}

dict<String, int> Text::read_vocab(int index) {
	dict<String, int> word2id;
	return read_vocab(word2id, index);
}

dict<string, int> Text::read_vocab_cstr() {
	dict<string, int> word2id;
//	__log(__PRETTY_FUNCTION__)
	string s;
	int index = 0;
	while (std::getline(file, s)) {
		strip(s);
		s = s.substr(0, s.find_first_of('\t'));
		assert(!s.empty());

		assert(word2id.count(s) == 0);

		word2id[s] = index++;
	}
	cout << "word2id.size() = " << word2id.size() << endl;
	cout << "index = " << index << endl;
	return word2id;
}

dict<String, int>& Text::read_vocab(dict<String, int> &word2id, int index) {
	word2id.clear();
	String s;
	for (String &s : *this) {
		strip(s);
		assert(!s.empty());
//		cout << s << " = " << index << endl;
//		cout << "s.size() = " << s.size() << endl;
		assert(word2id.count(s) == 0);

		word2id[s] = index++;
	}
	cout << "word2id.size() = " << word2id.size() << endl;
	cout << "index = " << index << endl;
	return word2id;
}

dict<string, int>& Text::read_vocab(dict<string, int> &word2id, int index) {
	__log(__PRETTY_FUNCTION__)
	word2id.clear();
	string s;
	while (std::getline(file, s)) {
		strip(s);
		assert(!s.empty());

		assert(word2id.count(s) == 0);

		word2id[s] = index++;
	}
	cout << "word2id.size() = " << word2id.size() << endl;
	cout << "index = " << index << endl;
	return word2id;
}

dict<char16_t, int> Text::read_vocab_char() {
	dict<char16_t, int> word2id;
	*this >> word2id;
	return word2id;
}

Text& Text::operator >>(dict<String, int> &word2id) {
	this->read_vocab(word2id, 2);
	return *this;
}

Text& Text::operator >>(dict<char16_t, int> &word2id) {
	word2id.clear();
	String s;
	size_t index = 2;
	for (String &s : *this) {
		strip(s);
		assert(s.size() == 1);
		char16_t ch = s[0];
//		cout << s << " = " << index << endl;
//		cout << "s.size() = " << s.size() << endl;
		assert(word2id.count(ch) == 0);

		word2id[ch] = index++;
	}
	cout << "word2id.size() = " << word2id.size() << endl;
	cout << "index = " << index << endl;
	assert(word2id.size() == index - 2);
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

#include<fstream>
using namespace std;

#if defined(_DEBUG) or defined(__WIN64__)
#include <windows.h>
string Text::unicode2gbk(const String &wstr) {
	string result;
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
#if defined(_DEBUG) //or defined(__WIN64__)
	cout << Text::unicode2gbk(unicode);
#else
	cout << Text::unicode2utf(unicode);
#endif

	return cout;
}

vector<String> split(const String &in) {
	vector<String> array;
	String s;
	for (auto ch : in) {
		if (iswspace(ch)) {
			if (!s.empty()) {
				array.push_back(s);
				s.clear();
			}
		} else {
			s += ch;
		}
	}

	if (!s.empty()) {
		array.push_back(s);
	}
	return array;
}

vector<string> split(const string &in) {
	vector<string> array;
	string s;
	for (auto ch : in) {
		if (isspace(ch)) {
			if (!s.empty()) {
				array.push_back(s);
				s.clear();
			}
		} else {
			s += ch;
		}
	}

	if (!s.empty()) {
		array.push_back(s);
	}
	return array;
}

vector<string> split(const string &in, char separator) {
	vector<string> array;
	string s;
	for (auto ch : in) {
		if (ch == separator) {
			if (!s.empty()) {
				array.push_back(s);
				s.clear();
			}
		} else {
			s += ch;
		}
	}

	if (!s.empty()) {
		array.push_back(s);
	}
	return array;
}

namespace std {
int strlen(const String &value) {
	int length = 0;
	for (auto ch : value) {
		if ((ch & 0xff80) != 0)
			length += 2;
		else
			++length;
	}
	return length;
}

int strlen(const string &value) {
	return value.size();
}

#include <sstream>
String toString(int d) {
	return toString(std::to_string(d));
}

String toString(const string &c_str) {
	String s;
	for (size_t j = 0, size = c_str.size(); j < size;) {
		auto &ch = c_str[j];
		auto length = Text::get_utf8_char_len(ch);
		if (length == 0)
			return u"??";

		j += length;

		ensure_le(j, size);

		Text::append_unicode(s, Text::utf2unicode(&ch, length));
	}

	return s;
}

//possible bugs
string to_string(const String &wstr) {
	string s;
	char str[7];
	for (word wc : wstr) {
		s += Text::unicode2utf(wc, str);
	}
	return s;
}

}

#pragma GCC diagnostic ignored "-Wparentheses"
//	[a-zA-Zａ-ｚＡ-Ｚα-ϋΑ-Ϋа-яА-Яⅰ-ⅿⅠ-Ⅿ]
//	[a-zａ-ｚα-ϋа-яⅰ-ⅿ]
//	[A-ZＡ-ＺΑ-ΫА-ЯⅠ-Ⅿ]
char16_t tolower(char16_t ch) {
	switch (ch) {
//English big letters:
	case u'A':
	case u'B':
	case u'C':
	case u'D':
	case u'E':
	case u'F':
	case u'G':
	case u'H':
	case u'I':
	case u'J':
	case u'K':
	case u'L':
	case u'M':
	case u'N':
	case u'O':
	case u'P':
	case u'Q':
	case u'R':
	case u'S':
	case u'T':
	case u'U':
	case u'V':
	case u'W':
	case u'X':
	case u'Y':
	case u'Z':
//English wide big letters:
	case u'Ａ':
	case u'Ｂ':
	case u'Ｃ':
	case u'Ｄ':
	case u'Ｅ':
	case u'Ｆ':
	case u'Ｇ':
	case u'Ｈ':
	case u'Ｉ':
	case u'Ｊ':
	case u'Ｋ':
	case u'Ｌ':
	case u'Ｍ':
	case u'Ｎ':
	case u'Ｏ':
	case u'Ｐ':
	case u'Ｑ':
	case u'Ｒ':
	case u'Ｓ':
	case u'Ｔ':
	case u'Ｕ':
	case u'Ｖ':
	case u'Ｗ':
	case u'Ｘ':
	case u'Ｙ':
	case u'Ｚ':
//Greek big letters:
	case u'Α':
	case u'Β':
	case u'Γ':
	case u'Δ':
	case u'Ε':
	case u'Ζ':
	case u'Η':
	case u'Θ':
	case u'Ι':
	case u'Κ':
	case u'Λ':
	case u'Μ':
	case u'Ν':
	case u'Ξ':
	case u'Ο':
	case u'Π':
	case u'Ρ':
	case u'Σ':
	case u'Τ':
	case u'Υ':
	case u'Φ':
	case u'Χ':
	case u'Ψ':
	case u'Ω':
	case u'Ϊ':
	case u'Ϋ':
//Russian big letters:
	case u'А':
	case u'Б':
	case u'В':
	case u'Г':
	case u'Д':
	case u'Е':
	case u'Ж':
	case u'З':
	case u'И':
	case u'Й':
	case u'К':
	case u'Л':
	case u'М':
	case u'Н':
	case u'О':
	case u'П':
	case u'Р':
	case u'С':
	case u'Т':
	case u'У':
	case u'Ф':
	case u'Х':
	case u'Ц':
	case u'Ч':
	case u'Ш':
	case u'Щ':
	case u'Ъ':
	case u'Ы':
	case u'Ь':
	case u'Э':
	case u'Ю':
	case u'Я':
		return ch + 32;
//Roman big digits
	case u'Ⅰ':
	case u'Ⅱ':
	case u'Ⅲ':
	case u'Ⅳ':
	case u'Ⅴ':
	case u'Ⅵ':
	case u'Ⅶ':
	case u'Ⅷ':
	case u'Ⅸ':
	case u'Ⅹ':
	case u'Ⅺ':
	case u'Ⅻ':
	case u'Ⅼ':
	case u'Ⅽ':
	case u'Ⅾ':
	case u'Ⅿ':
		return ch + 16;
	default:
		return ch;
	}
}
//∑∫∪∩√∈∏
char16_t toupper(char16_t ch) {
	switch (ch) {
//English small letters:
	case u'a':
	case u'b':
	case u'c':
	case u'd':
	case u'e':
	case u'f':
	case u'g':
	case u'h':
	case u'i':
	case u'j':
	case u'k':
	case u'l':
	case u'm':
	case u'n':
	case u'o':
	case u'p':
	case u'q':
	case u'r':
	case u's':
	case u't':
	case u'u':
	case u'v':
	case u'w':
	case u'x':
	case u'y':
	case u'z':
//English wide small letters:
	case u'ａ':
	case u'ｂ':
	case u'ｃ':
	case u'ｄ':
	case u'ｅ':
	case u'ｆ':
	case u'ｇ':
	case u'ｈ':
	case u'ｉ':
	case u'ｊ':
	case u'ｋ':
	case u'ｌ':
	case u'ｍ':
	case u'ｎ':
	case u'ｏ':
	case u'ｐ':
	case u'ｑ':
	case u'ｒ':
	case u'ｓ':
	case u'ｔ':
	case u'ｕ':
	case u'ｖ':
	case u'ｗ':
	case u'ｘ':
	case u'ｙ':
	case u'ｚ':
//Russian small letters:
	case u'а':
	case u'б':
	case u'в':
	case u'г':
	case u'д':
	case u'е':
	case u'ж':
	case u'з':
	case u'и':
	case u'й':
	case u'к':
	case u'л':
	case u'м':
	case u'н':
	case u'о':
	case u'п':
	case u'р':
	case u'с':
	case u'т':
	case u'у':
	case u'ф':
	case u'х':
	case u'ц':
	case u'ч':
	case u'ш':
	case u'щ':
	case u'ъ':
	case u'ы':
	case u'ь':
	case u'э':
	case u'ю':
	case u'я':
//Greek small letters:
	case u'α':
	case u'β':
	case u'γ':
	case u'δ':
	case u'ε':
	case u'ζ':
	case u'η':
	case u'θ':
	case u'ι':
	case u'κ':
	case u'λ':
	case u'μ':
	case u'ν':
	case u'ξ':
	case u'ο':
	case u'π':
	case u'ρ':
	case u'σ':
	case u'τ':
	case u'υ':
	case u'φ':
	case u'χ':
	case u'ψ':
	case u'ω':
	case u'ϊ':
	case u'ϋ':
		return ch - 32;
	case u'ς':
		return u'Σ'; // the Greek sigmar letter has two forms of small cases!
//Roman small digits
	case u'ⅰ':
	case u'ⅱ':
	case u'ⅲ':
	case u'ⅳ':
	case u'ⅴ':
	case u'ⅵ':
	case u'ⅶ':
	case u'ⅷ':
	case u'ⅸ':
	case u'ⅹ':
	case u'ⅺ':
	case u'ⅻ':
	case u'ⅼ':
	case u'ⅽ':
	case u'ⅾ':
	case u'ⅿ':
		return ch - 16;
	default:
		return ch;
	}
}

double pi_test(int n) {
	double sum = 0.0;
//#pragma omp parallel for num_threads(cpu_count()) reduction(+:sum)

	for (int i = 0; i < n; i++) {
		double factor;
		if (i % 2 == 0)
			factor = 1.0;
		else
			factor = -1.0;
		sum += factor / (2 * i + 1);
	}

	return 4.0 * sum;
}

Timer::Timer() {
	start = clock();
}

void Timer::report(const char *message) {
	auto end = clock();
	cout << message << " cost " << (end - start) / CLOCKS_PER_SEC << " seconds"
			<< endl;
	start = end;
}

#include <random>
void test_priority_dict() {
	priority_dict<String> pq;
	pq.insert(u"this");
	pq.insert(u"that");
	pq.insert(u"abc");
	pq.insert(u"def");
	pq.insert(u"ghi");
	pq.insert(u"jkl");
	pq.insert(u"mno");
	pq.insert(u"pqr");
	pq.insert(u"stu");
	pq.insert(u"vwx");
	pq.insert(u"yz");
	std::default_random_engine e;
	while (!pq.empty()) {
		auto element = pq.map.begin()->first;
		cout << "removing " << element << endl;
		pq.erase(pq.map.begin()->first);
		pq.insert(std::toString(e()));
	}

}

String& tolower(String &s) {
	if (s.empty()) {
		return s;
	}
	for (auto &ch : s) {
		ch = tolower(ch);
	}
	return s;
}

void seed_rand() {
	srand(time(0));
}

int nextInt(int max) {
	return rand() % max;
}

void append_file_separator(string &workingDirectory) {
	switch (workingDirectory.back()) {
	case '/':
	case '\\':
		break;
	default:
		workingDirectory += '/';
	}
}
void test_set_operation() {
	std::set<string> A { "a", "b", "c", "d" };
	std::set<string> B { "a", "b", "c", "e" };

	cout << "A = " << A << endl;
	cout << "B = " << B << endl;
	cout << "A - B = " << (A - B) << endl;
	cout << "A | B = " << (A | B) << endl;
	cout << "A & B = " << (A & B) << endl;
}

#include <unistd.h>

string getcwd() {
	char pPath[256] = { 0 };
	return ::getcwd(pPath, sizeof(pPath));
}

string now() {
	time_t t = time(0);
	char buffer[9] = { 0 };
	strftime(buffer, sizeof(buffer), "%H:%M:%S", localtime(&t));
//	strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&t));
	return buffer;
}

BinaryFile::BinaryFile(const string &file, ByteOrder byteorder) :
		byteorder(byteorder) {
	if (os_access(file)) {
		this->file.open(file, std::ios::in | std::ios::out | std::ios::binary);
	} else {
		this->file.open(file, std::ios::out | std::ios::binary);
	}
}

int BinaryFile::getsize() {
	size_t current_pos = file.tellg();
	file.seekg(0, std::ios::end);
	size_t ending_pos = file.tellg();
	int remaining_size = ending_pos - current_pos;

	file.seekg(-remaining_size, std::ios::cur);
	return remaining_size;
}

void BinaryFile::swap_byte_order(char *buffer, int length) {
	for (int i = 0; i < length / 2; ++i) {
		std::swap(buffer[i], buffer[length - 1 - i]);
	}
}

#include <unistd.h>

bool os_access(const std::string &name, int status) {
	return (access(name.c_str(), status) != -1);
}

