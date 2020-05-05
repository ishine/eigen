#include <time.h>
#include <string>
#include <iostream>
using namespace std;

#include <stdio.h>

#include "../deeplearning/utility.h"
#include "../deeplearning/classification.h"
#include "../deeplearning/bert.h"
#include "../deeplearning/lagacy.h"
#include "../deeplearning/CWSTagger.h"
#include "../deeplearning/SyntaxParser.h"
#include "../ahocorasick/public.h"

int main(int argc, char **argv) {
	cout << "argc = " << argc << endl;
	for (int i = 0; i < argc; ++i) {
		cout << argv[i] << endl;
	}

	if (1 < argc) {
		workingDirectory = argv[1];
	}

//	auto &phatic = Classifier::phatic_classifier();
//	auto &qatype = Classifier::qatype_classifier();
	auto &keyword_cn = ClassifierChar::keyword_cn_classifier();
	auto &keyword_en = ClassifierWord::keyword_en_classifier();
	auto &paraphrase = Pairwise::paraphrase();
	auto &hyponym = PairwiseVector::hyponymCN();

	cout << "hyponym = " << hyponym(u"Gui", u"服务器") << endl;
	auto &cwsTagger = CWSTagger::instance();

	cout << "segments = " << cwsTagger.predict(u"(1) 圖示所揭露之虛線之部分，為本案不主張之部分。") << endl;

	cout << "keyword = " << keyword_cn.predict(u"如图所示") << endl;

	cout << "keyword = " << keyword_en.predict(u"Pairwise Algorithm") << endl;

	auto &syntaxParser = SyntaxParser::instance();

	{
		vector<String> seg = { u"我们", u"研究", u"所有", u"东西", u"。"};
	vector<String> pos = {u"PN", u"VT", u"JJ", u"NN", u"PU"};
	vector<String> dep;
	auto heads = syntaxParser.predict(seg, pos, dep);
	cout << "seg = " << seg << endl;
	cout << "pos = " << pos << endl;
	cout << "dep = " << dep << endl;
	cout << "heads = " << heads << endl;
}
	{
		vector<String> seg = { u"你", u"说", u",", u"这", u"比", u"山", u"还", u"高", u"比", u"海", u"还", u"深", u"的", u"情谊", u",", u"我们", u"怎么", u"能", u"忘怀", u"?", u"仿写", u"句子"};
	vector<String> pos = {u"PN", u"VT", u"PU", u"DT", u"P", u"NN", u"AD", u"VA", u"P", u"NN", u"AD", u"VA", u"DE", u"NN", u"PU", u"PN", u"AD", u"MD", u"VT", u"PU", u"VT", u"NN"};
	vector<String> dep;
	auto heads = syntaxParser.predict(seg, pos, dep);
	cout << "seg = " << seg << endl;
	cout << "pos = " << pos << endl;
	cout << "dep = " << dep << endl;
	cout << "heads = " << heads << endl;
}

	vector<String> keywords = { u"显示器", u"服务器", u"导航系统", u"计算机", u"查询预设", u"推送信息库", u"导航请求消息", u"位置信息", u"图形用户界面", u"便携式", u"显示屏", u"传感器", u"卫星导航系统", u"多功能装置", u"子单元区", u"Gps", u"信息推送", u"目标推送信息", u"接收车载设备发送", u"浏览器", u"导航路线", u"管理计算机显示器", u"显示区", u"导航路径", u"导航装置", u"Gui", u"电子文档", u"导航指令", u"目的地", u"车载终端", u"导航地图", u"厂家预设", u"便携式电子", u"预设阈值", u"采集相关技术", u"服务器发送", u"定位信息", u"地图上位置区域", u"处理器", u"记录地图", u"导航信息", u"推送系统", u"目标推送信息发送", u"交互式", u"服务器接收", u"导航模块", u"数据库", u"停止推送信息", u"规划路线", u"目标地址进行分类"};
	vector<int> frequency = { 59, 54, 48, 34, 29, 29, 29, 28, 22, 21, 20, 19,
			18, 18, 18, 17, 17, 17, 17, 16, 16, 14, 14, 13, 13, 13, 13, 13, 13,
			13, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 9,
			9, 8, 8, 8 };

	Timer timer;
	cout << hyponymStructureCN(keywords, frequency);

	timer.report("hyponymStructureCN(keywords)");

//	cout << "phatic = " << phatic.predict(u"请问您在哪个城市,请提供您的有效联系方式") << endl;

//	cout << "qatype = " << qatype.predict(u"how are you today?") << endl;

	cout << "paraphrase score = " << paraphrase(u"你们公司有些什么业务", u"你们公司业务有哪些") << endl;
	cout << "paraphrase score = " << paraphrase(u"周末你去哪里玩", u"今天他去哪里玩？") << endl;

	cout << "hyponym score = " << hyponym(u"业务", u"公司业务") << endl;
	cout << "hyponym score = " << hyponym(u"今晚", u"今天") << endl;

	cout << "zero = " << zero << endl;
	cout << "one = " << one << endl;
	cout << "one_fifth = " << one_fifth << endl;
	cout << "half = " << half << endl;

	cout << "gcd_long(10, 46) = " << gcd_long(10, 46) << endl;
	cout << "gcd_qword(10, 46) = " << gcd_qword(10, 46) << endl;
	cout << "gcd_int(10, 46) = " << gcd_int(10, 46) << endl;
	cout << "gcd_dword(10, 46) = " << gcd_dword(10, 46) << endl;

	cout << "relu(10.1) = " << relu(10.1) << endl;
	cout << "relu(0.0) = " << relu(0.0) << endl;
	cout << "relu(-10.1) = " << relu(-10.1) << endl;
	cout << "hard_sigmoid(-10.1) = " << hard_sigmoid(-10.1) << endl;
	cout << "hard_sigmoid(10.1) = " << hard_sigmoid(10.1) << endl;
	cout << "hard_sigmoid(2.5) = " << hard_sigmoid(2.5) << endl;
	cout << "hard_sigmoid(-2.5) = " << hard_sigmoid(-2.5) << endl;
	cout << "hard_sigmoid(0) = " << hard_sigmoid(0) << endl;

	cout << "cpu_count() = " << cpu_count << endl;

//	double pi_test(int n);

//	for (int n = 100; n < 10000; ++n) {
//		printf("pi_test(%d) = %f\n", n, pi_test(n));
//		assert(pi_test(n) > 3);
//	}
//	void test_eigen();
//	test_eigen();
	return 0;
}
//https://www.cnblogs.com/listenscience/p/11509164.html

//https://academy.zhihuiya.com/#/user/learning/index
//15821495341
//https://ks.wjx.top/jq/69262900.aspx

//segment
//该-项
//播放|音乐|观看|电视
//加-装

//non-keyword:
//响应于
//应于检测到
//项技术
//使用情况
//检测到
//根据用户
//接收到
//根据用户输入
//S102
//S103
//S101
//规划路线时
//公开了
//获取用户
//发送给车载设备
//位置处
//能够根据
//根据确定
//多个传感器
//次数小于预设阈值
//目标推送信息包含
//能够提高用户体验
//用户使用导航
//设备还在设备
//由小到大
//其他用户
//方法基于
//开始导航
//第二相机接收
//常用目
//信息发送给车载

//keyword:
//便携式电子设备
//车载终端
//初始指定位置
//推送信息库
//接收车载设备发送

//update tbl_paraphrase_cn set score = 43 where text regexp '数学高手' and paraphrase regexp '数学' limit 40
