#include "utility.h"
#include "keras.h"

struct AugmentedLstm {
	AugmentedLstm(TorchReader &dis);
	Matrix input_linearity;
	DenseLayer state_linearity;
	int hidden_size;

	Matrix& operator ()(const Matrix &x, Matrix &y);
	Matrix& operator ()(const Matrix &x, Matrix &y, bool go_forward);

	Vector& activate(const Eigen::Block<const Matrix, 1, -1, 1> &x, Vector &h,
			Vector &c) const;
};

struct StackedBidirectionalLstm {
	StackedBidirectionalLstm(TorchReader &dis);
	AugmentedLstm forward_layer_0, backward_layer_0, forward_layer_1,
			backward_layer_1, forward_layer_2, backward_layer_2;
	Matrix& operator ()(Matrix &x);
};

struct BilinearMatrixAttention {
	BilinearMatrixAttention(TorchReader &dis);
	Matrix _weight_matrix;
	double _bias;
	Matrix operator ()(const Matrix &x, const Matrix &y);
};

struct BiaffineDependencyParser {
	BiaffineDependencyParser(TorchReader &dis);

	VectorI& predict(const VectorI &seg, const VectorI &pos, VectorI &dep,
			VectorI &heads);

	Vector _head_sentinel;
	Embedding text_field_embedder;

	StackedBidirectionalLstm encoder;

	DenseLayer head_arc_feedforward, child_arc_feedforward;

	BilinearMatrixAttention arc_attention;

	DenseLayer head_tag_feedforward, child_tag_feedforward;

	Bilinear tag_bilinear;

	Embedding _pos_tag_embedding;

	VectorI _mst_decode(const Matrix &head_tag_representation,
			const Matrix &child_tag_representation, Matrix &attended_arcs,
			VectorI &predicted_head_tags);

	Tensor energy(const Matrix &head_tag_representation,
			const Matrix &child_tag_representation, Matrix &attended_arcs);

	VectorI& _structure_decode(const Matrix &head_tag_representation,
			const Matrix &child_tag_representation, Matrix &attended_arcs,
			VectorI &predicted_head_tags, VectorI &heads);

	VectorI _run_mst_decoding(const Tensor &batch_energy,
			VectorI &predicted_head_tags);
};

struct SyntaxParser {
	string modelFolder;
	dict<String, int> vocab;
	dict<String, int> posTags;
	vector<String> depTags;

	BiaffineDependencyParser model;

	vector<String> convertToDEPtags(const VectorI &ids);

	VectorI predict(const vector<String> &seg, const vector<String> &pos,
			vector<String> &dep);

	VectorI& predict(const vector<String> &seg, const vector<String> &pos,
			vector<String> &dep, VectorI &heads);

	SyntaxParser(const string &modelFolder);

	static SyntaxParser& instance();
	static SyntaxParser& instantiate();
};
