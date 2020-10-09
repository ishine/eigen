#include "SyntaxParser.h"
#include "utility.h"
#include "../std/utility.h"

vector<String> SyntaxParser::convertToDEPtags(const VectorI &ids) {
	int n = ids.size();
	vector<String> dep(n);
//	__debug(ids);
//	__debug(depTags);
	for (int i = 0; i < n; ++i) {
		dep[i] = depTags[ids[i]];
	}
	return dep;
}

VectorI SyntaxParser::predict(const vector<String> &seg,
		const vector<String> &pos, vector<String> &dep) {
	__debug(__PRETTY_FUNCTION__)

	__debug(seg)
	__debug(pos)
	cout << "vocab.size() = " << vocab.size() << endl;
	cout << "posTags.size() = " << posTags.size() << endl;

	auto seg_ids = string2id(seg, this->vocab);
	auto pos_ids = string2id(pos, this->posTags);
	VectorI dep_ids;
	cout << "seg_ids = " << seg_ids << endl;
	cout << "pos_ids = " << pos_ids << endl;
	VectorI heads;
	heads = this->model.predict(seg_ids, pos_ids, dep_ids, heads);

	for (auto &id : heads) {
		--id;
	}

	dep = this->convertToDEPtags(dep_ids);
	return heads;
}

VectorI& SyntaxParser::predict(const vector<String> &seg,
		const vector<String> &pos, vector<String> &dep, VectorI &heads) {
	__debug(__PRETTY_FUNCTION__);

	__debug(seg);
	__debug(pos);
	__debug(heads);
	__debug(vocab.size());
	__debug(posTags.size());

	auto seg_ids = string2id(seg, this->vocab);
	auto pos_ids = string2id(pos, this->posTags);
	VectorI dep_ids;

	__debug(seg_ids);
	__debug(pos_ids);

	for (auto &id : heads) {
		++id;
	}

	__debug(heads);

	heads = this->model.predict(seg_ids, pos_ids, dep_ids, heads);

	__debug(heads);
	for (auto &id : heads) {
		--id;
	}
	__debug(heads);

	dep = this->convertToDEPtags(dep_ids);
	return heads;
}

VectorI& BiaffineDependencyParser::predict(const VectorI &seg,
		const VectorI &pos, VectorI &predicted_head_tags,
		VectorI &predicted_heads) {
	__debug(__PRETTY_FUNCTION__);

	int seq_len = seg.size();
	auto segEmbedding = text_field_embedder(seg);
	auto posEmbedding = _pos_tag_embedding(pos);

	Matrix embedded_text_input;
	embedded_text_input.resize(seq_len,
			segEmbedding.cols() + posEmbedding.cols());
	embedded_text_input << segEmbedding, posEmbedding;

	print_shape(embedded_text_input);
	__debug(embedded_text_input)

	auto &_encoded_text = encoder(embedded_text_input);
	Matrix encoded_text;
	encoded_text.resize(seq_len + 1, _encoded_text.cols());

//	print_shape(_head_sentinel);
	__debug(_head_sentinel)
	__debug(_encoded_text);

	encoded_text << _head_sentinel, _encoded_text;

//	print_shape(encoded_text);

	Matrix head_arc_representation, child_arc_representation,
			head_tag_representation, child_tag_representation;

//	__debug(encoded_text)
	head_arc_feedforward(encoded_text, head_arc_representation);
//	print_shape(head_arc_representation);

	child_arc_feedforward(encoded_text, child_arc_representation);
//	print_shape(child_arc_representation);
//	__debug(head_arc_representation)
//	__debug(child_arc_representation)
	auto attended_arcs = arc_attention(head_arc_representation,
			child_arc_representation);

	head_tag_feedforward(encoded_text, head_tag_representation);
	child_tag_feedforward(encoded_text, child_tag_representation);

	__debug(attended_arcs)
	if (predicted_heads.empty())
		predicted_heads = _mst_decode(head_tag_representation,
				child_tag_representation, attended_arcs, predicted_head_tags);

	else {
		predicted_heads.insert(predicted_heads.begin(), 0);
		predicted_heads = _structure_decode(head_tag_representation,
				child_tag_representation, attended_arcs, predicted_head_tags,
				predicted_heads);
	}

//	__debug(predicted_heads)
//	__debug(predicted_head_tags)
	predicted_heads.erase(predicted_heads.begin());
	predicted_head_tags.erase(predicted_head_tags.begin());

//	__debug(predicted_heads)
//	__debug(predicted_head_tags)

	return predicted_heads;
}

SyntaxParser::SyntaxParser(const string &modelFolder) :
		modelFolder(modelFolder),

		vocab(Text(modelFolder + "vocabulary/tokens.txt").read_vocab(1)),

		posTags(Text(modelFolder + "vocabulary/pos.txt").read_vocab(1)),

		depTags(Text(modelFolder + "vocabulary/head_tags.txt").readlines()),

		model(
				(TorchReader&) (const TorchReader&) TorchReader(
						modelFolder + "model.h5")) {
	__log(__PRETTY_FUNCTION__)
}

BiaffineDependencyParser::BiaffineDependencyParser(TorchReader &dis) :
		_head_sentinel(dis.read_tensor()[0].row(0)),

		text_field_embedder(dis),

		encoder(dis),

		head_arc_feedforward(dis, Activator::elu),

		child_arc_feedforward(dis, Activator::elu),

		arc_attention(dis),

		head_tag_feedforward(dis, Activator::elu),

		child_tag_feedforward(dis, Activator::elu),

		tag_bilinear(dis),

		_pos_tag_embedding(dis) {
	__debug(__PRETTY_FUNCTION__)
}

SyntaxParser& SyntaxParser::instance() {
	__debug(__PRETTY_FUNCTION__)
	static SyntaxParser instance(modelsDirectory() + "cn/dep/");

	return instance;
}

SyntaxParser& SyntaxParser::instantiate() {
	auto &instance = SyntaxParser::instance();

	instance = SyntaxParser(instance.modelFolder);

	return instance;
}

AugmentedLstm::AugmentedLstm(TorchReader &dis) :
		input_linearity(dis.read_matrix().transpose()), state_linearity(dis,
				Activator::linear),

		hidden_size(input_linearity.cols() / 6) {
	__debug(__PRETTY_FUNCTION__)
}

StackedBidirectionalLstm::StackedBidirectionalLstm(TorchReader &dis) :
		forward_layer_0(dis), backward_layer_0(dis),

		forward_layer_1(dis), backward_layer_1(dis),

		forward_layer_2(dis), backward_layer_2(dis) {
	__debug(__PRETTY_FUNCTION__)
}

Matrix& AugmentedLstm::operator ()(const Matrix &sequence_tensor,
		Matrix &output_accumulator) {
	__debug(__PRETTY_FUNCTION__)
	int total_timesteps = sequence_tensor.rows();

//	cout << "total_timesteps = " << total_timesteps << endl;

	output_accumulator.resize(total_timesteps, hidden_size);
	Vector previous_memory, previous_state;

	previous_memory = previous_state = Vector::Zero(hidden_size);

	print_shape(input_linearity);

	for (int index = 0; index < total_timesteps; ++index) {
		output_accumulator.row(index) = activate(sequence_tensor.row(index),
				previous_state, previous_memory);
	}

	return output_accumulator;
}

Vector& AugmentedLstm::activate(
		const Eigen::Block<const Matrix, 1, -1, 1> &timestep_input,
		Vector &previous_state, Vector &previous_memory) const {
//	cout << "hidden_size = " << hidden_size << endl;

//	print_shape(timestep_input);

	auto projected_input = timestep_input * input_linearity;
//	__debug(projected_input)
//	print_shape(projected_input);

//	print_shape(previous_state);

//	print_shape(state_linearity.weight);

	auto projected_state = state_linearity(previous_state);
//	__debug(projected_state)
	Vector input_gate = projected_input.leftCols(hidden_size)
			+ projected_state.leftCols(hidden_size);
	input_gate = sigmoid(input_gate);

//	__debug(input_gate)
	Vector forget_gate = projected_input.middleCols(hidden_size, hidden_size)
			+ projected_state.middleCols(hidden_size, hidden_size);
	forget_gate = sigmoid(forget_gate);
//	__debug(forget_gate)

	Vector memory_init = projected_input.middleCols(2 * hidden_size,
			hidden_size)
			+ projected_state.middleCols(2 * hidden_size, hidden_size);
	memory_init = tanh(memory_init);
//	__debug(memory_init)

	Vector output_gate = projected_input.middleCols(3 * hidden_size,
			hidden_size)
			+ projected_state.middleCols(3 * hidden_size, hidden_size);
	output_gate = sigmoid(output_gate);
//	__debug(output_gate)

	Vector memory = input_gate.cwiseProduct(memory_init)
			+ forget_gate.cwiseProduct(previous_memory);

	previous_memory = memory;
//error: memory is mutated after this operation! so save it before changing it!
	Vector timestep_output = output_gate.cwiseProduct(tanh(memory));
//	__debug(timestep_output)

	Vector highway_gate = projected_input.middleCols(4 * hidden_size,
			hidden_size)
			+ projected_state.middleCols(4 * hidden_size, hidden_size);
	highway_gate = sigmoid(highway_gate);

	Vector highway_input_projection = projected_input.rightCols(hidden_size);

	timestep_output = highway_gate.cwiseProduct(timestep_output)
			+ (Vector::Ones(highway_gate.cols()) - highway_gate).cwiseProduct(
					highway_input_projection);

	previous_state = timestep_output;

	return previous_state;

}

Matrix& AugmentedLstm::operator ()(const Matrix &sequence_tensor,
		Matrix &output_accumulator, bool go_forward) {
	if (go_forward)
		return (*this)(sequence_tensor, output_accumulator);

	__debug(__PRETTY_FUNCTION__)
	int total_timesteps = sequence_tensor.rows();
	output_accumulator.resize(total_timesteps, hidden_size);

	Vector previous_memory, previous_state;
	previous_memory = previous_state = Vector::Zero(hidden_size);

	for (int index = total_timesteps - 1; index >= 0; --index) {
		output_accumulator.row(index) = activate(sequence_tensor.row(index),
				previous_state, previous_memory);
	}

	return output_accumulator;
}

Matrix& StackedBidirectionalLstm::operator ()(Matrix &output_sequence) {
	__debug(__PRETTY_FUNCTION__)

	Matrix forward_output, backward_output;

	forward_layer_0(output_sequence, forward_output);
	backward_layer_0(output_sequence, backward_output, false);

//	__debug(forward_output)
//	__debug(backward_output)

	output_sequence.resize(output_sequence.rows(),
			forward_output.cols() + backward_output.cols());
	output_sequence << forward_output, backward_output;
//	print_shape(output_sequence);

	forward_layer_1(output_sequence, forward_output);
	backward_layer_1(output_sequence, backward_output, false);

	output_sequence.resize(output_sequence.rows(),
			forward_output.cols() + backward_output.cols());
	output_sequence << forward_output, backward_output;

//	print_shape(output_sequence);

	forward_layer_2(output_sequence, forward_output);
	backward_layer_2(output_sequence, backward_output, false);

	output_sequence.resize(output_sequence.rows(),
			forward_output.cols() + backward_output.cols());
	output_sequence << forward_output, backward_output;

//	print_shape(output_sequence);

	return output_sequence;
}

BilinearMatrixAttention::BilinearMatrixAttention(TorchReader &dis) :
		_weight_matrix(dis.read_matrix()), _bias(dis.read_double()) {
	__debug(__PRETTY_FUNCTION__)
}

Matrix BilinearMatrixAttention::operator ()(const Matrix &_matrix_1,
		const Matrix &_matrix_2) {
	__debug(__PRETTY_FUNCTION__)

	auto bias1 = Matrix::Ones(_matrix_1.rows(), 1);
	auto bias2 = Matrix::Ones(_matrix_2.rows(), 1);

	Matrix matrix_1, matrix_2;
	matrix_1.resize(_matrix_1.rows(), _matrix_1.cols() + 1);
	matrix_2.resize(_matrix_2.rows(), _matrix_2.cols() + 1);

//	print_shape(matrix_1);
//	print_shape(_matrix_1);
//	print_shape(bias1);

	matrix_1 << _matrix_1, bias1;

//	print_shape(matrix_2);
//	print_shape(_matrix_2);
//	print_shape(bias2);

	matrix_2 << _matrix_2, bias2;
	Matrix final = matrix_1 * _weight_matrix * matrix_2.transpose();
	return final += _bias;
}

Tensor BiaffineDependencyParser::energy(const Matrix &head_tag,
		const Matrix &child_tag, Matrix &attended_arcs) {
//	__debug(head_tag)
//	__debug(child_tag)
//	print_shape(head_tag);
//	print_shape(child_tag);
//	print_shape(attended_arcs);
	__debug(__PRETTY_FUNCTION__)
	int sequence_length = head_tag.rows();
	Tensor head_tag_representation(sequence_length);
	for (int i = 0; i < sequence_length; ++i) {
		head_tag_representation[i] = broadcast(head_tag.row(i),
				sequence_length);
	}

	Tensor child_tag_representation(sequence_length);
	for (int i = 0; i < sequence_length; ++i) {
		child_tag_representation[i] = child_tag;
	}

//	__debug(head_tag_representation)
//	__debug(child_tag_representation)
	auto pairwise_head_logits = tag_bilinear(head_tag_representation,
			child_tag_representation);

	print_tensor(pairwise_head_logits);

	pairwise_head_logits = transpose<2, 0, 1>(
			log_softmax(pairwise_head_logits));
//	pairwise_head_logits = transpose_201(log_softmax(pairwise_head_logits));

	print_tensor(pairwise_head_logits);

	print_shape(attended_arcs);
	log_softmax(attended_arcs).transposeInPlace();
//	__debug(predicted_head_tags)
//	__debug(pairwise_head_logits)
//	__debug(attended_arcs)

	return exp(pairwise_head_logits += attended_arcs);
}

VectorI BiaffineDependencyParser::_mst_decode(const Matrix &head_tag,
		const Matrix &child_tag, Matrix &attended_arcs,
		VectorI &predicted_head_tags) {
	return _run_mst_decoding(energy(head_tag, child_tag, attended_arcs),
			predicted_head_tags);
}

VectorI& BiaffineDependencyParser::_structure_decode(const Matrix &head_tag,
		const Matrix &child_tag, Matrix &attended_arcs,
		VectorI &instance_head_tags, VectorI &instance_heads) {
	__debug(__PRETTY_FUNCTION__);

	auto energy = this->energy(head_tag, child_tag, attended_arcs);

	MatrixI tag_ids = argmax(energy, 0);

	int seq_len = instance_heads.size();
	instance_head_tags.resize(seq_len);
//	# Find the labels which correspond to the edges in the max spanning tree.
	for (int child = 0; child < seq_len; ++child) {
		int parent = instance_heads[child];
		instance_head_tags[child] = tag_ids[parent][child];
	}

//	# We don't care what the head or tag is for the root token, but by default it's;
//	# not necesarily the same in the bat;ched vs unbatched case, which is annoying.;
//	# Here we'll just set them to zero.;
	instance_heads[0] = 0;
	instance_head_tags[0] = 0;

	return instance_heads;
}

bool _find_cycle(VectorI &parents, int length, vector<bool> &current_nodes,
		VectorI &ret) {
	__debug(__PRETTY_FUNCTION__)
	vector<bool> added(length);

	added[0] = true;
//	__debug(added)

	std::set<int> cycle;
	auto has_cycle = false;
	for (int i = 1; i < length; ++i) {
		if (has_cycle)
			break;
//        # don't redo nodes we've already
//        # visited or aren't considering.
		if (added[i] || !current_nodes[i])
			continue;
//        # Initialize a new possible cycle.
		std::set<int> this_cycle = { i };
		added[i] = true;
		has_cycle = true;
		auto next_node = i;

		while (!this_cycle.count(parents[next_node])) {
			next_node = parents[next_node];
//            # If we see a node we've already processed,
//            # we can stop, because the node we are
//            # processing would have been in that cycle.
			if (added[next_node]) {
				has_cycle = false;
				break;
			}
			added[next_node] = true;
			this_cycle.insert(next_node);
		}

		if (has_cycle) {
			auto original = next_node;
			cycle.insert(original);
			next_node = parents[original];
			while (next_node != original) {
				cycle.insert(next_node);
				next_node = parents[next_node];
			}
			break;
		}
	}

	ret = list(cycle);
	return has_cycle;
}

void chu_liu_edmonds(int length, Matrix &score_matrix,
		vector<bool> &current_nodes, dict<int, int> &final_edges,
		MatrixI &old_input, MatrixI &old_output,
		vector<std::set<int>> &representatives) {
	__debug(__PRETTY_FUNCTION__)
	/*
	 Applies the chu-liu-edmonds algorithm recursively
	 to a graph with edge weights defined by score_matrix.

	 Note that this function operates in place, so variables
	 will be modified.

	 Parameters
	 ----------
	 length : ``int``, required.
	 The number of nodes.
	 score_matrix : ``numpy.ndarray``, required.
	 The score matrix representing the scores for pairs
	 of nodes.
	 current_nodes : ``List[bool]``, required.
	 The nodes which are representatives in the graph.
	 A representative at it's most basic represents a node,
	 but as the algorithm progresses, individual nodes will
	 represent collapsed cycles in the graph.
	 final_edges: ``Dict[int, int]``, required.
	 An empty dictionary which will be populated with the
	 nodes which are connected in the maximum spanning tree.
	 old_input: ``numpy.ndarray``, required.
	 old_output: ``numpy.ndarray``, required.
	 representatives : ``List[Set[int]]``, required.
	 A list containing the nodes that a particular node
	 is representing at this iteration in the graph.

	 Returns
	 -------
	 Nothing - all variables are modified in place.

	 */
	VectorI parents = { -1 };
	for (int node1 = 1; node1 < length; ++node1) {
		parents.push_back(0);
		if (current_nodes[node1]) {
			auto max_score = score_matrix(0, node1);
			for (int node2 = 1; node2 < length; ++node2) {
				if (node2 == node1 || !current_nodes[node2])
					continue;

				auto new_score = score_matrix(node2, node1);
				if (new_score > max_score) {
					max_score = new_score;
					parents[node1] = node2;
				}
			}
		}
	}

//# Check if this solution has a cycle.
	VectorI cycle;
//	__debug(parents)
//	__debug(current_nodes)
	auto has_cycle = _find_cycle(parents, length, current_nodes, cycle);
//    # If there are no cycles, find all edges and return.
//	__debug(cycle)
	if (!has_cycle) {
		final_edges[0] = -1;
		for (int node = 1; node < length; ++node) {
			if (!current_nodes[node])
				continue;

			auto parent = old_input[parents[node]][node];
			auto child = old_output[parents[node]][node];
			final_edges[child] = parent;
		}
		return;
	}

	//# Otherwise, we have a cycle so we need to remove an edge.
	//# From here until the recursive call is the contraction stage of the algorithm.
	auto cycle_weight = 0.0;
	//# Find the weight of the cycle.
	auto index = 0;
	for (auto node : cycle) {
		index += 1;
		cycle_weight += score_matrix(parents[node], node);
	}

//# For each node in the graph, find the maximum weight incoming
//# and outgoing edge into the cycle.
	auto cycle_representative = cycle[0];

	for (int node = 0; node < length; ++node) {
		if (!current_nodes[node] || contains(cycle, node))
			continue;

		auto in_edge_weight = -oo;
		auto in_edge = -1;
		auto out_edge_weight = -oo;
		auto out_edge = -1;

		for (auto node_in_cycle : cycle) {
			if (score_matrix(node_in_cycle, node) > in_edge_weight) {
				in_edge_weight = score_matrix(node_in_cycle, node);
				in_edge = node_in_cycle;
			}
//        # Add the new edge score to the cycle weight
//        # and subtract the edge we're considering removing.
			auto score = (cycle_weight + score_matrix(node, node_in_cycle)
					- score_matrix(parents[node_in_cycle], node_in_cycle));

			if (score > out_edge_weight) {
				out_edge_weight = score;
				out_edge = node_in_cycle;
			}
		}

		score_matrix(cycle_representative, node) = in_edge_weight;
		old_input[cycle_representative][node] = old_input[in_edge][node];
		old_output[cycle_representative][node] = old_output[in_edge][node];

		score_matrix(node, cycle_representative) = out_edge_weight;
		old_output[node][cycle_representative] = old_output[node][out_edge];
		old_input[node][cycle_representative] = old_input[node][out_edge];
	}
//# For the next recursive iteration, we want to consider the cycle as a
//    # single node. Here we collapse the cycle into the first node in the
//    # cycle (first node is arbitrary), set all the other nodes not be
//    # considered in the next iteration. We also keep track of which
//    # representatives we are considering this iteration because we need
//    # them below to check if we're done.

	vector<std::set<int>> considered_representatives;
	for (int i = 0, size = cycle.size(); i < size; ++i) {
		auto node_in_cycle = cycle[i];
		considered_representatives.push_back(std::set<int>());
		if (i > 0) {
//            # We need to consider at least one
//            # node in the cycle, arbitrarily choose
//            # the first.
			current_nodes[node_in_cycle] = false;
		}

		for (auto node : representatives[node_in_cycle]) {
			considered_representatives[i].insert(node);
			if (i > 0)
				representatives[cycle_representative].insert(node);
		}
	}

//	__debug(length)
//	__debug(score_matrix)
//	__debug(current_nodes)
//	__debug(final_edges)
//	__debug(old_input)
//	__debug(old_output)
//	__debug(representatives)
	chu_liu_edmonds(length, score_matrix, current_nodes, final_edges, old_input,
			old_output, representatives);

//# Expansion stage.
//# check each node in cycle, if one of its representatives
//# is a key in the final_edges, it is the one we need.
	auto found = false;
	auto key_node = -1;
	for (int i = 0, size = cycle.size(); i < size; ++i) {
		auto node = cycle[i];
		for (auto cycle_rep : considered_representatives[i]) {
			if (final_edges.count(cycle_rep)) {
				key_node = node;
				found = true;
				break;
			}
		}
		if (found)
			break;
	}

	auto previous = parents[key_node];
	while (previous != key_node) {
		auto child = old_output[parents[previous]][previous];
		auto parent = old_input[parents[previous]][previous];
		final_edges[child] = parent;
		previous = parents[previous];
	}
}

VectorI decode_mst(Matrix &scores) {
//	__debug(__PRETTY_FUNCTION__)
//	int max_length = scores.rows();
	int length = scores.rows();
	auto &original_score_matrix = scores;
	auto score_matrix = scores;
	MatrixI old_input = Zero(length, length);
	MatrixI old_output = Zero(length, length);
	vector<bool> current_nodes(length, true);
	vector<std::set<int>> representatives(length);

	for (int node1 = 0; node1 < length; ++node1) {
		original_score_matrix(node1, node1) = 0.0;
		score_matrix(node1, node1) = 0.0;
		representatives[node1] = as_set( { node1 });

		for (int node2 = node1 + 1; node2 < length; ++node2) {
			old_input[node1][node2] = node1;
			old_output[node1][node2] = node2;

			old_input[node2][node1] = node2;
			old_output[node2][node1] = node1;
		}
	}

	dict<int, int> final_edges;

//	__debug(length)
//	__debug(score_matrix)
//	__debug(current_nodes)
//	__debug(final_edges)
//	__debug(old_input)
//	__debug(old_output)
//	__debug(representatives)
	chu_liu_edmonds(length, score_matrix, current_nodes, final_edges, old_input,
			old_output, representatives);

	VectorI heads(length);

	for (auto &p : final_edges) {
		auto child = p.first;
		auto parent = p.second;
		heads[child] = parent;
	}

	return heads;
}

VectorI BiaffineDependencyParser::_run_mst_decoding(const Tensor &energy,
		VectorI &instance_head_tags) {
	__debug(__PRETTY_FUNCTION__);
	Matrix scores;
	MatrixI tag_ids = argmax(energy, scores, 0);

	int seq_len = tag_ids.size();
	for (int j = 0; j < seq_len; ++j) {
		//    # Although we need to include the root node so that the MST includes it,
		//    # we do not want any word to be the parent of the root node.
		//    # Here, we enforce this by setting the scores for all word -> ROOT edges
		//    # edges to be 0.
		scores(0, j) = 0;
	}

//	__debug(scores)
//	__debug(tag_ids)
//    # Decode the heads. Because we modify the scores to prevent
//    # adding in word -> ROOT edges, we need to find the labels ourselves.
	auto instance_heads = decode_mst(scores);

//    # Find the labels which correspond to the edges in the max spanning tree.
	instance_head_tags.resize(seq_len);
//	__debug(instance_head_tags)

	for (int child = 0; child < seq_len; ++child) {
		int parent = instance_heads[child];
		if (parent < 0)
			instance_head_tags[child] = 0;
		else
			instance_head_tags[child] = tag_ids[parent][child];
	}
//    # We don't care what the head or tag is for the root token, but by default it's
//    # not necesarily the same in the batched vs unbatched case, which is annoying.
//    # Here we'll just set them to zero.
	instance_heads[0] = 0;
	instance_head_tags[0] = 0;
//	__debug(instance_head_tags)
	return instance_heads;
}

