#include <iostream>
#include <string>
#include <vector>
using std::cout;
using std::endl;
using std::vector;
using std::string;
#include<string.h>
#include "H5Cpp.h"
using namespace H5;

vector<string> openAttribute(const Group &group, const char *name) {
	const Attribute &attribute = group.openAttribute(name);
	auto storageSize = attribute.getStorageSize();
	cout << "attribute.getStorageSize() = " << storageSize << endl;
	std::vector<string> attributes;
	if (!storageSize)
		return attributes;

	vector<char> buf(storageSize, '\0');

	attribute.read(attribute.getDataType(), &buf.front());

	hsize_t num_of_attributes;
	cout << "attribute.getSpace().getSimpleExtentNdims() = "
			<< attribute.getSpace().getSimpleExtentDims(&num_of_attributes, 0)
			<< endl;

	cout << "num_of_attributes = " << num_of_attributes << endl;

	hsize_t size_of_attribute = storageSize / num_of_attributes;
	cout << "size_of_attribute = " << size_of_attribute << endl;

	for (hsize_t i = 0; i < storageSize; i += size_of_attribute) {
		auto c_str = &buf[i];
		string str(c_str, c_str + size_of_attribute);
		cout << str << endl;
		attributes.push_back(str);
	}

	return attributes;
}

void reverse(char *arr, int size) {
	for (int i = 0, length = size / 2; i < length; ++i) {
		std::swap(arr[i], arr[size - 1 - i]);
	}

}

//H5File file(FILE_NAME, H5F_ACC_RDONLY);
std::pair<vector<int>, vector<double>>& read_keras_model(const H5File &file,
		const Group &group, const string &weight_name,
		std::pair<vector<int>, vector<double>> &tuple) {
	/*
	 * Try block to detect exceptions raised by any of the calls inside it
	 */

	try {
		/*
		 * Turn off the auto-printing when failure occurs so that we can
		 * handle the errors appropriately
		 */
		Exception::dontPrint();

		/*
		 * Open the specified file and the specified dataset in the file.
		 */

		DataSet weight = group.openDataSet(weight_name);

		/*
		 * Get dataspace of the dataset.
		 */
		DataSpace dataspace = weight.getSpace();

		/*
		 * Get the number of dimensions in the dataspace.
		 */
		int rank = dataspace.getSimpleExtentNdims();

		/*
		 * Get the dimension size of each dimension in the dataspace and
		 * display them.
		 */
		vector<hsize_t> shape(rank);
		int ndims = dataspace.getSimpleExtentDims(&shape.front(), NULL);

		cout << "ndims = " << ndims << endl;

		for (auto dimension : shape) {
			cout << dimension << "\t";
		}

		cout << endl;
		vector<char> buf(weight.getStorageSize(), '\0');
		weight.read(&buf.front(), weight.getDataType());

		/*
		 * Get the class of the datatype that is used by the dataset.
		 */
		switch (weight.getTypeClass()) {
		case H5T_INTEGER: {
//					* Get class of datatype and print message if it's an integer.
			cout << "Data set has INTEGER type" << endl;

			/*
			 * Get the integer datatype
			 */
			IntType inttype = weight.getIntType();

			/*
			 * Get order of datatype and print message if it's a little endian.
			 */
			H5std_string order_string;
			inttype.getOrder(order_string);
			cout << order_string << endl;

			/*
			 * Get size of the data element stored in file and print it.
			 */
			size_t size = inttype.getSize();
			cout << "integer byte size is " << size << endl;
			break;
		}
		case H5T_FLOAT: {
			cout << "Data set has FLOAT type" << endl;

			/*
			 * Get the integer datatype
			 */
			FloatType floattype = weight.getFloatType();

			/*
			 * Get size of the data element stored in file and print it.
			 */
			size_t float_size = floattype.getSize();
			cout << "float size is " << float_size << endl;

			tuple.first = vector<int>(shape.begin(), shape.end());

			/*
			 * Get order of datatype and print message if it's a little endian.
			 */

			auto byte_order = floattype.getOrder();
			vector<double> raw_data;
			for (size_t i = 0; i < buf.size(); i += float_size) {
				double x;
				if (byte_order == H5T_ORDER_BE) {
					reverse(&buf[i], float_size);
				}

				switch (float_size) {
				case 4: {
					float f;
					memcpy(&f, &buf[i], float_size);
					x = f;
					break;
				}
				case 8: {
					memcpy(&x, &buf[i], float_size);
					break;
				}
				case 16: {
					long double f;
					memcpy(&f, &buf[i], float_size);
					x = f;
					break;
				}
				}

				raw_data.push_back(x);
			}

			tuple.second = raw_data;
			return tuple;
		}

		default:
			break;
		}
	}  // end of try block

	// catch failure caused by the H5File operations
	catch (FileIException &error) {
		error.printErrorStack();
	}

	// catch failure caused by the DataSet operations
	catch (DataSetIException &error) {
		error.printErrorStack();
	}

	// catch failure caused by the DataSpace operations
	catch (DataSpaceIException &error) {
		error.printErrorStack();
	}

	// catch failure caused by the DataSpace operations
	catch (DataTypeIException &error) {
		error.printErrorStack();
	}

	return tuple;  // successfully terminated
}

