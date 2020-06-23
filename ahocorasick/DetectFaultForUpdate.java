package test;

import static test.Common.dictionaryMap;
import static test.Common.loadDictionary;
import static test.Common.naiveConstruct;
import static test.Common.naiveUpdate;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.TreeMap;

import com.util.Utility;

import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class DetectFaultForUpdate extends TestCase {

	void initialize(Collection<String> dictionary) throws IOException {
		dictionaryMap = new TreeMap<String, String>();
		for (String word : dictionary) {
			dictionaryMap.put(word, String.format("[%s]", word));
		}
		System.out.println("dictionary.size() = " + dictionary.size());
	}

	public boolean testNaiveUpdate() throws Exception {
		return naiveConstruct().equals(naiveUpdate());
	}

	void rotate(ArrayList<String> list) {
		int initial_size = list.size();

		int size_to_be_moved = initial_size / split_size;

		int new_size = initial_size - size_to_be_moved;
		ArrayList<String> newlist = new ArrayList<String>();

		newlist.addAll(list.subList(new_size, initial_size));
		newlist.addAll(list.subList(0, new_size));

		list.clear();
		list.addAll(newlist);
	}

	final int split_size = 5;

	boolean run_epoch(ArrayList<String> list) throws Exception {
		ArrayList<String> newlist;
		for (int i = 0; i < split_size; ++i) {
			int initial_size = list.size();

			int size_to_be_moved = initial_size / split_size;
			if (size_to_be_moved == 0)
				return false;
			int new_size = initial_size - size_to_be_moved;

			newlist = new ArrayList<String>();
//[*][*][*](*)			
			newlist.addAll(list.subList(0, new_size));
			initialize(newlist);
			if (!testNaiveUpdate()) {
				Utility.writeString("dictionary.txt", newlist);
				System.out.println("successfully shrinking the erroneous dataset!");
				System.out.println("newlist.size() = " + newlist.size());
				return true;
			}
			rotate(list);
		}
		return false;
	}

	public void test_detect_fault() throws Exception {
		Utility.writeString("dictionary.txt", loadDictionary("dictionary.txt", 80000));
		do {
			ArrayList<String> list = loadDictionary("dictionary.txt");
			if (!run_epoch(list))
				break;
		} while (true);
	}
}
