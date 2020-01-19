package test;

import static test.Common.dictionaryMap;
import static test.Common.naiveConstruct;
import static test.Common.naiveDelete;
import static test.Common.text;
import static test.Common.wordsToBeDeleted;

import org.ahocorasick.trie.Trie;

import com.util.Utility;

import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestDelete extends TestCase {
	static {
		dictionaryMap.remove(wordsToBeDeleted);
	}

	public void testSearchTrie() throws Exception {
		try (Utility.Printer printer = new Utility.Printer("./debug.txt")) {
			for (String word : Common.loadDictionary()) {
				dictionaryMap.put(wordsToBeDeleted, wordsToBeDeleted);
				dictionaryMap.remove(word);
				wordsToBeDeleted = word;
				System.out.println("testing word: " + word);

				Trie naiveConstruct = naiveConstruct();
				Trie naiveUpdate = naiveDelete();

				boolean equals = naiveConstruct.rootState.equals(naiveUpdate.rootState);

				assertTrue(equals);
				assertEquals(naiveConstruct.parseText(text).size(), naiveUpdate.parseText(text).size());
			}
		}
	}
}
