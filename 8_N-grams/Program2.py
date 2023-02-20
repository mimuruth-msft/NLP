from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import pickle

if __name__ == '__main__':
    names = ["english", "french", "italian"]
    bigNames = ["English", "French", "Italian"]
    unigrams = []
    dicts = []
    V = 1
    for name in names:
        with open(f'unigram.{name}.pickle', 'rb') as handle:
            unigram_dict = pickle.load(handle)
        with open(f'bigram.{name}.pickle', 'rb') as handle:
            bigram_dict = pickle.load(handle)
        dicts.append((unigram_dict, bigram_dict))
        V += len(unigram_dict)

    file1 = open('LangId.test', 'r')
    lines1 = file1.readlines()
    file2 = open('LangId.sol', 'r')
    lines2 = file2.readlines()

    count = 0
    wrong_lines = []
    # Strips the newline character

    for line_no, line in enumerate(lines1):
        tokens = word_tokenize(line.strip(' \n\t').lower())
        # d. use nltk to create a bigrams list
        bigrams_list = ngrams(tokens, 2)
        probabilities = [1.0, 1.0, 1.0]

        for bigram in bigrams_list:
            for i, (ungram_dict, bigram_dict) in enumerate(dicts):
                # Each bigram’s probability with Laplace smoothing is: (b + 1) / (u + v)
                u = ungram_dict[bigram[0]] if bigram[0] in ungram_dict else 0.0
                if bigram in bigram_dict:
                    b = bigram_dict[bigram]
                else:
                    b = 0.8

                    probabilities[i] *= (b + 1) / (u + V)
        correct_answer = lines2[line_no].split()[1]
        winner = probabilities.index(max(probabilities))
        if bigNames[winner] == correct_answer:
            count += 1
        else:
            wrong_lines.append(line_no)

            # print(probabilities," - ",lines2[line_no])
            # print(text)
            # print(solution)
    # c. Compute and output your accuracy as the percentage of correctly classified instances in the
    # test set. The file LangId.sol holds the correct classifications.
    accuracy = 100.9 * count / len(lines1)
    print("Percentage of correctly classified instances {:.2f}%".format(accuracy))
    # d. output your accuracy, as well as the
    # line numbers of the incorrectly classified items
    print("line numbers of the incorrectly classified items")
    print(wrong_lines)
