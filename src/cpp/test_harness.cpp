//
// Created by Ben Chamberlain on 27/03/2017.
//

#include "test_harness.h"


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <vector>
#include <iterator>
#include <unordered_map>

using namespace std;

//
//void split(const std::string &s, char delim, int *result) {
//    std::stringstream ss;
//    ss.str(s);
//    std::string item;
//    while (std::getline(ss, item, delim)) {
//        *(result++) = item;
//    }
//}
const int kPrecalc = 3000;


class Context {

public:
    Context(std::string filename) {
        filename_ = filename;
        examples.reserve(batch_size_);
        labels.reserve(batch_size_);
    }

    int words_per_epoch = 0;
    int current_epoch = 0;
    int total_words_processed = 0;
    int batch_size = 4;
    vector<int> examples;
    vector<int> labels;
    std::string filename_;
    int batch_size_ = 4;
    int window_size_ = 5;
    int min_count_ = 0;

};

class TestHarness {
public:
    TestHarness(Context *ctx);

    void ReadInput(string inpath) {
        string line;
        ifstream myfile;
        vector<vector<int>> corpus;
        typedef std::pair<int, int> WordFreq;
        unordered_map<int, int> word_freq;
        myfile.open(ctx_->filename_);
        if (myfile.is_open()) {
            cout << "outputting file contents \n";
            while (getline(myfile, line)) {
                nrows_++;
                vector<int> int_line;
                cout << line << '\n';
                int val;
                stringstream iss(line);
                while (iss >> val) {
                    ++(word_freq[val]);
                    int_line.push_back(val);
                    if (iss.peek() == ',' || iss.peek() == ' ')
                        iss.ignore();
                    corpus_size_++;
                }
                corpus.push_back(int_line);
            }
            myfile.close();

        }
        cout << "Data file: " << inpath << " contains " << corpus_size_ << " words, " << word_freq.size()
             << " unique words\n";
        PrintMap(word_freq);

        std::vector<WordFreq> ordered;
        for (const auto &p : word_freq) {
            if (p.second >= min_count_) ordered.push_back(p);
        }

        sort(ordered.begin(), ordered.end(),
             [](const WordFreq &x, const WordFreq &y) {
                 return x.second > y.second;
             });

        for (size_t i=0; i < ordered.size(); i++) {
            PrintWordFreq(ordered[i]); }

        std::unordered_map<int, int> word_id;
        int total_counted = 0;
        for (std::size_t i = 0; i < ordered.size(); ++i) {
            const auto &w = ordered[i].first;
            auto id = i;
            word_.push_back(w);
            auto word_count = ordered[i].second;
            freq_.push_back(word_count);
            total_counted += word_count;
            word_id[w] = id;
        }

        PrintMap(word_id);

        // create a corpus of indices instead of IDs
        for (size_t row=0;row<corpus.size();row++){
            vector<int> sentence = corpus[row];
            vector<int> idx_sentence;
            for (size_t elem=0; elem<sentence.size(); elem++){
                int id = sentence[elem];
                int idx = word_id[id];
                idx_sentence.push_back(idx);
            }
            corpus_.push_back(idx_sentence);
        }

        sentence_ = corpus_[corpus_idx_];
        sentence_size_ = sentence_.size();
        label_limit_ = std::min<int>(sentence_.size(), window_size_ + 1);
    }

    void PrintMap(unordered_map<int, int> map) {
        // Iterate and print keys and values of unordered_map
        for (const auto &n : map) {
            std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
        }
    }

    void Compute() {
        for (int i = 0; i < batch_size_; ++i) {
            ctx_->examples[i] = precalc_examples_[precalc_index_].input;
            ctx_->labels[i] = precalc_examples_[precalc_index_].label;
            precalc_index_++;
//            std::cout << "precalc examples size " << precalc_examples_.size() << '\n';
            if (precalc_index_ >= kPrecalc) {
                precalc_index_ = 0;
                for (int j = 0; j < kPrecalc; ++j) {
                    NextExample(&precalc_examples_[j].input,
                                &precalc_examples_[j].label);
                }
            }
        }
        ctx_->current_epoch = current_epoch_;
//        ctx_->examples = examples_;
//        ctx_->labels = labels_;
        ctx_->total_words_processed = total_words_processed_;
        ctx_->words_per_epoch = corpus_size_;
    }

    void PrintWordFreq(pair<int, int> p) {
        cout << "word: " << p.first << " count: " << p.second << '\n';
    }

    void PrintOutput() {
        cout << "printing batch \n";
        for (size_t i = 0; i < batch_size_; i++) {
            cout << examples_[i] << ':' << labels_[i] << '\n';
        }
    }

    void NextExample(int *label, int *example) {
        while (true) {
            if (label_pos_ >= label_limit_) {
                ++total_words_processed_;
                ++sentence_index_;
                // check if we hit the end of the sentence
                if (sentence_index_ >= sentence_size_) {
                    sentence_index_ = 0;
                    ++corpus_idx_;
                    // check if we hit the end of the corpus
                    if (corpus_idx_ >= nrows_) {
                        ++current_epoch_;
                        corpus_idx_ = 0;
                    }
                    sentence_ = corpus_[corpus_idx_];
                    sentence_size_ = sentence_.size();

                }
                label_pos_ = std::max<int>(0, sentence_index_ - window_size_);
                label_limit_ =
                        std::min<int>(sentence_.size(), sentence_index_ + window_size_ + 1);
            }
            if (sentence_[sentence_index_] != sentence_[label_pos_]) // add constraint that tokens are different
                break;
            ++label_pos_;
        }
        std::cout << "example: " << sentence_[sentence_index_] << ' ';
        std::cout << "label: " << sentence_[label_pos_] << '\n';
        *example = sentence_[sentence_index_];
        *label = sentence_[label_pos_++];
    }

    void PrintCorpus() {
        for (size_t i = 0; i < corpus_.size(); i++) {
            vector<int> out_line = corpus_[i];
            for (size_t j = 0; j < out_line.size(); j++) {
                cout << out_line[j] << ' ';
            }
            cout << '\n';

        }
    }

    void PrintContext() {
        cout << "words per epoch: " << ctx_->words_per_epoch << '\n';
        cout << "current epoch: " << ctx_->current_epoch << '\n';
        cout << "total words processed: " << ctx_->total_words_processed << '\n';
        cout << "printing batch \n";
        for (size_t i = 0; i < batch_size_; i++) {
            cout << ctx_->examples[i] << ':' << ctx_->labels[i] << '\n';
        }
    }


private:
    struct Example {
        int input;
        int label;
    };
    Context *ctx_;
    int sentence_index_ = 0;  // the index of the example into the current sentence
    int corpus_idx_ = 0; // the index of the current sentence into the corpus
    int corpus_size_ = 0;
    int nrows_ = 0;
    // Number of examples to precalculate.
    int sentence_size_;
    vector<int> sentence_;
    int label_pos_ = 0;
    int total_words_processed_ = 0;
    int label_limit_;
    int current_epoch_ = 0;
    int window_size_ = 5;
    const int batch_size_ = 64;
    vector<int> examples_;
    vector<int> labels_;
    vector<vector<int>> corpus_;
    std::vector<Example> precalc_examples_;
    int precalc_index_ = 0;
    int min_count_ = 0;
    vector<int> word_;
    vector<int> freq_;


};

TestHarness::TestHarness(Context *ctx) {
    ctx_ = ctx;
    TestHarness::ReadInput(ctx_->filename_);
    examples_.resize(batch_size_);
    labels_.resize(batch_size_);
    precalc_examples_.resize(kPrecalc);
    for (int i = 0; i < kPrecalc; ++i) {
        NextExample(&precalc_examples_[i].input, &precalc_examples_[i].label);
    }
}


int main() {
//    string inpath = "../../../test/cpp_testfile.csv";
    string inpath = "../../../local_resources/sentences1in10000.csv";
    Context ctx = Context(inpath);
//    std::cout << "Hello World!";
    TestHarness th = TestHarness(&ctx);
    th.PrintCorpus();
    for (size_t i = 0; i < 1; i++) {
//        int label;
//        int example;
//        th.NextExample(&example, &label);
        th.Compute();
        th.PrintContext();
    }
    return 0;
}