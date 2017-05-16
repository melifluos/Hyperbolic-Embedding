//
// Created by Ben Chamberlain on 27/03/2017.
//

//
// Created by Ben Chamberlain on 21/03/2017.
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <vector>
#include <iterator>


namespace tensorflow {

    REGISTER_OP("SkipgramWord2vec")
            .Output("vocab_word: int32")
            .Output("vocab_freq: int32")
            .Output("words_per_epoch: int64")
            .Output("current_epoch: int32")
            .Output("total_words_processed: int64")
            .Output("examples: int32")
            .Output("labels: int32")
            .SetIsStateful()
            .Attr("filename: string")
            .Attr("batch_size: int")
            .Attr("window_size: int = 5")
//            .Attr("min_count: int = 5")
//            .Attr("subsample: float = 1e-3")
            .Doc(R"doc(
Parses a text file and creates a batch of examples.
words_per_epoch: Number of words per epoch in the data file.
current_epoch: The current epoch number.
total_words_processed: The total number of words processed so far.
examples: A vector of word ids.
labels: A vector of word ids.
filename: The corpus's text file name.
batch_size: The size of produced batch.
window_size: The number of words to predict to the left and right of the target.
)doc");


    // Number of examples to precalculate.
    const int kPrecalc = 3000;
// Number of words to read into a sentence before processing.

    namespace {
        // Reads the next word into word and advances the pointer into *input
        bool ScanWord(StringPiece *input, string *word) {
            str_util::RemoveLeadingWhitespace(input);
            StringPiece tmp;
            if (str_util::ConsumeNonWhitespace(input, &tmp)) {
                word->assign(tmp.data(), tmp.size());
                return true;
            } else {
                return false;
            }
        }

    }  // end namespace


    class SkipgramWord2vecOp : public OpKernel {
    public:
        explicit SkipgramWord2vecOp(OpKernelConstruction *ctx)
//        things to initialise
                : OpKernel(ctx), rng_(&philox_) {
            string filename;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
//            OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &min_count_));
//            OP_REQUIRES_OK(ctx, ctx->GetAttr("subsample", &subsample_));
            OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));

            mutex_lock l(mu_);
            sentence_ = corpus2d_[corpus_idx_];
//            sentence_size_ = sentence_.size();
            sentence_index_ = 0;
            label_pos_ = 0;
            label_limit_ = std::min<int32>(sentence_.size(), window_size_ + 1);
            for (int i = 0; i < kPrecalc; ++i) {
                NextExample(&precalc_examples_[i].input, &precalc_examples_[i].label);
            }
        }

        void ReadInput(string filename) {
            string line;
            std::ifstream myfile;
            std::vector<std::vector<int32>> corpus;
            typedef std::pair<int32, int32> WordFreq;
            std::unordered_map<int32, int32> word_freq;
            myfile.open(filename);
            if (myfile.is_open()) {
                std::cout << "outputting file contents \n";
                while (getline(myfile, line)) {
                    nrows_++;
                    std::vector<int32> int_line;
//                    std::cout << line << '\n';
                    int val;
                    std::stringstream iss(line);
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
            std::cout << "file written to corpus \n";
            std::cout << "Data file: " << filename << " contains " << corpus_size_ << " words, " << word_freq.size()
                      << " unique words\n";

            std::vector<WordFreq> ordered;
            for (const auto &p : word_freq) {
                if (p.second >= min_count_) ordered.push_back(p);
            }
            std::cout << "sorting by frequency \n";
            std::sort(ordered.begin(), ordered.end(),
                      [](const WordFreq &x, const WordFreq &y) {
                          return x.second > y.second;
                      });
            std::cout << "sorting complete \n";
            vocab_size_ = static_cast<int32>(ordered.size());
            Tensor word(DT_INT32, TensorShape({vocab_size_}));
            Tensor freq(DT_INT32, TensorShape({vocab_size_}));
            std::unordered_map<int32, int32> word_id;
            int64 total_counted = 0;
            std::cout << "creating index \n";
            for (std::size_t i = 0; i < ordered.size(); ++i) {
                const auto &w = ordered[i].first;
                auto id = i;
                word.flat<int32>()(id) = w;
                auto word_count = ordered[i].second;
                freq.flat<int32>()(id) = word_count;
                word_id[w] = id;
            }
            std::cout << "assigning tensors \n";
            word_ = word;
            freq_ = freq;

            // create a corpus of indices instead of IDs
            std::cout << "re-indexing IDs \n";
            for (size_t row = 0; row < corpus.size(); row++) {
                std::vector<int32> sentence = corpus[row];
                std::vector<int32> idx_sentence;
                for (size_t elem = 0; elem < sentence.size(); elem++) {
                    int32 id = sentence[elem];
                    int32 idx = word_id[id];
                    idx_sentence.push_back(idx);
                }
                corpus2d_.push_back(idx_sentence);
            }
        }

        void Compute(OpKernelContext *ctx) override {
//            std::cout << "calling compute\n";
            Tensor words_per_epoch(DT_INT64, TensorShape({}));
            Tensor current_epoch(DT_INT32, TensorShape({}));
            Tensor total_words_processed(DT_INT64, TensorShape({}));
            Tensor examples(DT_INT32, TensorShape({batch_size_}));
            auto Texamples = examples.flat<int32>();
            Tensor labels(DT_INT32, TensorShape({batch_size_}));
            auto Tlabels = labels.flat<int32>();
            {
                mutex_lock l(mu_);
                for (int i = 0; i < batch_size_; ++i) {
//                    std::cout << "populating tensors \n";
                    Texamples(i) = precalc_examples_[precalc_index_].input;
                    Tlabels(i) = precalc_examples_[precalc_index_].label;
                    precalc_index_++;
//                    std::cout << "precalc examples size " << precalc_examples_.size() << '\n';
//                    std::cout << "precalc index " << precalc_index_ << '\n';
                    if (precalc_index_ >= kPrecalc) {
//                        std::cout << "generating more precalculated examples\n";
                        precalc_index_ = 0;
//                        int initial_epoch = current_processed_epoch
                        for (int j = 0; j < kPrecalc; ++j) {
                            NextExample(&precalc_examples_[j].input,
                                        &precalc_examples_[j].label);
                        }
                    }
                }
//                std::cout << "setting tensors \n";
                words_per_epoch.scalar<int64>()() = corpus_size_;
                current_epoch.scalar<int32>()() = current_epoch_;
                total_words_processed.scalar<int64>()() = total_words_processed_;
            }
//            std::cout << "writing output \n";
//            ctx->set_output(0, words_per_epoch);
//            ctx->set_output(1, current_epoch);
//            ctx->set_output(2, total_words_processed);
//            std::cout << "writing examples \n";
//            ctx->set_output(3, examples);
//            std::cout << "writing labels \n";
//            ctx->set_output(4, labels);
//            std::cout << "exiting Compute \n";

            ctx->set_output(0, word_);
            ctx->set_output(1, freq_);
            ctx->set_output(2, words_per_epoch);
            ctx->set_output(3, current_epoch);
            ctx->set_output(4, total_words_processed);
            ctx->set_output(5, examples);
            ctx->set_output(6, labels);
        }


    private:
        struct Example {
            int32 input;
            int32 label;
        };
        int nrows_ = 0;
        int32 batch_size_ = 0;
        int32 window_size_ = 5;
        float subsample_ = 1e-3;
        int min_count_ = 0;
        int32 vocab_size_ = 0;
        Tensor word_;
        Tensor freq_;

//        int sentence_size_;

        int64 corpus_size_ = 0;
//        std::vector<int32> corpus_;
        std::vector<std::vector<int32>> corpus2d_;
        std::vector<Example> precalc_examples_;
        int precalc_index_ = 0;
        std::vector<int32> sentence_;

        mutex mu_;
        // A Philox is a parallel random number generator that was introduced here:
        // http://ieeexplore.ieee.org/abstract/document/6114424/
        random::PhiloxRandom philox_ GUARDED_BY(mu_);
        random::SimplePhilox rng_ GUARDED_BY(mu_);
        int32 current_epoch_ GUARDED_BY(mu_) = 0;
        int64 total_words_processed_ GUARDED_BY(mu_) = 0;
        int32 example_pos_ GUARDED_BY(mu_);
        // the index of the example into the current sentence
        int32 sentence_index_ GUARDED_BY(mu_);
        // the index of the current sentence into the corpus
        int32 corpus_idx_ GUARDED_BY(mu_) = 0;
        int32 label_pos_ GUARDED_BY(mu_);
        int32 label_limit_ GUARDED_BY(mu_);


        void NextExample(int32 *example, int32 *label) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
//            std::cout << "next example called\n";
            while (true) {
                if (label_pos_ >= label_limit_) {
//                    std::cout << "label position exceeded limit. Updating\n";
                    ++total_words_processed_;
                    ++sentence_index_;
                    // check if we hit the end of the sentence
                    if (sentence_index_ >= sentence_.size()) {
                        sentence_index_ = 0;
                        ++corpus_idx_;
                        // check if we hit the end of the corpus
                        if (corpus_idx_ >= nrows_) {
                            ++current_epoch_;
                            corpus_idx_ = 0;
                        }
                        sentence_ = corpus2d_[corpus_idx_];

                    }
                    label_pos_ = std::max<int>(0, sentence_index_ - window_size_);
                    label_limit_ =
                            std::min<int>(sentence_.size(), sentence_index_ + window_size_ + 1);
                }
                if (sentence_[sentence_index_] != sentence_[label_pos_]) // add constraint that tokens are different
                    break;
                ++label_pos_;
            }
            *example = sentence_[sentence_index_];
            *label = sentence_[label_pos_++];
        }


//        Read data into the corpus private member
        Status Init(Env *env, const string &filename) {
            ReadInput(filename);
            precalc_examples_.resize(kPrecalc);


            return Status::OK();
        }
    };

    REGISTER_KERNEL_BUILDER(Name("SkipgramWord2vec").Device(DEVICE_CPU), SkipgramWord2vecOp);
}