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

namespace tensorflow {

    REGISTER_OP("SkipgramWord2vec")
            .Output("vocab_word: string")
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
            .Attr("min_count: int = 5")
            .Attr("subsample: float = 1e-3")
            .Doc(R"doc(
Parses a text file and creates a batch of examples.
vocab_word: A vector of words in the corpus.
vocab_freq: Frequencies of words. Sorted in the non-ascending order.
words_per_epoch: Number of words per epoch in the data file.
current_epoch: The current epoch number.
total_words_processed: The total number of words processed so far.
examples: A vector of word ids.
labels: A vector of word ids.
filename: The corpus's text file name.
batch_size: The size of produced batch.
window_size: The number of words to predict to the left and right of the target.
min_count: The minimum number of word occurrences for it to be included in the
    vocabulary.
subsample: Threshold for word occurrence. Words that appear with higher
    frequency will be randomly down-sampled. Set to 0 to disable.
)doc");


// Number of examples to precalculate.
    const int kPrecalc = 3000;
// Number of words to read into a sentence before processing.
    const int kSentenceSize = 1000;

    namespace {

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
            OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &min_count_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("subsample", &subsample_));
            OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));

            mutex_lock l(mu_);
            example_pos_ = corpus_size_;
            label_pos_ = corpus_size_;
            label_limit_ = corpus_size_;
            sentence_index_ = kSentenceSize;
            for (int i = 0; i < kPrecalc; ++i) {
                NextExample(&precalc_examples_[i].input, &precalc_examples_[i].label);
            }
        }
//abstract base function
        void Compute(OpKernelContext *ctx) override {
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
                    Texamples(i) = precalc_examples_[precalc_index_].input;
                    Tlabels(i) = precalc_examples_[precalc_index_].label;
                    precalc_index_++;
                    if (precalc_index_ >= kPrecalc) {
                        precalc_index_ = 0;
                        for (int j = 0; j < kPrecalc; ++j) {
                            NextExample(&precalc_examples_[j].input,
                                        &precalc_examples_[j].label);
                        }
                    }
                }
                words_per_epoch.scalar<int64>()() = corpus_size_;
                current_epoch.scalar<int32>()() = current_epoch_;
                total_words_processed.scalar<int64>()() = total_words_processed_;
            }
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

        int32 batch_size_ = 0;
        int32 window_size_ = 5;
        float subsample_ = 1e-3;
        int min_count_ = 5;
        int32 vocab_size_ = 0;
        Tensor word_;
        Tensor freq_;
        int64 corpus_size_ = 0;
        std::vector<int32> corpus_;
        std::vector<Example> precalc_examples_;
        int precalc_index_ = 0;
        std::vector<int32> sentence_;
        int sentence_index_ = 0;

        mutex mu_;
        random::PhiloxRandom philox_ GUARDED_BY(mu_);
        random::SimplePhilox rng_ GUARDED_BY(mu_);
        int32 current_epoch_ GUARDED_BY(mu_) = -1;
        int64 total_words_processed_ GUARDED_BY(mu_) = 0;
        int32 example_pos_ GUARDED_BY(mu_);
        int32 label_pos_ GUARDED_BY(mu_);
        int32 label_limit_ GUARDED_BY(mu_);

        // {example_pos_, label_pos_} is the cursor for the next example.
        // example_pos_ wraps around at the end of corpus_. For each
        // example, we randomly generate [label_pos_, label_limit) for
        // labels.
        void NextExample(int32 *example, int32 *label) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            while (true) {
                if (label_pos_ >= label_limit_) {
                    ++total_words_processed_;
                    ++sentence_index_;
                    if (sentence_index_ >= kSentenceSize) {
                        sentence_index_ = 0;
                        for (int i = 0; i < kSentenceSize; ++i, ++example_pos_) {
                            if (example_pos_ >= corpus_size_) {
                                ++current_epoch_;
                                example_pos_ = 0;
                            }
                            if (subsample_ > 0) {
                                int32 word_freq = freq_.flat<int32>()(corpus_[example_pos_]);
                                // See Eq. 5 in http://arxiv.org/abs/1310.4546
                                float keep_prob =
                                        (std::sqrt(word_freq / (subsample_ * corpus_size_)) + 1) *
                                        (subsample_ * corpus_size_) / word_freq;
                                if (rng_.RandFloat() > keep_prob) {
                                    i--;
                                    continue;
                                }
                            }
                            sentence_[i] = corpus_[example_pos_];
                        }
                    }
                    const int32 skip = 1 + rng_.Uniform(window_size_);
                    label_pos_ = std::max<int32>(0, sentence_index_ - skip);
                    label_limit_ =
                            std::min<int32>(kSentenceSize, sentence_index_ + skip + 1);
                }
                if (sentence_index_ != label_pos_) {
                    break;
                }
                ++label_pos_;
            }
            *example = sentence_[sentence_index_];
            *label = sentence_[label_pos_++];
        }

        Status Init(Env *env, const string &filename) {
            string data;
            TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
            StringPiece input = data;
            string w;
            corpus_size_ = 0;
            std::unordered_map<string, int32> word_freq;
            while (ScanWord(&input, &w)) {
                ++(word_freq[w]);
                ++corpus_size_;
            }
            if (corpus_size_ < window_size_ * 10) {
                return errors::InvalidArgument("The text file ", filename,
                                               " contains too little data: ",
                                               corpus_size_, " words");
            }
            typedef std::pair<string, int32> WordFreq;
            std::vector<WordFreq> ordered;
            for (const auto &p : word_freq) {
                if (p.second >= min_count_) ordered.push_back(p);
            }
            LOG(INFO) << "Data file: " << filename << " contains " << data.size()
                      << " bytes, " << corpus_size_ << " words, " << word_freq.size()
                      << " unique words, " << ordered.size()
                      << " unique frequent words.";
            word_freq.clear();
            std::sort(ordered.begin(), ordered.end(),
                      [](const WordFreq &x, const WordFreq &y) {
                          return x.second > y.second;
                      });
            vocab_size_ = static_cast<int32>(1 + ordered.size());
            Tensor word(DT_STRING, TensorShape({vocab_size_}));
            Tensor freq(DT_INT32, TensorShape({vocab_size_}));
            word.flat<string>()(0) = "UNK";
            static const int32 kUnkId = 0;
            std::unordered_map<string, int32> word_id;
            int64 total_counted = 0;
            for (std::size_t i = 0; i < ordered.size(); ++i) {
                const auto &w = ordered[i].first;
                auto id = i + 1;
                word.flat<string>()(id) = w;
                auto word_count = ordered[i].second;
                freq.flat<int32>()(id) = word_count;
                total_counted += word_count;
                word_id[w] = id;
            }
            // set the first element to be the total number of infrequent words that all get assigned to UNK
            freq.flat<int32>()(kUnkId) = corpus_size_ - total_counted;
            word_ = word;
            freq_ = freq;
            corpus_.reserve(corpus_size_);
            input = data;
            while (ScanWord(&input, &w)) {
                corpus_.push_back(gtl::FindWithDefault(word_id, w, kUnkId));
            }
            precalc_examples_.resize(kPrecalc);
            sentence_.resize(kSentenceSize);
            return Status::OK();
        }
    };

    REGISTER_KERNEL_BUILDER(Name("SkipgramWord2vec").Device(DEVICE_CPU), SkipgramWord2vecOp);

}

