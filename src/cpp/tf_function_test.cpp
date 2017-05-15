/*
// Created by Ben Chamberlain on 28/03/2017
Needed because I can't get CLION to link to
 */

#include <iostream>
#include <fstream>
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


class TestHarness {
public:
    TestHarness();

    void FileReadTest(const tensorflow::string &inpath) {
        tensorflow::string data;
        tensorflow::OpKernelConstruction *ctx;
        tensorflow::ReadFileToString(ctx->env(), inpath, &data);
        std::cout << data;
    }

private:
    int row_idx;


};

int main() {
    std::string inpath = "../../../test/cpp_testfile.csv";
    std::cout << "Hello World!";
    TestHarness th = TestHarness();
    th.FileReadTest(inpath);
//        std::cout << dummy;
    return 0;
}
