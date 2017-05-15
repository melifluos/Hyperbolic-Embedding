#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
        .Input("to_zero: int32")
        .Output("zeroed: int32")
        .Output("o2: int32");
//        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
//            c->set_output(0, c->input(0));
//            return Status::OK();
//        });


class ZeroOutOp : public OpKernel {
public:
    explicit ZeroOutOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat<int32>();

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));
        auto output = output_tensor->flat<int32>();

        Tensor *output_tensor1 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, input_tensor.shape(),
                                                         &output_tensor1));
        auto output1 = output_tensor1->flat<int32>();

        // Set all but the first element of the output tensor to 0.
        const int N = input.size();
        for (int i = 0; i < N; i++) {
            output(i) = 0;
            output1(i) = 0;
        }

        // Preserve the first input value if possible.
        if (N > 0) output(0) = input(0);
    }
//
//    Tensor eval(OpKernelContext *context) {
//        return context->output(0);
//    }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
