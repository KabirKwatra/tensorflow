/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CONVERT_TENSOR_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CONVERT_TENSOR_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

using stream_executor::port::StatusOr;

// Converts an TensorFlow tensor proto into an MLIR elements attribute.
StatusOr<mlir::ElementsAttr> ConvertTensorProto(const TensorProto& input_tensor,
        mlir::Builder* builder);

// Converts an TensorFlow tensor into an MLIR elements attribute.
StatusOr<mlir::ElementsAttr> ConvertTensor(const Tensor& input_tensor,
        mlir::Builder* builder);

// Converts a shape from MLIR to a TensorFlow tensor shape proto.
void ConvertToTensorShapeProto(llvm::ArrayRef<int64_t> shape,
                               TensorShapeProto* output_shape);

// Converts an MLIR type to a TensorFlow tensor shape.
PartialTensorShape ConvertTypeToTensorShape(const mlir::Type& type);

// Converts an MLIR elements attribute to a TensorFlow tensor proto.
Status ConvertToTensorProto(mlir::ElementsAttr attr,
                            TensorProto* output_tensor);

// Converts an MLIR elements attribute to a TensorFlow tensor.
Status ConvertToTensor(mlir::ElementsAttr attr, Tensor* output_tensor);

// Decodes the given opaque elements attribute holding tensor content into a
// human-readable elements attribute.
StatusOr<mlir::ElementsAttr> DecodeOpaqueTensor(
    mlir::OpaqueElementsAttr input_attr, mlir::Builder builder);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CONVERT_TENSOR_H_
