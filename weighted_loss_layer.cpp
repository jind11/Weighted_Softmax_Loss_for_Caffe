#include <vector>

#include "caffe/layers/weighted_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void WeightedLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
}

template <typename Dtype>
void WeightedLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "The data and label should have the same first dimension.";
  CHECK_EQ(bottom[0]->shape(0), bottom[2]->shape(0))
      << "The data and weight map should have the same first dimension.";
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}

INSTANTIATE_CLASS(WeightedLossLayer);

}  // namespace caffe
