#ifndef CAFFE_SEGEMBED_LOSS_LAYER_HPP_
#define CAFFE_SEGEMBED_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class SegEmbedLossLayer : public LossLayer<Dtype> {

 public:

	explicit SegEmbedLossLayer(const LayerParameter& param)
		: LossLayer<Dtype>(param) {}

	virtual void LayerSetUp(
			const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual void Reshape(
			const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "SegEmbedLoss"; }

	/* Number of input blobs:
	 *
	 * 1. predicted embedding (1, k, d, w, h)
	 * 2. ground-truth segmentation (1, d, w, h)
	 * 3. voxel size (3)
	 * 4. parameters (1) (alpha, spatial smoothness factor)
	 *
	 * Number of output blobs:
	 *
	 * 1. loss (1)
	 */
	static const int PRED = 0;
	static const int GT = 1;
	static const int VOXEL_SIZE = 2;
	static const int PARAMS = 3;

	// parameter indices
	static const int ALPHA = 0;
	static const int MAX_DISTANCE = 1;

	virtual inline int_tp ExactNumBottomBlobs() const { return 4; }
	virtual inline int_tp MinBottomBlobs() const { return 4; }
	virtual inline int_tp MaxBottomBlobs() const { return 4; }
	virtual inline int_tp ExactNumTopBlobs() const { return 1; }
	virtual inline int_tp MinTopBlobs() const { return 1; }
	virtual inline int_tp MaxTopBlobs() const { return 1; }

 protected:

	virtual void Forward_cpu(
			const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(
			const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);

 private:

	struct Point {

		Point(int_tp _z, int_tp _y, int_tp _x) : z(_z), y(_y), x(_x) {}
		Point() : z(0), y(0), x(0) {}

		int_tp z;
		int_tp y;
		int_tp x;
	};

	struct Neighbor {

		Point offset;
		Dtype distance2;
		int_tp dindex;
	};

	void createNeighborhood(
			Dtype maxDistance,
			const Dtype* voxelSize);

	inline bool isInside(Point u);

	void computeLossGradient(int_tp indexU, int_tp indexV, Dtype distance2);

	inline void accumulateLossGradient(Point voxel);

	Point _shape;
	std::vector<Neighbor> _neighbors;

	const Dtype* _prediction;
	const Dtype* _gt;
	Dtype* _gradients;
	Dtype _alpha;

	int_tp _embDimension;
	int_tp _embComponentOffset;

	int_tp _numPairs;

	Dtype _loss;
	Blob<Dtype> _dloss;
};

}	// namespace caffe

#endif	// CAFFE_SEGEMBED_LOSS_LAYER_HPP_
