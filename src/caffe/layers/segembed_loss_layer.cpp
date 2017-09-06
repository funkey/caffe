#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/segembed_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void SegEmbedLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template<typename Dtype>
void SegEmbedLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	LossLayer<Dtype>::Reshape(bottom, top);

	// (1[batch], k[dimensionality], depth, height, width)
	vector<int_tp> predShape = bottom[PRED]->shape();

	assert(predShape[0] == 1);
	_embDimension = predShape[1];
	_shape = Point(
			predShape[2],
			predShape[3],
			predShape[4]);

	_embComponentOffset = _shape.z*_shape.y*_shape.x;

	_dloss.Reshape(predShape);
}

template<typename Dtype>
void
SegEmbedLossLayer<Dtype>::createNeighborhood(
		Dtype maxDistance2,
		const Dtype* voxelSize) {

	_neighbors.clear();

	Point maxSize;
	maxSize.z = maxDistance2/voxelSize[0];
	maxSize.y = maxDistance2/voxelSize[1];
	maxSize.x = maxDistance2/voxelSize[2];

	std::cout << "neighborhood:" << std::endl;
	for (int_tp dz = -maxSize.z; dz <= maxSize.z; dz++)
		for (int_tp dy = -maxSize.y; dy <= maxSize.y; dy++)
			for (int_tp dx = -maxSize.x; dx <= maxSize.x; dx++) {

				// offset into flattened array
				int_tp dindex = dx + dy*_shape.x + dz*_shape.x*_shape.y;

				// consider only forward offsets
				if (dindex <= 0)
					continue;

				Dtype distance2 =
						dz*dz*voxelSize[0]*voxelSize[0] +
						dy*dy*voxelSize[1]*voxelSize[1] +
						dx*dx*voxelSize[2]*voxelSize[2];

				// consider only in circle around center
				if (distance2 > maxDistance2)
					continue;

				Point offset(dz, dy, dx);

				Neighbor neighbor;
				neighbor.offset = offset;
				neighbor.distance2 = distance2;
				neighbor.dindex = dindex;
				_neighbors.push_back(neighbor);

				std::cout << "\t" << dz << ", " << dy << ", " << dx << " at " << distance2 << " d-index " << dindex << std::endl;
			}
}

template<typename Dtype>
bool
SegEmbedLossLayer<Dtype>::isInside(Point u) {

	if (u.x >= 0 && u.x < _shape.x &&
	    u.y >= 0 && u.y < _shape.y &&
	    u.z >= 0 && u.z < _shape.z)
		return true;

	return false;
}

template<typename Dtype>
void
SegEmbedLossLayer<Dtype>::computeLossGradient(int_tp indexU, int_tp indexV, Dtype distance2) {

	Dtype embDistance2 = 0.0;
	for (int_tp k = 0; k < _embDimension; k++)
		embDistance2 += std::pow(
				_prediction[k*_embComponentOffset + indexU] -
				_prediction[k*_embComponentOffset + indexV], 2);

	bool same = (_gt[indexU] == _gt[indexV]);
	std::cout << "\t\thave embedded distance " << embDistance2 << std::endl;
	std::cout << "\t\thave labels " << _gt[indexU] << " and " << _gt[indexV] << std::endl;

	Dtype loss = 0;

	if (same) {

		std::cout << "\t\thave same label" << std::endl;

		// max(0, |e_U - e_V|^2 - alpha*|U-V|^2)
		loss = std::max((Dtype)0.0, embDistance2 - _alpha*distance2);

		std::cout << "\t\tloss = " << loss << std::endl;

		if (loss > 0) {

			for (int_tp k = 0; k < _embDimension; k++) {

				// dL/de_U = 2*(e_U - e_V)
				Dtype gradientUk =
					2*_prediction[k*_embComponentOffset + indexU] -
					2*_prediction[k*_embComponentOffset + indexV];

				// dL/de_V = -2*(e_U - e_V)
				Dtype gradientVk = -gradientUk;

				_gradients[k*_embComponentOffset + indexU] += gradientUk;
				_gradients[k*_embComponentOffset + indexV] += gradientVk;

				std::cout << "\t\tgradient on u, k=" << k << ": " << gradientUk  << std::endl;
				std::cout << "\t\tgradient on v, k=" << k << ": " << gradientVk  << std::endl;
			}
		}

	} else {

		std::cout << "\t\thave different label" << std::endl;

		// max(0, 4 - |e_U - e_V|^2 - alpha*|U-V|^2)
		// (4 is max squared distance between unit vectors)
		loss = std::max((Dtype)0.0, (Dtype)4.0 - embDistance2 - _alpha*distance2);

		std::cout << "\t\tloss = " << loss << std::endl;

		if (loss > 0) {

			for (int_tp k = 0; k < _embDimension; k++) {

				// dL/de_U = -2*(e_U - e_V)
				Dtype gradientUk =
					2*_prediction[k*_embComponentOffset + indexV] -
					2*_prediction[k*_embComponentOffset + indexU];

				// dL/de_V = 2*(e_U - e_V)
				Dtype gradientVk = -gradientUk;

				_gradients[k*_embComponentOffset + indexU] += gradientUk;
				_gradients[k*_embComponentOffset + indexV] += gradientVk;

				std::cout << "\t\tgradient on u, k=" << k << ": " << gradientUk  << std::endl;
				std::cout << "\t\tgradient on v, k=" << k << ": " << gradientVk  << std::endl;
			}
		}
	}

	_loss += loss;
}

template<typename Dtype>
void
SegEmbedLossLayer<Dtype>::accumulateLossGradient(Point u) {

	std::cout << "computing loss of " << u.z << ", " << u.y << ", " << u.x << std::endl;

	// for each neighbor
	for (const Neighbor& neighbor : _neighbors) {

		Point v = Point(
				u.z + neighbor.offset.z,
				u.y + neighbor.offset.y,
				u.x + neighbor.offset.x);

		std::cout << "\tto " << v.z << ", " << v.y << ", " << v.x << std::endl;

		if (!isInside(v)) {

			std::cout << "\t=> outside of volume" << std::endl;
			continue;
		}

		int_tp indexU = u.x + u.y*_shape.x + u.z*_shape.y*_shape.x;
		int_tp indexV = v.x + v.y*_shape.x + v.z*_shape.y*_shape.x;

		std::cout << "\tindices are " << indexU << " and " << indexV << std::endl;

		// compute loss and gradient
		computeLossGradient(indexU, indexV, neighbor.distance2);

		_numPairs++;
	}
}

template<typename Dtype>
void SegEmbedLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	// read parameters
	_alpha = bottom[PARAMS]->cpu_data()[ALPHA];
	Dtype maxDistance = bottom[PARAMS]->cpu_data()[MAX_DISTANCE];

	_prediction = bottom[PRED]->cpu_data();
	_gt = bottom[GT]->cpu_data();

	createNeighborhood(
			maxDistance*maxDistance,
			bottom[VOXEL_SIZE]->cpu_data());

	caffe_set(_dloss.count(), Dtype(0.0), _dloss.mutable_cpu_data());
	_loss = 0;
	_gradients = _dloss.mutable_cpu_data();
	_numPairs = 0;

	// for each voxel
	for (int_tp z = 0; z < _shape.z; z++)
		for (int_tp y = 0; y < _shape.y; y++)
			for (int_tp x = 0; x < _shape.x; x++)
				accumulateLossGradient(Point(z, y, x));

	// normalize loss
	top[0]->mutable_cpu_data()[0] = _loss/_numPairs;

	// normalize gradient
	for (int_tp i = 0; i < _dloss.count(); i++)
		_gradients[i] /= _numPairs;
}

template<typename Dtype>
void SegEmbedLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

	if (propagate_down[PRED]) {

		Dtype* bottom_diff = bottom[PRED]->mutable_cpu_diff();
		const Dtype* gradients = _dloss.cpu_data();

		// Clear the diff
		caffe_set(bottom[PRED]->count(), Dtype(0.0), bottom_diff);

#pragma omp parallel for
		for (int_tp i = 0; i < bottom[PRED]->count(); ++i) {
			bottom_diff[i] = gradients[i];
		}
	}
}

INSTANTIATE_CLASS(SegEmbedLossLayer);
REGISTER_LAYER_CLASS(SegEmbedLoss);

} // namespace caffe
