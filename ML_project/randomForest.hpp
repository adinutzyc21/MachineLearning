#pragma once

#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <cstdlib>

using namespace std;

double rand_d() {
	//return a random double in the range [0,1)
	return double(rand())/(RAND_MAX-1);
}

size_t uniform_size_t(size_t max) {
	double r = rand_d()*(max-1);
	return max < r ? max : size_t(r);
}

//a balanced binary decision tree
template<typename V, typename L>
struct decisionTree {
	//depth of the tree
	size_t depth;
	//heaps to represent tree: size = 2^(depth-1) - 1
	//what features do we evaluate on?
	size_t *features;
	//what values do we compare the features to?
	V *values;
	//what are the outputs? (size = 2^(depth-1))
	L *leaves;
	//default constructor zeros the pointers
	decisionTree() : depth(0), features(0), values(0), leaves(0) {}
	decisionTree(size_t depth, size_t *features, V* values, L *leaves) :
		depth(depth), features(features), values(values), leaves(leaves) {}
};
template<typename V, typename L>
const decisionTree<V,L> makeConstTree(const size_t depth, const size_t *features, const V* values, const L* leaves) {
    return decisionTree<V,L>(depth, const_cast<size_t*>(features),
                             const_cast<V*>(values), const_cast<L*>(leaves));
}

//evaluate a decision tree
template<typename V, typename L>
L evaluateTree(const V *value, const decisionTree<V, L> &tree) {
	size_t pos = 0;
	for(size_t d = 0; d < tree.depth-1; ++d) {
		size_t f = tree.features[pos];
		if(value[f] < tree.values[pos])
			pos = pos*2+1;
		else
			pos = pos*2+2;
	}
	size_t nnodes = (1 << (tree.depth-1))-1;
	return tree.leaves[pos - nnodes];
}

//point with class
template<typename V>
struct cpoint {
	const V *v;
	unsigned int c;
	cpoint() : v(), c() {}
	cpoint(const V *v, unsigned int c) : v(v), c(c) {}
	cpoint(const cpoint &cp) : v(cp.v), c(cp.c) {}
	const V& operator[](size_t i) const { return v[i]; }
};

//loss functions computed using histogram of classes:
double entropy(const vector<unsigned int> &counts, size_t npoints) {
	double h = 0;
    for(size_t i = 0; i < counts.size(); ++i) {
		double cn = double(counts[i])/npoints;
		if(cn > 0) h += cn*log(cn);
	}
	return -h/log(2.0);
}

double gini(const vector<unsigned int> &counts, size_t npoints) {
    double g = 0;
    for(size_t i = 0; i < counts.size(); ++i) {
        g += counts[i]*(npoints-counts[i]);
    }
    return g/(npoints*npoints);
}

vector<size_t> rperm(const size_t n, const size_t range) {
    //draw n elements from [0,range)
    //using rejection sampling:
    set<size_t> s;
    for(size_t i = 0; i < n; ++i) {
        size_t r;
        do {
            r = uniform_size_t(range-1);
        } while(s.count(r));
        s.insert(r);
    }
    return vector<size_t>(s.begin(),s.end());
}

template<typename V, typename L>
vector<bool> trainBaggedClassifierTree(const V *x, const L *y, const size_t npoints,const size_t ndims,
                               size_t *features, V *values, L *leaves, const size_t depth,
							   const size_t nranddims) {
	//make the decision tree structure
	decisionTree<V,L> dt(depth,features,values,leaves);
	
	//resample training points and
	//replace labels with class index
	map<L, int> label2class;
	vector<bool> outofbag(npoints, true);
	vector<L> class2label;
	vector<cpoint<V> > points(npoints);
	vector<unsigned int> counts; //how many of each class do we have?
	unsigned int c, next_c = 0;
	for(size_t ix = 0; ix < npoints; ++ix) {
		size_t i = uniform_size_t(npoints-1); //sample a point from the training set
		if(!label2class.count(y[i])) {
			label2class[y[i]] = next_c++;
			class2label.push_back(y[i]);
			counts.push_back(0);
		}
		outofbag[i] = false;
		points[ix].v = x + i*ndims;
		c = label2class[y[i]];
		points[ix].c = c;
		counts[c]++;
	}
	//these will define the range of points we're working on:
	size_t begin = 0;
	size_t end = points.size();
	//recursively train the tree:
	_trainClassifierTree(points, ndims, class2label, dt, counts, begin, end, 0, depth, nranddims, entropy);

	return outofbag;
}

//train a decision tree
template<typename V, typename L>
void trainClassifierTree(const V *x, const L *y, const size_t npoints, const size_t ndims,
                         size_t *features, V *values, L *leaves, const size_t depth,
                         const size_t nranddims) {

	//make the tree structure
	decisionTree<V,L> dt(depth, features, values, leaves);
	
    //replace labels with class index
	map<L, int> label2class;
	vector<L> class2label;
	vector<cpoint<V> > points(npoints);
	vector<unsigned int> counts; //how many of each class do we have?
	unsigned int c, next_c = 0;
	for(size_t i = 0; i < npoints; ++i) {
		if(!label2class.count(y[i])) {
			label2class[y[i]] = next_c++;
			class2label.push_back(y[i]);
			counts.push_back(0);
		}
		points[i].v = x + i*ndims;
		c = label2class[y[i]];
		points[i].c = c;
		counts[c]++;
	}
	//these will define the range of points we're working on:
	size_t begin = 0;
	size_t end = points.size();
	//recursively train the tree:
	_trainClassifierTree(points, ndims, class2label, dt, counts, begin, end, 0, depth, nranddims, entropy);
}

//compare to random-access things on a particular dimension
struct on_dim {
	size_t d;
	on_dim(size_t d) : d(d) {}
	template<typename RandomAccess>
	bool operator()(const RandomAccess& a, const RandomAccess& b) const { return a[d] < b[d];}
};

template<typename T>
struct greater_than {
	const T v;
	greater_than(const T &v) : v(v) {}
	bool operator()(const T& v1) const { return v1 > v; }
};

template<typename V, typename L, typename loss_t>
void _trainClassifierTree(vector<cpoint<V> > &points, const size_t ndims,
		const vector<L> &class2label, decisionTree<V,L> &dt,
		const vector<unsigned int> &counts, const size_t begin, const size_t end,
		const size_t pos, const size_t max_depth, const size_t nranddims,
        const loss_t &loss) {
	//we're operating on the points in range begin to end
	//sort them on each dimension, compute loss for each possible split
	//keep the best dimension-split pair and recurse again
	//stop recursing when we hit the maximum depth or if all one class
	size_t nnodes = (1 << (max_depth-1))-1;
	if(pos >= nnodes) { //we've hit the max-depth
		//compute the leaf's value instead of splitting the tree more
		//what's the offset of the most frequent class in the leaf's set?
		size_t c = max_element(counts.begin(), counts.end()) - counts.begin();
		//what label does it correspond to?
		dt.leaves[pos - nnodes] = class2label[c];
		return;
	}

	if(count_if(counts.begin(),counts.end(),greater_than<double>(0)) == 1) { //there's only 1 class
		//set values for tree (they won't really do anything 'cause it's all 1 class)
		dt.features[pos] = 0;
		dt.values[pos] = points[begin][0];
		//recurse, just so that it'll get down to the leaves and do the right value.
		_trainClassifierTree(points, ndims, class2label, dt, counts, begin, end, 2*pos+1, max_depth, nranddims, loss);
		_trainClassifierTree(points, ndims, class2label, dt, counts, begin, end, 2*pos+2, max_depth, nranddims, loss);
		return;
	}
	//record best loss and info that goes along with it:
	double best_loss = numeric_limits<double>::infinity();
	size_t best_feature = 0;
	vector<cpoint<V> > best_points(end-begin);
	size_t best_split = 0;
	vector<unsigned int> best_left(counts.size(),0);
	vector<unsigned int> best_right(counts.size(),0);
	//what featurs are we going to look at?
    vector<size_t> features = rperm(nranddims, ndims);
	//iterate over dimensions:
	for(size_t id = 0; id < nranddims; ++id) {
        size_t d = features[id];
		//initialize class-frequency counts for each side of the split
		vector<unsigned int> left_counts(counts.size(),0); //same size, but zeros
		vector<unsigned int> right_counts(counts); //copy of initial counts
		//sort the points along the dimension
		sort(points.begin()+begin,points.begin()+end,on_dim(d));
		//iterate over splits
		for(size_t split = begin+1; split < end; ++split) {
			//update counts:
			left_counts[points[split-1].c]++;
			right_counts[points[split-1].c]--;
			//compute entropy of both sides
			double h = loss(left_counts, split-begin) + loss(right_counts, end-split);
			//is this split better than the best so far?
			//new entropy should be minimized in order to maximize information gain
			// (info gain) = (prev entropy) - sum(new entropies)
			if(h < best_loss) {
				//update record of best things:
				best_loss = h;
				best_split = split;
				best_feature = d;
				copy(points.begin()+begin, points.begin()+end, best_points.begin());
				copy(left_counts.begin(), left_counts.end(), best_left.begin());
				copy(right_counts.begin(), right_counts.end(), best_right.begin());
			}
		}
	}
	//copy back the best sort
	copy(best_points.begin(),best_points.end(),points.begin()+begin);
	//update tree to match this split:
	dt.features[pos] = best_feature;
	dt.values[pos] = (points[best_split][best_feature] + points[best_split-1][best_feature])/2;
	//recurse:
	_trainClassifierTree(points, ndims, class2label, dt, best_left, begin, best_split, 2*pos+1, max_depth, nranddims, loss);
	_trainClassifierTree(points, ndims, class2label, dt, best_right, best_split, end, 2*pos+2, max_depth, nranddims, loss);
}
