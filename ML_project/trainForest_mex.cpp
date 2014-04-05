#include <cstdlib>
#include <new>
#include "mex.h"

using namespace std;

//override new and delete so that stl libraries inside randomForest.hpp
// will use mxMalloc, mxFree

/*
void *operator new (size_t size, const nothrow_t &nothrow_constant) throw() {
    return mxMalloc((mwSize)size);
}
void *operator new (size_t size) throw (bad_alloc) {
    void *p = mxMalloc((mwSize)size);
    if(!p) bad_alloc();
    return p;
}

void operator delete (void *ptr) throw() {
    mxFree(ptr);
}
void operator delete (void *ptr, const nothrow_t &nothrow_constant) throw() {
    mxFree(ptr);
}
*/

#include "randomForest.hpp"

void trainForest(const double *x, const double *y, const mwSize nsamples, const mwSize ndims,
                 size_t *features, double *values, double *leaves,
                 const size_t ntrees, const size_t depth,
                 const size_t nrdims) {
    //x is (nsamples)x(ndims) (row-major)
    //y is (nsamples)
    //features, values are (ntrees)x(2^(depth-1)-1)
    //leaves is (ntrees)x(2^(depth-1))
    //nranddims is # of random features to use
    size_t nleaves = (1 << (depth-1));
    size_t nnodes = nleaves-1;
    
    for(size_t t = 0; t < ntrees; ++t) {
        size_t *ft = features + t*nnodes;
        double *vt = values + t*nnodes;
        double *lt = leaves + t*nleaves;
		trainClassifierTree(x, y, nsamples, ndims, ft, vt, lt, depth, nrdims);
    }
}

void trainBaggedForest(const double *x, const double *y, const mwSize nsamples, const mwSize ndims,
					size_t *features, double *values, double *leaves, char *oob,
					const size_t ntrees, const size_t depth,
					const size_t nrdims) {
    //x is (nsamples)x(ndims) (row-major)
    //y is (nsamples)
    //features, values are (ntrees)x(2^(depth-1)-1)
    //leaves is (ntrees)x(2^(depth-1))
    //nranddims is # of random features to use
    size_t nleaves = (1 << (depth-1));
    size_t nnodes = nleaves-1;
    
    for(size_t t = 0; t < ntrees; ++t) {
        size_t *ft = features + t*nnodes;
        double *vt = values + t*nnodes;
        double *lt = leaves + t*nleaves;

		vector<bool> oob_vec = trainBaggedClassifierTree(x, y, nsamples, ndims, ft, vt, lt, depth, nrdims);
		copy(oob_vec.begin(), oob_vec.end(), oob + t*nsamples);
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    //usage: features,values,leaves[,oob] = trainForest(x,y,depth,ntrees,nrdims[,seed])
    // x is (nsamples)x(ndims) (row-major)
    // y is (nsamples)
    // depth is uint32 > 1
    // ntrees is uint32 > 1
    // nrdims is uint32 > 1, < ndims (preferably much less)
    // seed is optional, unsigned int

    // features, values are (ntrees)x(2^(depth-1)-1)
    // leaves is (ntrees)x(2^(depth-1))
    
    //assume that the caller did all the argument checking
    const mxArray *x_arr = prhs[0];
    const mxArray *y_arr = prhs[1];
    const mxArray *d_arr = prhs[2];
    const mxArray *nt_arr = prhs[3];
    const mxArray *nd_arr = prhs[4];

    if(nrhs == 6) //re-seed the random number generator
        srand(*(unsigned int*)(mxGetData(prhs[5])));

    //get sizes:
    const mwSize *sx = mxGetDimensions(x_arr);
    mwSize ndims = sx[0]; //matlab reports dimensions in column-major order
    mwSize nsamples = sx[1];
    
    //dereference scalar data:
    mwSize depth = *(mwSize*)(mxGetData(d_arr));
    mwSize ntrees = *(mwSize*)(mxGetData(nt_arr));
    mwSize nrdims = *(mwSize*)(mxGetData(nd_arr));
    
    mwSize num_leaves = (1 << (depth-1));
    mwSize num_nodes = num_leaves-1;
    
    //make output arrays
    mxArray *features_arr = plhs[0] = mxCreateNumericMatrix(num_nodes,ntrees,mxUINT64_CLASS,mxREAL);
    mxArray *values_arr = plhs[1] = mxCreateNumericMatrix(num_nodes,ntrees,mxDOUBLE_CLASS,mxREAL);
    mxArray *leaves_arr = plhs[2] = mxCreateNumericMatrix(num_leaves,ntrees,mxDOUBLE_CLASS,mxREAL);
    
	if(nlhs == 4) {
		//bag stuff
		mxArray *oob_arr = plhs[3] = mxCreateNumericMatrix(nsamples,ntrees,mxUINT8_CLASS,mxREAL);
		trainBaggedForest(mxGetPr(x_arr), mxGetPr(y_arr), nsamples, ndims,
					(size_t*)(mxGetData(features_arr)), mxGetPr(values_arr),
					mxGetPr(leaves_arr), (char*)(mxGetData(oob_arr)), ntrees, depth, nrdims);
	} else {
		//no bagging
		trainForest(mxGetPr(x_arr), mxGetPr(y_arr), nsamples, ndims,
					(size_t*)(mxGetData(features_arr)), mxGetPr(values_arr),
					mxGetPr(leaves_arr), ntrees, depth, nrdims);
	}
}
