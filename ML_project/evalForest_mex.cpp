/* decisionTree.c
 * fast decision tree algorithms
 *
 *
 */

#include "mex.h"
#include "randomForest.hpp"

void evalForest(const double *x, const mwSize nsamples, const mwSize ndims,
                const size_t *features, const double *values, const double *leaves,
                const mwSize ntrees, const size_t depth,
                double *y) {
    //x is 2D: (nsamples)x(ndims)  (row-major order)
    //features, values are 2D: (ntrees)x(2^(depth-1)-1)
    //leaves is 2D: (ntrees)x(2^(depth-1))
    //y is 2D: (nsamples)x(ntrees)
    
    //compute (2^nlevels - 1
    mwSize nodes_len = (1 << (depth-1))-1;
    mwSize leaves_len = (1 << (depth-1));
    //loop over all the samples:
    for(mwSize s = 0; s < nsamples; ++s) {
        
        //index into x and y for the current sample:
        const double *xs = x + s*ndims;
        double *ys = y + s*ntrees;
        
        //loop over all the trees:
        for(mwSize t = 0; t < ntrees; ++t) {
            
            //index into dims and vals for the current tree:
            const size_t *ft = features + t*nodes_len;
            const double *vt = values + t*nodes_len;
            const double *lt = leaves + t*leaves_len;
            
            
            //evaluate the current tree:
            ys[t] = evaluateTree(xs, makeConstTree(depth,ft,vt,lt));
            
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    //usage: [y] = evalTree(x, features, values, leaves, depth)
    //  x is 2D: (nsamples)x(ndims)  (row-major order)
    //  depth is scalar
    //  features is 2D: (ntrees)x(2^(depth-1) - 1)
    //  values is 2D: (ntrees)x(2^(depth-1) - 1)
    //  leaves is 2D: (ntrees)x(2^(depth-1))
    //  y is 2D: (nsamples)x(ntrees)
    
    //assume that we've checked the arguments in the calling .m file
    //get pointer to x array:
    const mxArray *x_arr = prhs[0];
    //get x array's size:
    const mwSize *sx = mxGetDimensions(x_arr);
    mwSize ndims = sx[0]; //matlab reports dimensions in column-major order :(
    mwSize nsamples = sx[1];
    
    //get pointers to decision tree stuff:
    const mxArray *features_arr = prhs[1];
    const mxArray *values_arr = prhs[2];
    const mxArray *leaves_arr = prhs[3];
    const mxArray *depth_arr = prhs[4];
    //get depth of trees
    size_t depth = *(size_t*)(mxGetPr(depth_arr));
    //get number of trees
    sx = mxGetDimensions(features_arr);
    mwSize ntrees = sx[1]; //matlab reports dimensions in column-major order :(
    
    //make output array
    mxArray* y_arr = plhs[0] = mxCreateDoubleMatrix(ntrees, nsamples, mxREAL);
    
    //evaluate trees:
    evalForest(mxGetPr(x_arr), nsamples, ndims,
               (size_t*)mxGetData(features_arr), mxGetPr(values_arr), mxGetPr(leaves_arr),
               ntrees, depth,
               mxGetPr(y_arr));
}

