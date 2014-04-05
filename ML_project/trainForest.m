function [ forest ] = trainForest(X,Y,depth,ntrees,nrdims,bag,seed)
%TRAINFOREST Trains a random forest
%   INPUT:
%     X: double matrix [num_features, num_samples]
%     Y: double matrix [num_samples]
%     depth: scalar int (tree depth, including leaves)
%     ntrees: scalar int (number of trees in forest)
%     nrdims: scalar int (number of features to assess at each level)
%     bag: optional logical (train the trees with random subsamples of X,Y)
%     seed: optional scalar int (reseed the random number generator)

%check types:
assert(isdouble(X),'X must be of type double')
assert(isdouble(Y),'Y must be of type double')
%check dimensionality:
assert(ndims(X)==2,'X must be 1D or 2D')
assert(ndims(Y)==2,'Y must be 1D')
assert(isscalar(depth),'depth must be scalar')
assert(isscalar(nrdims),'nrdims must be scalar')
%make sure X & Y's dimensions match
sx = size(X);
sy = size(Y);
assert(sx(2)==sy(1)||sx(2)==sy(2),'Size mismatch between X and Y')
assert(sy(1)==1||sy(2)==1,'Y must be 1D')
%cast everything to the right type
depth = uint32(depth);
ntrees = uint32(ntrees);
nrdims = uint32(nrdims);
if nargin >= 6 %we're using the bag value
    assert(isscalar(bag),'bag must be scalar')
    bag = uint8(bag ~= 0);
else
    bag = uint8(0); %by default, don't bag
end
oob = 0;
if nargin == 7 %we're using the seed value
    assert(isscalar(seed),'seed must be scalar')
    seed = uint64(seed);
    if bag
        [features,values,leaves,oob] = trainForest_mex(X,Y,depth,ntrees,nrdims,seed);
    else
        [features,values,leaves] = trainForest_mex(X,Y,depth,ntrees,nrdims,seed);
    end
else
    if bag
        [features,values,leaves,oob] = trainForest_mex(X,Y,depth,ntrees,nrdims);
    else
        [features,values,leaves] = trainForest_mex(X,Y,depth,ntrees,nrdims);
    end
end
forest.depth = depth;
forest.features = features;
forest.values = values;
forest.leaves = leaves;
forest.oob = oob;
end

function [d] = isdouble(X)
    d = isreal(X) && isa(X,'double');
end