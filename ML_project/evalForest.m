function [ Y ] = evalForest( X, forest )
%EVALFOREST Evaluate a forest of decision trees
%   INPUTS:
%     X: matrix [num_features, num_samples]
%       The points to evaluate.
%     forest: scalar struct
%       A struct representing a forest of binary decision trees. Each
%       tree is represented by several heap data structures.
%     forest.depth: scalar
%       The depth (including leaves) of all the trees in the forest. The
%       number of nodes (num_nodes below) is 2^(forest.depth-1)-1. The
%       number of leaves (num_leaves below) is 2^(forest.depth-1).
%     forest.features: matrix [num_nodes, num_trees]
%       A matrix where each row is a heap of 'uint64's representing which 
%       feature of x to decide on at each node of the tree.
%     forest.values: matrix [num_nodes, num_trees]
%       A matrix where each row is a heap of 'double's that the selected
%       feature of x is compared to at each node of the tree. If the 
%       selected feature is less than the node's value, the left branch 
%       (position 2*i in the heap) of the tree is selected; otherwise the 
%       right branch (position 2*i+1 in the heap) is selected.
%     forest.leaves: matrix [num_leaves, num_trees]
%       A matrix where each row correspond to the leaves of a decision
%       tree. For a node i in the deepest level of the tree, the left child
%       will be leaf (2*i - num_nodes) and the right child will be leaf
%       (2*i + 1 - num_nodes).
%   OUTPUTS:
%     Y: matrix [num_trees, num_samples]
%       The results of evaluating each decision tree.


%check inputs for correct data types:
assert(isdouble(X), 'X must be of type double')
assert(isdouble(forest.values), 'forest.values must be of type double')
assert(isdouble(forest.leaves), 'forest.leaves must be of type double')
assert(strcmp(class(forest.features),'uint64'),'forest.features must be of type uint64')
%check inputs are correct dimensionality:
assert(ndims(X) == 2, 'X must be 1D or 2D')
assert(isscalar(forest.depth), 'forest.depth must be scalar')
assert(ndims(forest.features) == 2, 'forest.features must be 1D or 2D')
assert(ndims(forest.values) == 2, 'forest.values must be 1D or 2D')
assert(ndims(forest.leaves) == 2, 'forest.leaves must be 1D or 2D')
%make sure sizes match requirements:
fs = size(forest.features);
vs = size(forest.values);
assert(all(fs == vs), 'forest.features and forest.values must have same size')
ls = size(forest.leaves);
assert(fs(2)==ls(2), 'size(forest.values,2) must equal size(forest.leaves,2)')
assert(fs(1)== 2^(forest.depth-1)-1, 'size(forest.values,1) must equal 2^(forest.depth-1)-1')
assert(ls(1)== 2^(forest.depth-1), 'size(forest.leaves,1) must equal 2^(forest.depth-1)')
%evaluate forest:
Y = evalForest_mex(X, forest.features, forest.values, forest.leaves, uint64(forest.depth));
end

function [d] = isdouble(X)
    d = isreal(X) && isa(X,'double');
end

