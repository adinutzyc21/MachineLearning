
%load matlab's ionosphere dataset

addpath Data
load('aneurism.mat');
X = aneurism.X;
Y = aneurism.Y;
clear aneurism

depth = 15; %depth, including leaves
ntrees = 5000; %number of trees
nrdims = floor(sqrt(size(X,1))); %number of random dimensions to train on at each level
bag = 1; %optional--0 for no bagging, 1 for bagging
seed = 15; %optional--seed the random number generator
disp 'Training Random Forest...'
tic
forest = trainForest(X,Y,depth,ntrees,nrdims,bag,seed);
toc
%evaluate the forest at all of the points we trained on
Y_forest = evalForest(X,forest);

%compute the training error -- number of misclassifications
Y_forest_err = mode(Y_forest,1)~=Y;

trainerror = sum(Y_forest_err) / size(X,2);

% compute oob error
% forest.oob

Y_err = (repmat(Y, ntrees, 1) .* double(forest.oob')) ~= (Y_forest .* double(forest.oob'));

ooberror = sum(sum(Y_err,1)./sum(forest.oob,2)')/size(X,2);

classes = unique(Y);
ooberror_byclass = zeros(size(classes));
for ic = 1:length(classes)
    Y_mask = Y == classes(ic);
    
    trainerror_by_class(ic) = sum(mode(Y_forest(:,Y_mask),1) ~= Y(:,Y_mask)) / size(X(:,Y_mask),2);
    ooberror_byclass(ic) = sum(sum(Y_err(:,Y_mask),1)./sum(forest.oob(Y_mask,:),2)')/size(X(:,Y_mask),2);
end
%using Matlab's TreeBagger:
% disp 'Training random forest with Matlab TreeBagger...'
% tic
% B = TreeBagger(ntrees,X',Y');
% toc
% Y_treebagger = predict(B,X')';
% %convert back to numbers....
% Y_treebagger = arrayfun(@(c) str2num(c{1}),Y_treebagger);
% Y_treebagger_err = Y_treebagger.*Y ~= 1;

%%
% find(Y_forest_err~=Y_treebagger_err)