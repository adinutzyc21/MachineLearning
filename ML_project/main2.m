
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
Y_predict = mode(Y_forest,1);
%compute the training error -- number of misclassifications
Y_predict_err = Y_predict~=Y;

trainerror = sum(Y_predict_err) / size(X,2);

% compute oob error
% forest.oob

Y_ooberr = (repmat(Y, ntrees, 1) .* double(forest.oob')) ~= (Y_forest .* double(forest.oob'));

ooberror = sum(sum(Y_ooberr,1)./sum(forest.oob,2)')/size(X,2);

classes = unique(Y);
ooberror_byclass = zeros(size(classes));
ooberror_byclass2 = zeros(length(classes),length(classes));
trainerror_by_class = zeros(length(classes),length(classes));
for ic = 1:length(classes)
    %what class are we looking at?
    Y_mask = Y == classes(ic);
    ooberror_byclass(ic) = sum(sum(Y_ooberr(:,Y_mask),1)./sum(forest.oob(Y_mask,:),2)') / sum(Y_mask);
    for jc = 1:length(classes)
        %what fraction of them were classified as classes(jc)?
        %in whole training set (including out-of-bag):
        trainerror_by_class(ic, jc) = sum(Y_predict(Y_mask) == classes(jc)) / sum(Y_mask);
        %in out-of-bag set:
        
        %fraction of out of bag Ys that are supposed to equal classes(ic) but equal classes(jc)
        ooberror_byclass2(ic,jc) = sum(sum(Y_forest(:,Y_mask) .* double(forest.oob(Y_mask,:)') == classes(jc),1)./sum(forest.oob(Y_mask,:),2)')/sum(Y_mask); 
    end
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