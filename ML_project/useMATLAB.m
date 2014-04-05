% cd to ML_project
addpath Data;
load('aneurism.mat');
X = aneurism.X;
Y = aneurism.Y;

n_trees = 1000;
tree_depth = 1;
matlabpool('open', 4);
TreeObject = TreeBagger(n_trees,X',Y','Method','classification','OOBPred','on','OOBVarImp','on','NVarToSample','all','MinLeaf',tree_depth,'Options',statset('UseParallel','always'));
matlabpool('close');
oob_indices = TreeObject.OOBIndices;

for treeIndex = 1:n_trees
    Y_tree = predict(TreeObject,X','trees',[treeIndex]);
    
    Y_forest(treeIndex, :) = str2double(Y_tree);
end

Y_predict = mode(Y_forest,1);
Y_ooberr = (repmat(Y, n_trees, 1) .* double(oob_indices')) ~= (Y_forest .* double(oob_indices'));
ooberror = sum(sum(Y_ooberr,1)./sum(oob_indices,2)')/size(X,2);

classes = unique(Y);
ooberror_byclass = zeros(size(classes));
trainerror_byclass = zeros(size(classes));
ooberror_byclass2 = zeros(length(classes),length(classes));
trainerror_by_class2 = zeros(length(classes),length(classes));
for ic = 1:length(classes)
    Y_mask = Y == classes(ic);
    
    trainerror_by_class(ic) = sum(mode(Y_forest(:,Y_mask),1) ~= Y(:,Y_mask)) / size(X(:,Y_mask),2);
    ooberror_byclass(ic) = sum(sum(Y_ooberr(:,Y_mask),1)./sum(oob_indices(Y_mask,:),2)') / sum(Y_mask);
    
    for jc = 1:length(classes)
        trainerror_by_class2(ic, jc) = sum(Y_predict(Y_mask) == classes(jc)) / sum(Y_mask);
        ooberror_byclass2(ic,jc) = sum(sum(Y_forest(:,Y_mask) .* double(oob_indices(Y_mask,:)') == classes(jc),1)./sum(oob_indices(Y_mask,:),2)')/sum(Y_mask); 
    end
    
    disp(['Type ' num2str(ic) ' - Training: ' num2str(trainerror_by_class(ic)) ' OOB: ' num2str(ooberror_byclass(ic))]);
end

disp(['OOB Error: ' num2str(oobError(TreeObject, 'Mode', 'ensemble'))]);

%bar(TreeObject.OOBPermutedVarDeltaError)