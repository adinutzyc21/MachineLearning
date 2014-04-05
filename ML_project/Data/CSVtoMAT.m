% this creates the data matrix
% cd ML_project

X = csvread('X.csv');
Y  = csvread('Y.csv');
aneurism = struct('X', X, 'Y', Y);

save('aneurism.mat', 'aneurism');