%% Building Height Estimation - MATLAB
%  Author: Soham Ashok Karpe, M. Eng.
%  Institution: Technische Hochschule Deggendorf (DIT)

clc; clear; close all;

%% Load features and labels
fprintf('Loading features and ground truth labels...\n');
T = readtable('../data/features.csv');
labels = readtable('../data/labels.csv');  % columns: filename, height_m

% Align features with labels
[~, ia, ib] = intersect(T.filename, labels.filename);
X = table2array(T(ia, 1:end-1));
y = labels.height_m(ib);

fprintf('Dataset: %d samples, %d features\n', size(X,1), size(X,2));

%% Split train/test
cv = cvpartition(length(y), 'HoldOut', 0.2);
X_train = X(cv.training, :);
y_train = y(cv.training);
X_test  = X(cv.test, :);
y_test  = y(cv.test);

%% Normalize features
mu = mean(X_train);
sigma = std(X_train) + 1e-8;
X_train_norm = (X_train - mu) ./ sigma;
X_test_norm  = (X_test  - mu) ./ sigma;

%% Train model (Regression Tree Ensemble)
fprintf('\nTraining Ensemble of Regression Trees...\n');
model = fitensemble(X_train_norm, y_train, 'Bag', 200, ...
    'Tree', 'Type', 'regression');

%% Evaluate
y_pred = predict(model, X_test_norm);

mae  = mean(abs(y_test - y_pred));
rmse = sqrt(mean((y_test - y_pred).^2));
ss_res = sum((y_test - y_pred).^2);
ss_tot = sum((y_test - mean(y_test)).^2);
r2 = 1 - ss_res / ss_tot;

fprintf('\n=== Results ===\n');
fprintf('MAE:  %.3f m\n', mae);
fprintf('RMSE: %.3f m\n', rmse);
fprintf('R²:   %.3f\n', r2);

%% Save model
save('../models/height_model.mat', 'model', 'mu', 'sigma');
fprintf('\nModel saved to: models/height_model.mat\n');

%% Run visualization
run visualization.m
