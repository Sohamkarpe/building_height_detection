%% Building Height Detection - Feature Extraction
%  Author: Soham Ashok Karpe, M. Eng.
%  Institution: Technische Hochschule Deggendorf (DIT), Campus Cham
%  Period: Mar 2023 - Jul 2023 | DIT HiWi Project

clc; clear; close all;

%% Configuration
IMAGE_DIR = '../data/images/';
OUTPUT_FILE = '../data/features.csv';
TARGET_SIZE = [256, 256];

fprintf('Building Height Detection - Feature Extraction\n');
fprintf('Author: Soham Karpe, M.Eng. | DIT Campus Cham\n');
fprintf('================================================\n\n');

%% Load images
image_files = dir(fullfile(IMAGE_DIR, '*.jpg'));
image_files = [image_files; dir(fullfile(IMAGE_DIR, '*.png'))];

n = length(image_files);
fprintf('Found %d images in: %s\n\n', n, IMAGE_DIR);

feature_matrix = [];
filenames = {};

for i = 1:n
    img_path = fullfile(IMAGE_DIR, image_files(i).name);
    img = imread(img_path);

    % Convert to grayscale if needed
    if size(img, 3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end

    % Resize
    gray = imresize(gray, TARGET_SIZE);
    img_resized = imresize(img, TARGET_SIZE);

    fprintf('Processing [%d/%d]: %s\n', i, n, image_files(i).name);

    % Extract features
    f = extract_features(gray, img_resized);
    feature_matrix = [feature_matrix; f];
    filenames{end+1} = image_files(i).name;
end

%% Save features
fprintf('\nSaving features to: %s\n', OUTPUT_FILE);
T = array2table(feature_matrix);
T.filename = filenames';
writetable(T, OUTPUT_FILE);
fprintf('Done! Feature matrix size: %d x %d\n', size(feature_matrix));


%% Feature Extraction Function
function features = extract_features(gray, img_rgb)
    % 1. Edge features
    edges = edge(gray, 'Canny', [0.1, 0.3]);
    edge_density = sum(edges(:)) / numel(edges);

    [Gx, Gy] = imgradientxy(gray);
    Gmag = sqrt(Gx.^2 + Gy.^2);
    grad_mean = mean(Gmag(:));
    grad_std  = std(Gmag(:));

    % 2. Shadow analysis (dark regions)
    shadow_mask = gray < 80;
    shadow_ratio = sum(shadow_mask(:)) / numel(shadow_mask);

    % 3. Texture (GLCM)
    glcm = graycomatrix(gray, 'NumLevels', 32, 'Offset', [0 1]);
    stats = graycoprops(glcm, {'contrast','correlation','energy','homogeneity'});
    texture_feats = [stats.Contrast, stats.Correlation, ...
                     stats.Energy, stats.Homogeneity];

    % 4. Shape / contour
    bw = imbinarize(gray, 'adaptive');
    props = regionprops(bw, 'Area', 'BoundingBox', 'Solidity');
    if ~isempty(props)
        areas = [props.Area];
        [~, idx] = max(areas);
        bb = props(idx).BoundingBox;
        aspect_ratio = bb(4) / max(bb(3), 1);
        solidity = props(idx).Solidity;
        rel_height = bb(4) / size(gray, 1);
    else
        aspect_ratio = 0;
        solidity = 0;
        rel_height = 0;
    end

    % 5. Color histogram (HSV)
    if size(img_rgb, 3) == 3
        hsv = rgb2hsv(double(img_rgb) / 255);
        h_hist = histcounts(hsv(:,:,1), 18, 'Normalization', 'probability');
        s_hist = histcounts(hsv(:,:,2), 8, 'Normalization', 'probability');
        color_feats = [h_hist, s_hist];
    else
        color_feats = zeros(1, 26);
    end

    % Concatenate all features
    features = [edge_density, grad_mean, grad_std, ...
                shadow_ratio, aspect_ratio, solidity, rel_height, ...
                texture_feats, color_feats];
end
