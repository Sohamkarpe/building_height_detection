%% Visualization - Building Height Estimation
%  Author: Soham Ashok Karpe, M. Eng.

figure('Name', 'Building Height Estimation Results', 'NumberTitle', 'off');

%% 1. Predicted vs Actual
subplot(2,2,1);
scatter(y_test, y_pred, 50, '#00d4ff', 'filled', 'MarkerEdgeColor', '#0099bb');
hold on;
lims = [min([y_test; y_pred]), max([y_test; y_pred])];
plot(lims, lims, 'r--', 'LineWidth', 1.5);
xlabel('Actual Height (m)'); ylabel('Predicted Height (m)');
title('Predicted vs Actual'); grid on;
legend('Predictions', 'Ideal', 'Location', 'northwest');

%% 2. Residuals
subplot(2,2,2);
residuals = y_test - y_pred;
histogram(residuals, 20, 'FaceColor', '#7b2dff', 'EdgeColor', 'white');
xlabel('Residual (m)'); ylabel('Count');
title(sprintf('Residuals  MAE=%.2fm', mae));
xline(0, 'r--', 'LineWidth', 1.5); grid on;

%% 3. Height distribution
subplot(2,2,3);
histogram(y, 20, 'FaceColor', '#00ff88', 'EdgeColor', 'white', 'Normalization', 'probability');
xlabel('Height (m)'); ylabel('Probability');
title('Height Distribution (Ground Truth)'); grid on;

%% 4. Error per height range
subplot(2,2,4);
edges = 0:10:max(y_test)+10;
centers = edges(1:end-1) + 5;
errors_per_bin = zeros(1, length(centers));
for i = 1:length(centers)
    mask = y_test >= edges(i) & y_test < edges(i+1);
    if sum(mask) > 0
        errors_per_bin(i) = mean(abs(residuals(mask)));
    end
end
bar(centers, errors_per_bin, 'FaceColor', '#ff8c00', 'EdgeColor', 'white');
xlabel('Height Range (m)'); ylabel('MAE (m)');
title('Error by Height Range'); grid on;

sgtitle('Building Height Estimation — Soham Karpe, DIT Campus Cham', ...
    'FontSize', 12, 'FontWeight', 'bold');
