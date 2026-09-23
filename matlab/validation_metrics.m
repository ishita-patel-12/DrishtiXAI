% DrishtiXAI - MATLAB Performance & Validation Metrics Script (SIH26038)
% Calculates Confusion Matrix, Sensitivity, Specificity, F1 Score, and ROC AUROC.

function metrics = validation_metrics(y_true, y_pred, y_scores)
    % Multi-class confusion matrix
    cm = confusionmat(y_true, y_pred);
    disp('Confusion Matrix (0=No DR, 1=Mild, 2=Mod, 3=Sev, 4=PDR):');
    disp(cm);
    
    % Referable DR binary targets (Grade 0-1 vs Grade 2-4)
    ref_true = (y_true >= 2);
    ref_pred = (y_pred >= 2);
    
    tp = sum(ref_true & ref_pred);
    fp = sum(~ref_true & ref_pred);
    tn = sum(~ref_true & ~ref_pred);
    fn = sum(ref_true & ~ref_pred);
    
    sensitivity = tp / (tp + fn + 1e-9);
    specificity = tn / (tn + fp + 1e-9);
    precision = tp / (tp + fp + 1e-9);
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-9);
    
    metrics = struct();
    metrics.ConfusionMatrix = cm;
    metrics.Sensitivity = sensitivity;
    metrics.Specificity = specificity;
    metrics.Precision = precision;
    metrics.F1Score = f1_score;
    
    fprintf('\n--- REFERABLE DR VALIDATION METRICS ---\n');
    fprintf('Sensitivity (Recall): %.4f\n', sensitivity);
    fprintf('Specificity:          %.4f\n', specificity);
    fprintf('Precision:            %.4f\n', precision);
    fprintf('F1 Score:             %.4f\n', f1_score);
end
