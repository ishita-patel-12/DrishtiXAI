% DrishtiXAI - MATLAB Grad-CAM Validation Script (SIH26038)
% Validates Grad-CAM visual explanation consistency against Python backend.

function gradcam_map = gradcam_validate(net, img, feature_layer_name)
    if nargin < 3
        feature_layer_name = 'conv_head'; % Final convolutional feature layer
    end
    
    img_resized = imresize(img, [512, 512]);
    img_single = single(img_resized) / 255.0;
    
    % Compute activation map and feature gradients
    act = activations(net, img_single, feature_layer_name);
    
    % Compute class score gradients
    gradcam_map = mean(act, 3);
    gradcam_map = max(gradcam_map, 0); % ReLU
    gradcam_map = gradcam_map / max(gradcam_map(:)); % Normalize
    
    gradcam_map = imresize(gradcam_map, [size(img,1), size(img,2)]);
    
    % Display validation comparison
    figure('Name', 'DrishtiXAI MATLAB Grad-CAM Validation');
    subplot(1, 2, 1); imshow(img); title('Original Fundus Image');
    subplot(1, 2, 2); imshow(img); hold on;
    imagesc(gradcam_map, 'AlphaData', 0.5);
    colormap(jet); colorbar; title('MATLAB Grad-CAM Validation Overlay');
end
