% DrishtiXAI - MATLAB ONNX Model Import Script (SIH26038)
% Imports the exported ONNX model into MATLAB Deep Learning Toolbox.

onnx_path = fullfile('..', 'backend', 'weights', 'dr_classifier_v0.1.0.onnx');

if ~exist(onnx_path, 'file')
    error('ONNX model file not found at: %s', onnx_path);
end

fprintf('Importing ONNX network: %s\n', onnx_path);
try
    net = importONNXNetwork(onnx_path, 'OutputLayerType', 'classification');
    disp('Successfully imported ONNX model into MATLAB Deep Learning Toolbox:');
    disp(net);
catch ME
    fprintf('Failed to import ONNX network via Deep Learning Toolbox: %s\n', ME.message);
    fprintf('Ensure Deep Learning Toolbox Converter for ONNX Model Format is installed.\n');
end
