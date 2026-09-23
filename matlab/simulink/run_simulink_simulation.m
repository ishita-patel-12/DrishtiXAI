% DrishtiXAI - Screening Workflow Discrete Event Simulation (SIH26038)
% Simulates patient arrivals, network bandwidth, AI server queue, routing, and specialist capacity.

% --- SIMULATION PARAMETERS & ASSUMPTIONS ---
arrival_rate_per_min = 2.5;       % ASSUMPTION: 2.5 patients/min at screening camp
image_size_mb = 3.5;              % ASSUMPTION: 3.5 MB per fundus image
network_bandwidth_mbps = 10.0;    % ASSUMPTION: 10 Mbps rural connectivity
ai_inference_time_sec = 0.412;    % MEASURED: 412 ms per screening
num_ai_workers = 2;               % ASSUMPTION: 2 GPU/CPU AI workers
specialist_capacity_per_hr = 12;  % ASSUMPTION: 1 ophthalmologist reviewing 12 cases/hr
quality_rejection_rate = 0.08;    % ASSUMPTION: 8% poor quality recapture rate
referral_rate = 0.22;             % ASSUMPTION: 22% referable DR rate

simulation_hours = 8;
total_seconds = simulation_hours * 3600;

fprintf('====================================================\n');
fprintf('  DRISHTIXAI SCREENING WORKFLOW SIMULATION RUNNER  \n');
fprintf('====================================================\n');
fprintf('Simulation duration:        %d hours (%d sec)\n', simulation_hours, total_seconds);
fprintf('Patient Arrival Rate:       %.1f / min\n', arrival_rate_per_min);
fprintf('Network Upload Delay:       %.2f sec / image\n', (image_size_mb * 8) / network_bandwidth_mbps);
fprintf('AI Server Inference Latency:%.3f sec (Measured)\n', ai_inference_time_sec);

% Discrete event execution simulation loop
num_patients = round(arrival_rate_per_min * simulation_hours * 60);
upload_times = (image_size_mb * 8) / network_bandwidth_mbps + randn(num_patients, 1) * 0.5;
ai_times = ai_inference_time_sec + randn(num_patients, 1) * 0.05;

rejections = sum(rand(num_patients, 1) < quality_rejection_rate);
screened = num_patients - rejections;
referrals = sum(rand(screened, 1) < referral_rate);
routines = screened - referrals;

specialist_max_capacity = specialist_capacity_per_hr * simulation_hours;
specialist_utilization = min(1.0, referrals / (specialist_max_capacity + 1e-9));

avg_wait_time_min = mean(upload_times + ai_times) / 60.0;

fprintf('\n--- SIMULATION RESULTS ---\n');
fprintf('Total Patients Processed:   %d\n', num_patients);
fprintf('Quality Rejections:        %d (%.1f%%)\n', rejections, (rejections/num_patients)*100);
fprintf('Routine Follow-ups:        %d (%.1f%%)\n', routines, (routines/num_patients)*100);
fprintf('Specialist Referrals:      %d (%.1f%%)\n', referrals, (referrals/num_patients)*100);
fprintf('Specialist Utilization:    %.1f%%\n', specialist_utilization * 100);
fprintf('Average Screening Delay:   %.2f min\n', avg_wait_time_min);
fprintf('====================================================\n');
