# Simulink / SimEvents Screening Workflow Architecture (SIH26038)

The `screening_workflow.slx` model implements a discrete-event network simulation of the DrishtiXAI screening pipeline.

## Block Diagram Topology

```
+------------------+     +-------------------+     +---------------------+
| Patient Arrival  | --> | Fundus Image      | --> | Network Channel     |
| (Poisson Source) |     | Acquisition Block |     | (Bandwidth / Delay) |
+------------------+     +-------------------+     +---------------------+
                                                              |
                                                              v
+------------------+     +-------------------+     +---------------------+
| Recapture Queue  | <-- | Quality Gate      | <-- | AI Server Worker    |
| (Quality Fail)   |     | Decision Switch   |     | Entity Server Queue |
+------------------+     +-------------------+     +---------------------+
                                   | (Pass)
                                   v
                         +-------------------+
                         | DR Classifier     |
                         | Inference Server  |
                         +-------------------+
                                   |
                         +-------------------+
                         | Referral Switch   |
                         +-------------------+
                          /                 \
                         v                   v
             +-------------------+   +-------------------+
             | Routine Queue     |   | Specialist Review |
             | (Non-referable)   |   | Queue (Level 2+)  |
             +-------------------+   +-------------------+
                                               |
                                               v
                                     +-------------------+
                                     | Ophthalmologist   |
                                     | Review Capacity   |
                                     +-------------------+
```

## Configurable Simulation Parameters & Assumptions

- **Patient Arrival Rate**: `2.5` patients/min (Poisson process) - `ASSUMPTION`
- **Fundus Image File Size**: `3.5` MB / capture - `ASSUMPTION`
- **Network Bandwidth**: `10.0` Mbps rural tele-screening link - `ASSUMPTION`
- **AI Inference Latency**: `0.412` s / image - `MEASURED (DrishtiXAI PyTorch/ONNX Backend)`
- **AI Worker Threads**: `2` parallel workers - `CONFIGURABLE`
- **Quality Rejection Rate**: `8.0` % - `ASSUMPTION`
- **Referral Threshold Probability**: `0.50` - `CONFIGURABLE`
- **Ophthalmologist Capacity**: `12` case reviews / hour / clinician - `ASSUMPTION`
