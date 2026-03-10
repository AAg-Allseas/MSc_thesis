# NeuralSDE performance

## Initial JAX/Diffrax test
Baseline:
- Batch of 25
- 11 batches took 26:17. 
- Time per batch: 2.389 minutes or 143.34 seconds

Intial test:
- Batch of 32
- 11090 steps in 184827 seconds: 51 hours, 20 minutes, and 27 seconds
- Time per step: 16.666 seconds

Same timestep size, same sample length
Comparing metrics:
- Time per step: 8.6x speedup
- Samples per second: 0.174 vs 1.92 samples/s. 11.03x speedup

Inference:
- Toy model: 2.237s for 250s run @ 20Hz
- 50 runs: 7.399s on 28 cores
- TorchSDE: 146s for 10800 @ 20Hz - Manual integration
- TorchSDE: 17.352s of 250s @ 20Hz
- TorchSDE: 850s for 10800 @ 20Hz - Prior
- 



# Initial tuning of model with faster version
- Simplifying noise - 3 channels of noise, independent of time or state
  - Additive noise
  - Time invarient
  - 3DOF for noise
  
- Time independent dynamics model (time-invariant system)
  - f/h does not take t in the input