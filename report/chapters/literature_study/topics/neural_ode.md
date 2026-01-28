# Neural ODE

Fundamental approach:

Consider system:
dh/dt = f(h, theta), where h is the state and theta are parameters. f is the NN. Rather than a RNN approach where h_k+1 = h_k + f(h_k), the system is solved with time integration.

Stochastic Neural ODE is similar.
dh = f(h, theta)dt + g(h, phi)dB, where B is random noise. This uses two neural networks, one for the drift and another of the diffusion.

## Bayesian Neural ODE
dandekarBayesianNeuralOrdinary2022

Used Bayesian inference with Neural ODE to quantify epistemic uncertainty in Neural ODEs. Investigated different sampling approaches.

## Neural SDEs

### lookDeterministicApproximationNeural2023
- Provides an efficient method for inferring the mean and variance of a neural SDE without MC sampling
- Can be used during training time and inference
- Uses a deterministic method to approximate transition kernal:
    \[ p(x(t_n+1) | x(t_n))\]
- In toy case needed approximate cost of 12 MC samples, does not scale as nicely to higher dimensions however

### liuHowDoesNoise2020
- Used neural SDEs as an approach to increase robustness of neural ODEs
- Used a fixed diffusion matrix with neural drift term
- Allows neural ODE to use dropout or gaussian smoothing
- Determine stability criteria of neural SDEs (Networks must be Lipschitz)

### el-lahamVariationalNeuralStochastic2025
- Trains N-SDEs as VAEs, similar to Li et al., 2020
- Introduce change points, points at which the dynamics of the system significantly change
- Has different models for before and after change point
- Could be relevant when considering transient thruster/generator failure
- Mainly suited if dynamics have an unknown shift in the data

### sabbarSelectiveReviewModern2025
- Review of stochastic modelling
- Includes section on neural SDEs
- Mentions how UQ can be done
    - Bayesian Neural SDEs
    - Ensemble variational inference
    - Bootstrap resampling 

### liScalableGradientsStochastic2020
- Introduced the adjoint method for calculating gradients of N-SDE weights
- Uses adjoint method wth "gradient based stochastic variational inference scheme for training latent SDEs."
- Widely cited method for training
- https://github.com/google-research/torchsde

### tzenNeuralStochasticDifferential2019
- More theory based approach
- Predates Li et al., 2020 and Kidger et al., 2021
- Discusses an Autodifferentiation approach to determine gradients for training

### kidgerNeuralSDEsInfiniteDimensional2021
- Trains neural SDEs as Generative Adversorial Networks