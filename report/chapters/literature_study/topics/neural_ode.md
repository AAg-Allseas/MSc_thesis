# Neural ODE

Fundamental approach:

Consider system:
dh/dt = f(h, theta), where h is the state and theta are parameters. f is the NN. Rather than a RNN approach where h_k+1 = h_k + f(h_k), the system is solved with time integration.

Stochastic Neural ODE is similar.
dh = f(h, theta)dt + g(h, phi)dB, where B is random noise. This uses two neural networks, one for the drift and another of the diffusion.

## Bayesian Neural ODE
dandekarBayesianNeuralOrdinary2022

Used Bayesian inference with Neural ODE to quantify epistemic uncertainty in Neural ODEs. Investigated different sampling approaches.

