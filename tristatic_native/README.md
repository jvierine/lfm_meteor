# Experimental native fitter

Not a validated replacement for the production catalogue. No real-event mass
posterior has yet been accepted from this implementation.

The native library implements fixed-direction shrinking-radius propagation,
standard random-walk Metropolis-Hastings with adaptation confined to warm-up,
and an explicit bivariate Student-t likelihood. MPI ranks run independent
chains, writing separate HDF5 files. The prepared input must supply the actual
FFT reference path and an externally validated error scale matrix. The current
experimental likelihood assumes independent pulse pairs; it does not yet model
temporal residual correlation. Do not use it for publication uncertainties.

The Rust port was paused to prioritize raw-voltage injection/recovery requested
by the user. The current delay/beat observation model must pass those tests
before this native fitter is deployed. The preserved transmit-retarded timing
convention also needs an independent timestamp audit.

Local macOS tests:

```sh
HDF5_DIR=/opt/anaconda3 RUSTFLAGS='-C link-arg=-Wl,-rpath,/opt/anaconda3/lib' cargo test
```

CLI (prepared inputs only):

```sh
mpirun -n 4 tristatic-native input.h5 output_directory 4000 16000 20261029
```

Independent initializations and rank-normalized convergence diagnostics are
still needed before production use. The command currently uses the supplied
initial state on each rank with independent random seeds. Sampler acceptance
is not evidence of posterior convergence.
