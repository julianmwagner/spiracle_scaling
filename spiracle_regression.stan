functions {
  real f(real x_i, real a, real b) {
    real y_i = x_i*a + b;
    return y_i;
  }
}

data {
  int<lower=1> N;
  vector[N] x;
  vector[N] y;
  real priora;
  int<lower=1> N_ppc;
  vector[N_ppc] x_ppc;
}

parameters {
  real a;
  real b;
  real<lower=0> sigma;
}

transformed parameters {
  vector[N] mu;
  
  for (i in 1:N) {
    mu[i] = f(x[i], a, b);
  }
  
}

model {
  a ~ normal(priora, 0.3);
  b ~ normal(-1.0, 3.0);
  
  sigma ~ normal(0.0, 1.0);
  y ~ normal(mu, sigma);
  
}

generated quantities {
  vector[N_ppc] y_ppc;
  vector[N_ppc] mu_ppc;

  for (i in 1:N_ppc) {
    mu_ppc[i] = f(x_ppc[i], a, b);
  }
  for (i in 1:N_ppc) { 
      y_ppc[i] = normal_rng(mu_ppc[i], sigma);
  }
  
}