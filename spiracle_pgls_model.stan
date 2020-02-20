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
  matrix[N, N] cov_phylo;
}

parameters {
  real<lower=0> a;
  real b;
  //real<lower=0, upper=1> lambda;
  //real<lower=0> sigma;
}

transformed parameters {
  vector[N] mu;
  matrix[N, N] cov;
  
  for (i in 1:N) {
    mu[i] = f(x[i], a, b);
  }
  
  for (i in 1:N) {
    for (j in 1:N) {
      if (! i == j)
        cov[i, j] = cov_phylo[i,j];//*lambda*sigma;
      else
        cov[i, j] = cov_phylo[i,j];//*sigma;
    }
  }
  
}

model {
  a ~ normal(0.66, 0.4);
  b ~ normal(-1.0, 1.0);
  //lambda ~ beta(1.1, 1.1);
  //sigma ~ normal(0.0, 1.0);
  
  y ~ multi_normal(mu, cov);
  
}

generated quantities {
  vector[N] y_ppc;
  vector[N] mu_ppc;

  for (i in 1:N) {
    mu_ppc[i] = f(x[i], a, b);
  }
  
  y_ppc = multi_normal_rng(mu_ppc, cov);
  
}