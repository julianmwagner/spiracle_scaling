functions {
  real f(real x_i, real a, real b) {
    real y_i = x_i*a + b;
    return y_i;
  }
}

data {
  int<lower=1> N;
  vector[N] x;
  real a_mu;
  real a_sig;
  real b_mu;
  real b_sig;
  real lambda_alpha;
  real lambda_beta;
  matrix[N, N] cov_phylo;
}

generated quantities {
  // Data
  vector[N] y;
  matrix[N, N] cov;
  vector[N] mu;
  
  real a = normal_rng(a_mu, a_sig);
  real b = normal_rng(b_mu, b_sig);
  real lambda = beta_rng(lambda_alpha, lambda_beta);
  
  for (i in 1:N) {
    for (j in 1:N) {
      if (! i == j)
        cov[i, j] = cov_phylo[i,j]*lambda;
     else
     cov[i, j] = cov_phylo[i,j];
    }
  }
  
  
  for (i in 1:N) {
    mu[i] = f(x[i], a, b);
  }
  
  y = multi_normal_rng(mu, cov);
}