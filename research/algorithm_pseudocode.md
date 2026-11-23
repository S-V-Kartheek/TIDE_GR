# Stage 6b — Algorithm Pseudocode (design-level; not an implementation)

> **SUPERSEDED in detail by [TIDE-GR_spec_v1.md](TIDE-GR_spec_v1.md) §12**, which adds the common-user V-REx, 3-parameter calibration and weekly delayed-feedback ACI. The order of steps is unchanged.

Notation follows `mathematical_formulation.md`. Section references (§) point to that document.

```text
ALGORITHM 1  TIDE-GR — warm-up and stream
------------------------------------------------------------------------
Input : event stream partitioned into blocks B_1..B_K; item texts t_i;
        hyper-parameters H (frozen after warm-up tuning)
State : node beliefs {mu_v, lambda_v}; shared params phi = {omega, psi, a};
        memory M; calibration (kappa, b); forecast eta; ACI threshold tau;
        CUSUM C_v; text map W

# ---------- Warm-up ----------
1  G <- graph(B_1..B_{W0});  envs <- {1..W0}
2  init mu ~ N(0, 0.01 I), lambda <- lambda0, phi
3  (mu, phi) <- TRAIN(G, envs, data = B_1..B_{W0}, prior = (0, lambda0))
4  lambda    <- lambda0 + LAPLACE(G, mu, phi, data = B_1..B_{W0})
5  W         <- WEIGHTED_RIDGE({t_i, mu_i, lambda_i : warm items})
6  (kappa,b), eta <- FIT_CAL(validation slice of B_{W0})
7  M <- MEMORY_INIT(B_1..B_{W0});  C_v <- 0;  tau <- tau0

# ---------- Stream ----------
8  for k = W0 .. K-1:
9      # (1) TEST FIRST - prequential OOD evaluation on the unseen block
10     metrics_k   <- EVALUATE(PREDICT_ALL, B_{k+1})      # accuracy, ECE, risk-coverage
11     err_k       <- selective error of served users on B_{k+1}
12     tau         <- tau + gamma_aci * (err_k - alpha)   # ACI update (§11)

13     # (2) DETECT - per-node standardised surprise, then CUSUM (§9)
14     for v active in B_{k+1}:
15         pairs  <- positives of v in B_{k+1} plus n_neg sampled negatives per positive
16         S_v    <- sum(l_j - H(p_j)) / sqrt(sum Var0_j)
17         C_v    <- max(0, C_v + S_v - delta);   D_v <- C_v
18     for v inactive: C_v <- varsigma * C_v
19     alarm  <- GLOBAL_CUSUM(mean S) or JS(pop_k, pop_{k+1}) > q95_warmup

20     # (3) PREDICT step - Bayesian forgetting gated by drift (§6.1)
21     for v in V_k:
22         q_v      <- q0 + q1 * tanh(D_v / h)
23         lambda_v <- 1 / (1/lambda_v + q_v)
24     mu_prev, lambda_prev <- mu, lambda

25     # (4) ADMIT new nodes (§10)
26     for new user u in B_{k+1}:
27         S_u <- revealed first interactions
28         mu_u <- (|S_u|>0) ? zeta * mean_{i in S_u} m_i : mean of active users
29         lambda_u <- lambda0
30     for new item i in B_{k+1}:
31         mu_i <- W t_i;   lambda_i <- 1 / sigma2_res
32     G <- G + edges(B_{k+1})     # new edges get age 0, so prior rho(0)

33     # (5) UPDATE - joint SGD on the per-block objective (§12.1)
34     envs <- {k+1} ∪ envs(M)
35     epochs <- alarm ? E_ext : E
36     (mu, phi) <- TRAIN(G, envs, data = B_{k+1}^train ∪ M,
                         prior = (mu_prev, lambda_prev), epochs)

37     # (6) LAPLACE precision update; memory-selected edges are excluded (§6.3)
38     M_new  <- SELECT_FOR_MEMORY(B_{k+1}, pi, D)            # §8 priorities
39     lambda <- lambda_prev + gamma_F * GGN_DIAG(G, mu, phi, data = B_{k+1}^train \ M_new)

40     # (7) CLOSED-FORM fits
41     W            <- WEIGHTED_RIDGE({t_i, mu_i, lambda_i})
42     (kappa,b)    <- FIT_CAL(validation slice of B_{k+1})   # must precede the next DETECT
43     eta, psi*    <- alarm ? REFIT_ENV(validation slice) : (eta, psi)

44     # (8) MEMORY update with curvature hand-back on eviction
45     evicted <- M.insert_stratified(M_new, quota per env, one slot per season)
46     lambda  <- lambda + gamma_F * GGN_DIAG(G, mu, phi, data = evicted)

47     # (9) RETENTION evaluation (for forgetting / BWT only; never used for training)
48     for j <= k+1: retention[k+1][j] <- EVALUATE(PREDICT_ALL, heldout_test(B_j))
```

```text
PROCEDURE TRAIN(G, envs, data, prior=(mu_p, lambda_p), epochs)
------------------------------------------------------------------------
repeat epochs (early stop on validation NDCG@20):
  for minibatch drawn stratified over envs:
     sample r~ ~ Concrete(pi_omega(edge feats), t_c)          # §2.3
     Z <- LightGCN(G, weights = r~, Theta)                    # §2.4
     for e in envs: R_e <- mean BPR( s(u,i,e) - s(u,j,e) )    # §3, negatives from N_e
     L <- sum_e R_e + lambda_inv * Var_e(R_e)
          + beta_KL * mean KL(Bern(pi) || Bern(rho(Delta)))
          + (1/N) * [ 0.5 * sum_v ||theta_v - mu_p,v||^2_{lambda_p,v}
                      + 0.5 * lambda_phi ||phi - phi_prev||^2_F ]
     Adam step on (Theta, phi)
return (Theta as new mu, phi)
```

```text
PROCEDURE PREDICT_ALL  (cached once per block)
------------------------------------------------------------------------
m  <- LightGCN(G, weights = pi, mu)                    # mean pass
s  <- LightGCN_sq(G, weights = pi^2-normalised, 1/lambda)  # variance pass (approx.)
for s' = 1..S: Z^(s') <- LightGCN(G, weights ~ Bern(pi), mu)   # edge samples
for user u, candidate i:
   m_ui = m_u . m_i + alpha_u * h_psi(xhat_i^{k+1})
   v_ui = sum_c(m_uc^2 s_ic + m_ic^2 s_uc + s_uc s_ic) + Var_s'(Z_u^(s') . Z_i^(s'))
   p_ui = sigmoid((kappa*m_ui - b) / sqrt(1 + pi*kappa^2*v_ui/8))
   rank_ui = m_ui - beta_r * sqrt(v_ui)
c_u = mean_{top-K} p_ui
serve top-K by rank if c_u >= tau, else serve the environment-trend list with a low-confidence flag
```

```text
PROCEDURE GGN_DIAG(G, mu, phi, data)                   # §6.3
------------------------------------------------------------------------
Zhat <- LightGCN(G, pi, mu);  G_w <- 0 for all nodes w
for (u,i,j) in data:
   x <- s(u,i) - s(u,j);  c <- sigmoid(x)(1-sigmoid(x))
   G_u += c (Zhat_i - Zhat_j)^2;   G_i += c Zhat_u^2;   G_j += c Zhat_u^2
return LightGCN_sq^T(G, pi, G_w)        # sum_w P_wv^2 G_w (diagonal approx.)
```

**Complexity per block (estimates, to be measured).**

* TRAIN: $O(E\cdot L\cdot|E_{\le k}|\cdot d)$, the same order as LightGCN fine-tuning, plus the small edge MLP.
* PREDICT_ALL: $(2+S)$ propagations.
* GGN_DIAG: one pass over the data plus one propagation.
* DETECT: $O(\text{pairs})$.
* New-node admission: $O(n\,d)$.
* Memory: $O(M)$.

Expected total: ≈1.3–1.6× LightGCN fine-tuning per block. **This is an estimate to be verified empirically**, not a result.
