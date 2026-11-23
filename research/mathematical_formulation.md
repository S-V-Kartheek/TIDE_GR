# Stage 6a — Mathematical Formulation of TIDE-GR

> **SUPERSEDED by [TIDE-GR_spec_v1.md](TIDE-GR_spec_v1.md).** This v0 draft contains six issues (calibration shrinkage toward 0.5, a confirmation-bias edge feature, population-confounded V-REx, mixed loss scales, a vacuous block-level ACI bound, and a 0/0 in the surprise statistic). They are fixed in v1.0, §0. Keep this file for history only.

**TIDE-GR** = **T**emporally-**I**nvariant, **D**rift-gated, **E**vidence-calibrated **G**raph **R**ecommendation.

> Conventions:
>
> * Vectors are bold-free lower case. $\odot$ is the elementwise product, and squares of vectors are elementwise.
> * $\sigma(\cdot)$ is the logistic function and $\Phi$ the standard normal CDF.
> * "sg" denotes stop-gradient.
>
> Statements marked **(approx.)** are deliberate approximations, chosen for tractability and listed again in §13.

---

## 1. Data, graph and environments

**Entities.** Let $\mathcal{U}_k$ and $\mathcal{I}_k$ be the users and items observed up to the end of block $k$, and $\mathcal{V}_k=\mathcal{U}_k\cup\mathcal{I}_k$. Both sets grow: $\mathcal{U}_k\subseteq\mathcal{U}_{k+1}$ and $\mathcal{I}_k\subseteq\mathcal{I}_{k+1}$.

**Events and blocks.** Interactions are timestamped events $(u,i,\tau)$. With block boundaries $T_0<T_1<\dots<T_K$,

$$
B_k=\{(u,i,\tau):\ T_{k-1}\le \tau < T_k\}.
$$

**Temporal interaction graph.** This is the cumulative bipartite graph $G_{\le k}=(\mathcal{V}_k,E_{\le k})$ with $E_{\le k}=\{(u,i): \exists \tau<T_k\}$. Each edge carries:

* its last timestamp $\tau_{ui}$;
* its age $\Delta_{ui}^{(k)}=T_k-\tau_{ui}$;
* its repeat count $c_{ui}$;
* (optionally) its rating.

**Environments.** An environment is a time block, $e\in\{1,\dots,K\}$, with data distribution $P_e(u,i)$.

* The **historical distribution** at step $k$ is the mixture $P_{\le k}=\sum_{e\le k}\omega_e P_e$.
* The **future distribution** is $P_{k+1}$.
* The *training environment set* at step $k$ is $\mathcal{E}_k=\{k\}\cup\mathcal{E}^{\mathcal{M}}_k$, where $\mathcal{E}^{\mathcal{M}}_k$ are the past environments represented in the replay memory (§8).

**Environment covariates (observable spurious factors).** For item $i$ in environment $e$, with $n_i^{e}$ the count of $i$'s interactions in $B_e$:

$$
x_i^{e}=\Big[\log(1+n_i^{e}),\ \log(1+n_i^{e})-\log(1+n_i^{e-1}),\ \log(1+\text{age}_i^{e}),\ \mathrm{onehot}(\text{season}(e))\Big]\in\mathbb{R}^p .
$$

**Target of generalisation.** Let $f_k$ be the model after training on data up to $B_k$. Its OOD risk is

$$
\mathcal{R}_{k+1}(f_k)=\mathbb{E}_{(u,i)\sim P_{k+1}}\big[\ell(f_k;u,i)\big],
$$

and the prequential objective is $\frac{1}{K-W_0}\sum_{k\ge W_0}\mathcal{R}_{k+1}(f_k)$.

---

## 2. Representations

### 2.1 Invariant preference representation (node posterior)

Each node $v$ has a $d$-dimensional **preference parameter** $\theta_v$ with a Gaussian belief

$$
q_k(\theta_v)=\mathcal{N}\big(\mu_v^{(k)},\ \mathrm{diag}(\lambda_v^{(k)})^{-1}\big),\qquad \lambda_v^{(k)}\in\mathbb{R}^d_{>0}.
$$

$\mu_v$ is the MAP estimate and $\lambda_v$ the posterior precision. Collect $\Theta=[\theta_v]_v$.

### 2.2 Environment-specific representation

The environment-specific component is a low-capacity term built only from *observable* covariates:

$$
s^{\text{env}}(u,i,e)=\alpha_u\,h_\psi(x_i^{e}),\qquad \alpha_u=\mathrm{softplus}(a_u).
$$

* $h_\psi:\mathbb{R}^p\to\mathbb{R}$ is a two-layer MLP.
* $a_u$ is a per-user conformity scalar. Users differ in how much they follow trends (cf. DICE, PDA).

The split *preference factor $\theta$* vs. *environment term $s^{\text{env}}$* is enforced by the invariance penalty (§5), not by architecture alone.

### 2.3 Edge reliability (structural latent)

For every edge, introduce $r_{ui}\in\{0,1\}$ ("edge is preference-consistent").

* **Temporal prior** (staleness):
  $$
  p(r_{ui}=1)=\rho(\Delta_{ui})=\rho_\infty+(\rho_0-\rho_\infty)\,e^{-\Delta_{ui}/\tau_\rho}.
  $$
* **Amortised variational posterior:**
  $$
  q_\omega(r_{ui}=1)=\pi_{ui}=\sigma\!\big(g_\omega(f_{ui})\big),
  $$
  with edge features
  $$
  f_{ui}=\big[\log\Delta_{ui},\ \log d_u,\ \log d_i,\ \log(1+c_{ui}),\ \mathrm{sg}(\mu_u^\top\mu_i)\big].
  $$
* **Training samples** use the binary Concrete relaxation (Maddison et al. 2017; Jang et al. 2017) at temperature $t_c$: $\tilde r_{ui}\sim \mathrm{Concrete}(\pi_{ui},t_c)$.

### 2.4 Graph encoder (edge-weighted LightGCN)

Given edge weights $r$ (sampled $\tilde r$ during training, $\pi$ for the mean pass), define $D_u(r)=\sum_i r_{ui}+\epsilon$ (and likewise $D_i$), and the normalised adjacency

$$
\hat A_{ui}(r)=\frac{r_{ui}}{\sqrt{D_u(r)D_i(r)}} .
$$

Propagation is linear:

$$
Z^{(0)}=\Theta,\qquad Z^{(l+1)}=\hat A(r)\,Z^{(l)},\qquad \hat z_v=\sum_{l=0}^{L}\tfrac{1}{L+1}Z_v^{(l)}\ \ \Longleftrightarrow\ \ \hat Z=P(r)\,\Theta,
$$

where $P(r)=\sum_{l}\tfrac{1}{L+1}\hat A(r)^l$.

**User and item embeddings** are $\hat z_u$ and $\hat z_i$.

### 2.5 Score

$$
s(u,i,e)=\underbrace{\hat z_u^\top \hat z_i}_{\text{invariant preference}}+\underbrace{\alpha_u h_\psi(x_i^{e})}_{\text{environment}} .
$$

At inference for the unseen block $k+1$, $x_i^{k+1}$ is unavailable. It is **forecast** in the style of PDA:

$$
\hat x_i^{k+1}=x_i^{k}+\eta\,(x_i^{k}-x_i^{k-1}),
$$

applied to the count/trend coordinates, with the season one-hot known. Here $\eta\in[0,1]$ is fitted on the validation slice.

---

## 3. Recommendation loss (per environment)

The per-environment BPR risk, under edge sampling, is

$$
R_e(\Theta,\phi)=\frac{1}{|D_e|}\sum_{(u,i)\in D_e}\ \mathbb{E}_{j\sim \mathcal{N}_e}\ \mathbb{E}_{\tilde r\sim q_\omega}\Big[-\log\sigma\big(s(u,i,e)-s(u,j,e)\big)\Big].
$$

* $D_e$ is the training interactions of environment $e$: $B_k^{\text{train}}$ for $e=k$, and the memory slice $\mathcal{M}^{e}$ for past $e$.
* $\mathcal{N}_e$ is uniform negative sampling over items active in $B_e$ and not interacted with by $u$.
* $\phi=\{\omega,\psi,a\}$ are the shared parameters.
* The expectation over $\tilde r$ uses one sample per mini-batch.

Message passing always uses the **current** graph $G_{\le k}$ with reliability weights. Environment-specific information enters only through $x^{e}$.

---

## 4. Structural-uncertainty objective

The edge-posterior KL to the temporal prior is

$$
\mathcal{L}_{\text{edge}}=\sum_{(u,i)\in E_{\le k}}\mathrm{KL}\big(\mathrm{Bern}(\pi_{ui})\,\|\,\mathrm{Bern}(\rho(\Delta_{ui}))\big).
$$

Together with $\mathbb{E}_{q_\omega}[\cdot]$ in §3, this is a (tempered) ELBO over edge reliabilities: the BPR term plays the role of the log-likelihood and $\beta_{\text{KL}}$ is the temperature.

* Old edges are *a priori* less reliable.
* The data can overrule this when an old edge keeps explaining behaviour, which is a stable preference.

**Structural uncertainty** per edge is $u^{\text{edge}}_{ui}=\mathcal{H}(\mathrm{Bern}(\pi_{ui}))$. Per node it is the variance contribution $v^{\text{edge}}$ in §7.

---

## 5. Invariance objective (V-REx over time blocks)

$$
\mathcal{L}_{\text{inv}}=\mathrm{Var}_{e\in\mathcal{E}_k}\big[R_e(\Theta,\phi)\big]=\frac{1}{|\mathcal{E}_k|}\sum_{e}\big(R_e-\bar R\big)^2,\qquad \bar R=\frac{1}{|\mathcal{E}_k|}\sum_e R_e .
$$

**Rationale.**

* Because $s^{\text{env}}$ can absorb any variation explained by observable covariates, risk differences that *remain* across blocks signal that the preference factor is fitting block-specific (spurious) patterns. The penalty pushes those patterns out of $\theta$.
* This is a statement of **statistical invariance across observed time environments**. It is not causal identification (see `architecture_comparison.md` §2).
* The penalty requires several environments at every step. That is why the replay memory is stratified by environment (§8). Without memory, $|\mathcal{E}_k|=1$ and the penalty vanishes. This is a structural dependency between Challenges B and D.

---

## 6. Continual learning: online Laplace with drift-gated Bayesian forgetting

### 6.1 Predict step (between blocks; Kalman / Bayesian-forgetting form)

Model preference evolution as a Gaussian random walk, $\theta_v^{(k+1)}=\theta_v^{(k)}+\xi_v$ with $\xi_v\sim\mathcal{N}(0,q_v^{(k)}I)$. Then

$$
\mu_v^{\text{pred}}=\mu_v^{(k)},\qquad \lambda_v^{\text{pred}}=\Big(\big(\lambda_v^{(k)}\big)^{-1}+q_v^{(k)}\mathbf{1}\Big)^{-1}.
$$

The **process noise is gated by the drift score** $D_v^{(k)}$ of §9:

$$
q_v^{(k)}=q_0+q_1\tanh\!\big(D_v^{(k)}/h\big).
$$

* $q_0$ is a small baseline diffusion: stationary forgetting, as in Kurle et al. (2020).
* $q_1$ is the maximal additional plasticity for nodes whose behaviour departs from their own predictive distribution.

### 6.2 Update step (training on block $k+1$)

The prior term is the predicted Gaussian:

$$
\mathcal{L}_{\text{CL}}=\tfrac12\sum_{v\in\mathcal{V}_k}\big(\theta_v-\mu_v^{\text{pred}}\big)^\top\mathrm{diag}\big(\lambda_v^{\text{pred}}\big)\big(\theta_v-\mu_v^{\text{pred}}\big)\;+\;\tfrac12\lambda_\phi\big\|\phi-\phi^{(k)}\big\|^2_{F_\phi}.
$$

* With $q\equiv 0$ and a single global importance, this reduces to online EWC.
* With $q_v$ large it reduces to free fine-tuning for that node.
* **Stability and plasticity are thus decided per node by evidence (precision) and by a calibrated drift test.** This is the central coupling.
* $F_\phi$ is a diagonal empirical Fisher for the small shared-parameter set.

### 6.3 Precision update (Laplace, after training)

With $\mu^{(k+1)}$ the MAP, take the Gauss–Newton curvature of the BPR term. For a pair $(u,i,j)$ with margin $x=s(u,i)-s(u,j)$, $\partial^2(-\log\sigma(x))/\partial x^2=\sigma(x)(1-\sigma(x))$. The diagonal GGN with respect to the **final representations** is closed-form:

$$
G_u=\sum_{(i,j)}\sigma'(x)(\hat z_i-\hat z_j)^2,\qquad G_i\mathrel{+}=\sum_{(u,j)}\sigma'(x)\,\hat z_u^2,\qquad G_j\mathrel{+}=\sum_{(u,i)}\sigma'(x)\,\hat z_u^2 .
$$

Because $\hat Z=P\Theta$ is linear, the GGN with respect to $\Theta$ is $P^\top G P$. We keep its diagonal **(approx.: off-diagonal terms dropped)**:

$$
\lambda_v^{(k+1)}=\lambda_v^{\text{pred}}+\gamma_F\sum_{w}P_{wv}^2\,G_w .
$$

This is computed by one transposed propagation with squared weights, $\hat A^{\circ 2}$.

**No double counting.** Interactions selected into the replay memory at this step are *excluded* from $G$. When they are later evicted, their curvature is added, so each datum is represented either in the Gaussian or in the memory, never both (cf. Kurle et al. 2020).

---

## 7. Predictive distribution and uncertainty

### 7.1 Node-induced (epistemic) score variance

Fix $r=\pi$ (the mean graph). Assuming independent node posteriors **(approx.)**:

$$
m_v=\big(P(\pi)\mu\big)_v,\qquad s_v=\sum_w P_{vw}(\pi)^2\,\big(\lambda_w\big)^{-1}\quad\text{(computed with }\hat A^{\circ2}\text{ layer-wise; approx.)}.
$$

For an inner product of independent Gaussians, $\hat z_u\sim\mathcal{N}(m_u,\mathrm{diag}\,s_u)$ and $\hat z_i\sim\mathcal{N}(m_i,\mathrm{diag}\,s_i)$ **(approx.: the correlation from shared neighbours is ignored)**:

$$
\mathbb{E}[\hat z_u^\top\hat z_i]=m_u^\top m_i,\qquad v^{\text{node}}_{ui}=\sum_{c=1}^{d}\big(m_{u,c}^2 s_{i,c}+m_{i,c}^2 s_{u,c}+s_{u,c}s_{i,c}\big).
$$

### 7.2 Edge-induced (structural) score variance

Draw $S$ samples $r^{(s)}\sim\mathrm{Bern}(\pi)$ with $\Theta=\mu$:

$$
v^{\text{edge}}_{ui}=\widehat{\mathrm{Var}}_{s}\big[\hat z_u(r^{(s)})^\top \hat z_i(r^{(s)})\big].
$$

$S\in\{4,8\}$; the cost is $S$ linear propagations.

### 7.3 Total predictive score distribution (law of total variance)

$$
s_{ui}\ \dot\sim\ \mathcal{N}\big(m_{ui},\,v_{ui}\big),\qquad m_{ui}=m_u^\top m_i+\alpha_u h_\psi(\hat x_i^{k+1}),\qquad v_{ui}=v^{\text{node}}_{ui}+v^{\text{edge}}_{ui}.
$$

### 7.4 Calibrated interaction probability

Define $y_{ui}=1$ if $u$ interacts with $i$ in the next block. Calibration is defined relative to a fixed candidate distribution: positives vs. uniformly sampled active negatives at a stated ratio $1{:}n_{\text{neg}}$.

Using the probit approximation to the logistic–Gaussian integral (MacKay 1992):

$$
p_{ui}=\mathbb{E}_{s\sim\mathcal{N}(m,v)}\big[\sigma(\kappa s-b)\big]\ \approx\ \sigma\!\left(\frac{\kappa m_{ui}-b}{\sqrt{1+\pi\kappa^2 v_{ui}/8}}\right).
$$

$(\kappa,b)$ are fitted by BCE on the validation slice after each update. High variance shrinks $p$ toward the base rate: **confidence decreases with insufficient evidence.**

### 7.5 Decomposition

$$
\underbrace{p(1-p)}_{\text{total}}\;=\;\underbrace{\mathbb{E}_{s}\big[\sigma(1-\sigma)\big]}_{\text{aleatoric}}\;+\;\underbrace{\mathrm{Var}_{s}\big[\sigma(\kappa s-b)\big]}_{\text{epistemic}},\qquad \mathrm{Var}_s[\sigma]\approx \big(\sigma'(\kappa m-b)\big)^2\kappa^2 v .
$$

The epistemic part is further split into node and edge components in proportion to $v^{\text{node}}$ and $v^{\text{edge}}$ (delta method).

---

## 8. Continual-learning memory

The replay memory $\mathcal{M}_k=\bigcup_{e\in\mathcal{E}^{\mathcal{M}}_k}\mathcal{M}^{e}$ has total budget $|\mathcal{M}_k|\le M$, with an equal per-environment quota $M/|\mathcal{E}^{\mathcal{M}}_k|$ over the most recent $W_{\mathcal{M}}$ environments plus one slot per season (so recurring seasonal environments are retained).

Within each environment, edges are kept by **reliability- and drift-aware reservoir sampling**. The retention priority is

$$
\varrho_{ui}=\pi_{ui}\cdot\exp\!\big(-D_u/h\big)\cdot\exp\!\big(-D_i/h\big),
$$

using reservoir sampling with weights (Efraimidis & Spirakis A-Res). The memory therefore stores *still-valid, reliable* knowledge, and evicts edges of drifted nodes. Ablation: a uniform reservoir.

---

## 9. Shift detection

### 9.1 Per-node standardised predictive surprise

When block $k+1$ arrives, and **before** training on it, take for node $u$ its observed positives plus $n_{\text{neg}}$ sampled negatives per positive, $\{(j,y_j)\}$, scored by $f_k$.

The per-pair Bernoulli NLL is $\ell_j=-y_j\log p_j-(1-y_j)\log(1-p_j)$. Under the null that $y_j\sim\mathrm{Bern}(p_j)$ independently (the model is calibrated and nothing changed):

$$
\mathbb{E}_0[\ell_j]=\mathcal{H}(p_j),\qquad \mathrm{Var}_0[\ell_j]=p_j(1-p_j)\Big(\log\tfrac{p_j}{1-p_j}\Big)^2 .
$$

$$
S_u^{(k+1)}=\frac{\sum_j\big(\ell_j-\mathcal{H}(p_j)\big)}{\sqrt{\sum_j \mathrm{Var}_0[\ell_j]}}\ \xrightarrow{d}\ \mathcal{N}(0,1)\ \text{under }H_0\ \text{(Lyapunov CLT, many pairs)}.
$$

**Why this separates "insufficient evidence" from "preference changed".**

* For a sparse node, $v$ is large, so $p_j$ is moderated toward the base rate. $\mathcal{H}(p_j)$ is then large and $\mathrm{Var}_0$ is small, so wrong predictions produce *small* $S_u$.
* For a well-evidenced node that changes behaviour, $p_j$ is confident and wrong, which gives a *large* $S_u$.

### 9.2 Sequential accumulation (one-sided CUSUM)

$$
C_u^{(k+1)}=\max\!\big(0,\ C_u^{(k)}+S_u^{(k+1)}-\delta\big),\qquad D_u^{(k+1)}=C_u^{(k+1)} .
$$

$\delta$ is the allowance and $h$ the decision scale used in §6.1. Inactive nodes keep their $C$ and decay it: $C\leftarrow \varsigma C$.

### 9.3 Global environment monitor

$$
J_{k+1}=\mathrm{JS}\big(\mathrm{pop}_{k}\,\|\,\mathrm{pop}_{k+1}\big),\qquad \bar S_{k+1}=\mathrm{mean}_u S_u^{(k+1)} .
$$

A global CUSUM on $\bar S$, or $J$ above its warm-up 95th percentile, raises an **alarm**. The alarm triggers (i) a refit of $h_\psi$ and $\eta$, and (ii) an extended epoch budget $E_{\text{ext}}$.

Degree-distribution statistics are logged for analysis only.

---

## 10. Open-world initialisation

**New user** $u$ with $n\ge 0$ revealed interactions $S_u$:

$$
\mu_u^{0}=\begin{cases}\zeta\,\frac{1}{n}\sum_{i\in S_u} m_i & n\ge1\quad\text{(inductive neighbour mean; IDCF/INMO spirit)}\\[2pt] \bar\mu_{\mathcal{U}}^{\text{active}} & n=0\end{cases},\qquad \lambda_u^{0}=\lambda_0\mathbf{1}+\gamma_F\textstyle\sum_{w}P_{wu}^2G_w .
$$

* With $n=0$ the precision is the small prior $\lambda_0$, so the variance is maximal.
* As interactions arrive, a **local MAP refinement** runs $T_{\text{loc}}$ gradient steps on $u$'s own BPR loss plus its prior, with all other parameters frozen. It costs $O(n\,d\,T_{\text{loc}})$ and requires no retraining.

**New item** $i$ with text embedding $t_i$ from a frozen encoder:

$$
\mu_i^{0}=W t_i,\qquad \lambda_i^{0}=\big(\hat\sigma^{2}_{\text{res}}\big)^{-1},
$$

where $W$ solves the **precision-weighted ridge regression** over warm items:

$$
\mathcal{L}_{\text{cold}}(W)=\sum_{i\in\mathcal{I}^{\text{warm}}_k}\big(\mu_i-Wt_i\big)^\top\mathrm{diag}(\lambda_i)\big(\mu_i-Wt_i\big)+\lambda_W\|W\|_F^2 ,
$$

solved per output dimension in closed form. $\hat\sigma^2_{\text{res}}$ is the per-dimension weighted residual variance. Well-determined items dominate the fit, and the new item's uncertainty reflects how well text predicts CF position.

---

## 11. Final ranking score and selective recommendation

**Ranking score** for user $u$ and candidate $i$ in block $k+1$:

$$
\mathrm{rank}_{ui}=m_{ui}-\beta_r\sqrt{v_{ui}} .
$$

* $\beta_r=0$ gives accuracy mode.
* $\beta_r>0$ gives reliability mode, a lower-confidence-bound ranking.
* $\beta_r<0$ would be exploration, which is **not evaluated offline**.

**Selective recommendation.** User-level confidence is $c_u=\frac1K\sum_{i\in\text{top-}K}p_{ui}$.

* Serve the personalised list if $c_u\ge\tau_k$.
* Otherwise fall back to the environment-trend list, ranked by $h_\psi(\hat x_i^{k+1})$, and flag it as low-confidence.

**Online threshold (ACI-style controller).** Let $\mathrm{err}_k$ be the selective error, i.e. the fraction of *served* users with zero hits in their top-$K$ in block $k$:

$$
\tau_{k+1}=\tau_k+\gamma_{\text{aci}}\big(\mathrm{err}_k-\alpha\big).
$$

Following Gibbs & Candès (2021, Prop. 4.1), provided $\mathrm{err}_k\in[0,1]$ and $\tau$ at its bounds forces $\mathrm{err}=0$ (abstain all, by convention) or the maximal error, the long-run average satisfies

$$
\big|\tfrac{1}{T}\textstyle\sum_k \mathrm{err}_k-\alpha\big|\le \tfrac{\tau_{\max}-\tau_{\min}+\gamma_{\text{aci}}}{\gamma_{\text{aci}}T}
$$

**for any sequence of distributions**. This is the guarantee we claim for reliability under drift: an *average-over-time* guarantee, not a per-user one.

---

## 12. Overall objective and optimisation

### 12.1 Complete per-block objective (step $k$, learning from $B_k$ and $\mathcal{M}_{k-1}$)

$$
\boxed{
\begin{aligned}
\mathcal{L}_k(\Theta,\phi)=\;&\sum_{e\in\mathcal{E}_k} R_e(\Theta,\phi)\;+\;\lambda_{\text{inv}}\,\mathrm{Var}_{e\in\mathcal{E}_k}\!\big[R_e(\Theta,\phi)\big]\\
&+\;\frac{\beta_{\text{KL}}}{|E_{\le k}|}\sum_{(u,i)\in E_{\le k}}\mathrm{KL}\big(\mathrm{Bern}(\pi_{ui})\,\|\,\mathrm{Bern}(\rho(\Delta_{ui}))\big)\\
&+\;\frac{1}{N_k}\Big[\tfrac12\sum_{v}\|\theta_v-\mu_v^{\text{pred}}\|^2_{\mathrm{diag}(\lambda_v^{\text{pred}})}+\tfrac12\lambda_\phi\|\phi-\phi^{(k-1)}\|^2_{F_\phi}\Big]
\end{aligned}}
$$

$N_k=|B_k^{\text{train}}|+|\mathcal{M}_{k-1}|$ converts the summed Gaussian prior to the mean-risk scale, so the objective is a tempered generalised posterior.

The following are solved in **closed form after** minimising $\mathcal{L}_k$:

* $\mathcal{L}_{\text{cold}}$ (§10);
* the Laplace precision (§6.3);
* the calibration $(\kappa,b)$ and forecast $\eta$ (§7.4, §2.5).

They are excluded from the gradient objective by design: they depend on the MAP solution and have no gradient pathway that would benefit joint training.

### 12.2 Why these terms and not others

| Considered objective | Included? | Reason |
|---|---|---|
| Recommendation (BPR) | ✔ | Core likelihood |
| Temporal modelling loss | ✘ as a separate loss | Time enters via the predict step (§6.1), the edge prior (§4) and environment covariates/forecasting (§2.5). A separate sequence loss would add an encoder whose effect is not attributable |
| Invariance | ✔ V-REx | Works with BPR; no adversary |
| Contrastive | ✘ | Not a temporal-invariance mechanism; overlaps with invariance; complicates attribution |
| DRO | ✘ | Redundant with V-REx at large $\lambda_{\text{inv}}$; kept as baselines |
| Structural uncertainty | ✔ KL to temporal prior | Gives the edge posterior |
| Predictive calibration | ✔ post-hoc and online | Calibration *inside* the training loss would conflict with the ranking objective and is not guaranteed under drift; post-hoc + ACI has a guarantee |
| Continual regulariser | ✔ drift-gated online Laplace | Coupled to uncertainty |
| Replay loss | ✔ implicitly: memory environments are included in $\sum_e R_e$ | Also enables the invariance penalty |
| Open-world objective | ✔ closed-form weighted ridge | Cheap; no retraining |

### 12.3 Compatibility notes

1. **V-REx makes the objective non-additive over data.** The Gaussian belief is therefore a Laplace approximation of a *generalised (Gibbs) posterior* for the loss $\mathcal{L}_k$, not of a Bayesian likelihood posterior. The curvature is computed from the BPR term only **(approx.)**. We state this rather than claim exact Bayesian inference.
2. **Invariance vs. forgetting.**
   * Invariance pulls the preference factor toward what was predictive in *all* retained environments.
   * Drift-gated forgetting lets *specific nodes* move.
   * These are compatible because they act on different axes: invariance acts across environments for shared structure; forgetting acts per node over time. They would conflict only if a large fraction of users drifted at once. In that case the global alarm extends training and the memory evicts drifted nodes' edges, so the penalty is computed on still-valid data.
3. **Edge reliability vs. drift.** Low $\pi$ for old edges and high drift both say "the past is less relevant", but at different granularities: edge vs. node. Ablations isolate each.

### 12.4 Schedule

* **Warm-up (sequential, then joint).**
  * Train on $B_1..B_{W_0}$, with each block an environment and isotropic prior $\lambda=\lambda_0$ (equivalent to weight decay).
  * Use full joint SGD (Adam) on $\mathcal{L}$ with early stopping on prequential validation.
  * Then compute the precision, $W$, $(\kappa,b)$, $\eta$ and $\tau$ initialisation.
* **Stream step $k\to k+1$.** The order is *test → detect → predict → admit new nodes → update (joint SGD) → Laplace → closed-form fits → memory*; see `algorithm_pseudocode.md`.
  * The gradient step is **joint** over $(\Theta,\phi)$.
  * The block-level procedure is **sequential/alternating**: detection → prior → MAP → curvature → calibration.

### 12.5 Hyperparameters

| Symbol | Meaning | Suggested search range |
|---|---|---|
| $d, L$ | Embedding size, propagation layers | 64; {2, 3} |
| $\lambda_{\text{inv}}$ | Invariance strength | {0, 0.1, 1, 10} |
| $\beta_{\text{KL}}$ | Edge-posterior temperature | {1e-3, 1e-2, 1e-1} |
| $\rho_0,\rho_\infty,\tau_\rho$ | Edge prior: fresh, stale, decay time | 0.95; {0.5, 0.7}; {1, 2, 4} blocks |
| $t_c$ | Concrete temperature | 0.5 |
| $q_0,q_1$ | Baseline and drift-gated process noise | $q_0\in\{0, 10^{-3}\}\cdot\bar\lambda^{-1}$; $q_1\in\{0.1, 1\}\cdot\bar\lambda^{-1}$ |
| $\delta,h,\varsigma$ | CUSUM allowance, scale, decay | 0.5; {3, 5}; 0.9 |
| $\gamma_F$ | Curvature tempering | {0.1, 1} |
| $M, W_{\mathcal{M}}$ | Memory budget, number of retained environments | {1%, 5%} of $\lvert B\rvert$; {4, 8} |
| $\lambda_0,\zeta,T_{\text{loc}}$ | New-node prior precision, neighbour-mean scale, local steps | grid on warm-up |
| $\lambda_W$ | Ridge for the text map | {1e-2, 1, 1e2} |
| $\beta_r$ | Risk aversion in ranking | {0, 0.5, 1} |
| $\alpha,\gamma_{\text{aci}}$ | Target selective error, ACI step | task-defined; 0.05 |
| $n_{\text{neg}}$ | Negatives per positive for calibration and surprise | 99 |
| $S$ | Edge MC samples | 4 |

---

## 13. Summary of approximations (to be stated in the paper)

1. Diagonal Gaussian node posteriors; independence across nodes.
2. Layer-wise squared-weight variance propagation, which ignores path covariance.
3. The inner-product variance ignores the $u$–$i$ correlation induced by shared neighbours.
4. GGN from the BPR term only; off-diagonal terms are dropped; the posterior is a generalised posterior.
5. The probit approximation in §7.4.
6. The CLT null for $S_u$ is inaccurate for nodes with very few pairs. We use $n_{\text{neg}}$ negatives so each active node has ≥100 pairs, and report the empirical null on non-drifted synthetic users.
7. The ACI guarantee is marginal over time, not conditional on users.
