

## Hierarchical log-beta prior derivation

Integrating out per-mouse aging rate $\beta^{(i)}_{\star}$, we get the
likelihood for the per-run aging rates  $\beta^{(i)}_{j}$ in terms of the 
priors $\beta_0, \tau, \sigma$.

From: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
$$\begin{aligned}
p\left(\mathcal{\beta^{(i)}} \mid \beta_0, \sigma^{2}, \tau^{2}\right) &=\int\left[\prod_{j=1}^{n} \mathcal{N}\left(\beta^{(i)}_{j} \mid \beta^{(i)}_{\star}, \sigma^{2}\right)\right] \mathcal{N}\left(\beta^{(i)}_{\star} \mid \beta_0, \tau^{2}\right) d \beta^{(i)}_{\star} \\
&=\frac{\sigma}{(\sqrt{2 \pi} \sigma)^{n} \sqrt{n \tau^{2}+\sigma^{2}}}
\exp \left(-\frac{\sum_{j} \left(\beta^{(i)}_{j}\right)^{2}}{2 \sigma^{2}}-\frac{\beta_0^{2}}{2 \tau^{2}}\right)
\exp \left(\frac{\frac{\tau^{2} n^{2} \bar{\beta^{(i)}}^{2}}{\sigma^{2}}+\frac{\sigma^{2} \beta_0^{2}}{\tau^{2}}+2 n \bar{\beta^{(i)}} \beta_0}{2\left(n \tau^{2}+\sigma^{2}\right)}\right)
\end{aligned}$$

Taking log probabilities and summing across mice:



$$\begin{aligned}
\sum_i \ell\left(\beta^{(i)}\right) &= \sum_i \left[
\log C_i
+ \left(-\frac{\sum_{j} \left(\beta^{(i)}_{j}\right)^{2}}{2 \sigma^{2}}-\frac{\beta_0^{2}}{2 \tau^{2}}\right)
+ \left(\frac{\frac{\tau^{2} n_i^{2} \bar{\beta^{(i)}}^{2}}{\sigma^{2}}+\frac{\sigma^{2} \beta_0^{2}}{\tau^{2}}+2 n_i \bar{\beta^{(i)}} \beta_0}{2\left(n_i \tau^{2}+\sigma^{2}\right)}\right) \right]\\
&= \sum_{i, j} \left(-\frac{\left(\beta^{(i)}_{j}\right)^{2}}{2 \sigma^{2}}\right) +
\sum_i \left[
C_i +
\left(\frac{\frac{\tau^{2} n_i^{2} \bar{\beta^{(i)}}^{2}}{\sigma^{2}}+\frac{\sigma^{2} \beta_0^{2}}{\tau^{2}}+2 n_i \bar{\beta^{(i)}} \beta_0}{2\left(n_i \tau^{2}+\sigma^{2}\right)}\right) -\frac{\beta_0^{2}}{2 \tau^{2}} \right]
\end{aligned}$$

where $$C_i = \log(\sigma) - n_i \cdot \log \left( \sqrt{2 \pi} \sigma \right) - 0.5\cdot\log\left(n_i \tau^{2}+\sigma^{2}\right)$$


