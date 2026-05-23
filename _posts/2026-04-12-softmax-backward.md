# Deriving formula for Backward Gradient of Softmax

In this short note, we aim to derive the backward gradient of a softmax calculation (in Flash Attention) with minimal math background.

# Forward Pass

Given a vector of attention scores $S \in \mathbb{R}^{N}$, softmax produces:

$$
P[i] = \frac{e^{S[i]}}{\sum_{k=0}^{N-1} e^{S[k]}}
$$

Note $P[i] > 0$ and $\sum_i P[i] = 1$.

In flash attention, softmax is applied independently to each row of the attention score matrix $S = QK^T$. Everything below applies per row.

---

# Meaning of dP

In backward prop, we are given the input that has known values and the same shape as $P$:

$$
dP = \frac{\partial L}{\partial P}
$$

where $L$ is a scalar loss. Each element:

$$
dP[i] = \frac{\partial L}{\partial P[i]}
$$

measures how much the loss $L$ changes if $P[i]$ increases slightly.

---

# Goal

We want to compute

$$
dS = \frac{\partial L}{\partial S}
$$

Each element

$$
dS[i] = \frac{\partial L}{\partial S[i]}
$$

represents how much the loss changes if $S[i]$ increases.

---

# Chain Rule

To compute a single element $dS[i]$, we apply the chain rule through $P$:

$$
dS[i] = \frac{\partial L}{\partial P}
\cdot
\frac{\partial P}{\partial S[i]} = \sum_{j=0}^{N-1}
\frac{\partial L}{\partial P[j]}
\cdot
\frac{\partial P[j]}{\partial S[i]}
= \sum_{j=0}^{N-1}
dP[j]
\cdot
\frac{\partial P[j]}{\partial S[i]}
$$

Note both $\frac{\partial L}{\partial P}$ and $\frac{\partial P}{\partial S[i]}$ have the same shape as $P$, and $dS[i]$ is the sum of pointwise product of them.

Also note element at index $[j]$ in $\frac{\partial P}{\partial S[i]}$ is just $\frac{\partial P[j]}{\partial S[i]}$.

Note: unlike matmul, **every** $P[j]$ depends on $S[i]$ because $S[i]$ appears in the denominator $\sum_k e^{S[k]}$. So no terms drop out.

---

# Compute the Partial Derivatives

We need $\frac{\partial P[j]}{\partial S[i]}$ for two cases.

**Case 1: $j = i$**

$$
P[i] = \frac{e^{S[i]}}{\sum_k e^{S[k]}}
$$

Using the quotient rule where the numerator is $e^{S[i]}$ and the denominator is $\sum_k e^{S[k]}$:

$$
\frac{\partial P[i]}{\partial S[i]}
= \frac{e^{S[i]} \cdot \sum_k e^{S[k]} - e^{S[i]} \cdot e^{S[i]}}{\left(\sum_k e^{S[k]}\right)^2}
$$

$$
= \frac{e^{S[i]}}{\sum_k e^{S[k]}} - \frac{e^{S[i]}}{\sum_k e^{S[k]}} \cdot \frac{e^{S[i]}}{\sum_k e^{S[k]}}
$$

$$
= P[i] - P[i]^2 = P[i](1 - P[i])
$$

**Case 2: $j \ne i$**

$$
P[j] = \frac{e^{S[j]}}{\sum_k e^{S[k]}}
$$

Here the numerator $e^{S[j]}$ does not depend on $S[i]$, so using the quotient rule:

$$
\frac{\partial P[j]}{\partial S[i]}
= \frac{0 - e^{S[j]} \cdot e^{S[i]}}{\left(\sum_k e^{S[k]}\right)^2}
= -P[j] \cdot P[i]
$$

---

# Substitute Back

Split the chain rule sum into the $j = i$ term and the $j \ne i$ terms:

$$dS[i] =dP[i] \cdot P[i](1 - P[i])+\sum_{j \ne i}dP[j] \cdot (-P[j] \cdot P[i])$$

Factor out $P[i]$:

$$
dS[i] = P[i] \left(
dP[i] \cdot (1 - P[i]) - \sum_{j \ne i} dP[j] \cdot P[j]
\right)
$$

Expand the first term:

$$
dS[i] = P[i] \left(
dP[i] - dP[i] \cdot P[i]- \sum_{j \ne i} dP[j] \cdot P[j]
\right)
$$

Notice that $dP[i] \cdot P[i] + \sum_{j \ne i} dP[j] \cdot P[j] = \sum_{j} dP[j] \cdot P[j]$, so:

$$
dS[i] = P[i] \left(
dP[i] - \sum_{j} dP[j] \cdot P[j]
\right)
$$

Define the dot product as a single scalar:

$$
D = \sum_{j} dP[j] \cdot P[j] = dP \cdot P
$$

So:

$$
dS[i] = P[i] \cdot (dP[i] - D)
$$

Note $D$ is a fixed scalar no matter what value $i$ is.

---

# Final Result

For the forward operation

$$
P = \text{softmax}(S)
$$

the backward gradient is

$$
dS[i] = P[i] \cdot (dP[i] - D)
$$

where

$$
D = \sum_{j} dP[j] \cdot P[j]
$$

Or in vector form:

$$
dS = P \odot (dP - D)
$$

where $\odot$ is elementwise multiplication and $D$ is a scalar (per row).

---

# Why This Matters for Flash Attention

In flash attention, softmax is applied row-wise to the attention score matrix $S = QK^T$, and the backward pass must be computed **without materializing the full attention matrix** in HBM.

The formula $dS = P \odot (dP - D)$ is perfectly suited for this because:

1. **$D$ is just a scalar per row.** By definition:

$$
D[i] = \sum_j dP[i][j] \cdot P[i][j]
$$

This looks like it needs both $dP$ and $P$, which are full-sized attention matrices we want to avoid materializing. But recall the forward output of attention is $O = PV$, and its gradient is $dO$. We have $dP = dO \cdot V^T$, so:

$$
D[i] = \sum_j dP[i][j] \cdot P[i][j] = \sum_j \left(\sum_l dO[i][l] \cdot V[j][l]\right) \cdot P[i][j]
$$

Swapping the order of summation:

$$
D[i] = \sum_l dO[i][l] \sum_j P[i][j] \cdot V[j][l] = \sum_l dO[i][l] \cdot O[i][l]
$$

since $\sum_j P[i][j] \cdot V[j][l] = O[i][l]$ by the forward definition $O = PV$. So:

$$
D[i] = \sum_l dO[i][l] \cdot O[i][l]
$$

This is just a row-wise dot product of $dO$ and $O$ — both of which are already available in HBM from the forward pass and the incoming gradient. No need to recompute $P$ or $dP$ for this step.

2. **Recomputing $P$ per tile using saved row sum.** During the forward pass, we never have a full row of true $P$ values at once — each tile only sees a partial denominator. But the forward pass saves the sum of exponentials per row:

$$
L[i] = \sum_{k} e^{S[i][k]}
$$

In the backward pass, for a tile covering column block $j$, we recompute the local attention scores $S[i][j] = Q[i] \cdot K[j]^T$ and recover the true softmax values for that tile:

$$
P[i][j] = \frac{e^{S[i][j]}}{L[i]}
$$

This is exactly the softmax definition. The key insight is that $L[i]$ encodes the full-row denominator, so any tile can produce its correct $P$ values independently.

3. **Each tile is independent.** With $D$ precomputed (step 1) and $P$ recoverable per tile (step 2), we form $dP$ from $dO$ and $V^T$ for that tile, and apply $P \odot (dP - D)$. The subtraction of $D$ is the only term that couples different columns within a row, and since $D$ is already a known scalar, each tile can be processed independently.