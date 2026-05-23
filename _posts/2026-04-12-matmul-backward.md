# Deriving formula for Backward Gradient of Matrix Multiplication

Reference: https://cs231n.stanford.edu/handouts/derivatives.pdf

In this short note, we derive the backward gradient of a matrix multiplication with minimal math background.

# Forward Pass

$$
Y = WX
$$

Matrix shapes:

$$
Y \in \mathbb{R}^{M \times K}
$$

$$
W \in \mathbb{R}^{M \times N}
$$

$$
X \in \mathbb{R}^{N \times K}
$$

---

# Meaning of dY

In backward prop, we are given the input that has known values and the same shape as $Y$:

$$
dY = \frac{\partial L}{\partial Y}
$$

where $L$ is a scalar loss.

Thus:

$$
dY[i][j] = \frac{\partial L}{\partial Y[i][j]}
$$

Intuitively, $dY[i][j]$ measures how much the loss $L$ changes if $Y[i][j]$ increases slightly.

$$
dY[i][j] = \frac{L(Y[i][j] + h) - L(Y[i][j])}{h}
$$

for a very small $h$.

---

# Goal

We want to compute

$$
dX = \frac{\partial L}{\partial X}
$$

Each element

$$
dX[i][j] = \frac{\partial L}{\partial X[i][j]}
$$

represents how much the loss changes if $X[i][j]$ increases.

---

# Chain Rule

To compute a single element $dX[i][j]$ in $dX$, we apply the chain rule through $Y$:

$$
dX[i][j]= \frac{\partial L}{\partial Y}
\cdot
\frac{\partial Y}{\partial X[i][j]} =
\sum_{a=0}^{M-1} \sum_{b=0}^{K-1}
\frac{\partial L}{\partial Y[a][b]}
\cdot
\frac{\partial Y[a][b]}{\partial X[i][j]}
$$

Note both $\frac{\partial L}{\partial Y}$ and $\frac{\partial Y}{\partial X[i][j]}$ are of the same shape as Y, and
$dX[i][j]$ is the sum of pointwise product of them.

Also note element at index $[a][b]$ in $\frac{\partial Y}{\partial X[i][j]}$ is just $\frac{\partial Y[a][b]}{\partial X[i][j]}$.

Using the shorthand $dY[a][b] = \frac{\partial L}{\partial Y[a][b]}$:

$$
dX[i][j]=
\sum_{a,b}
dY[a][b]
\cdot
\frac{\partial Y[a][b]}{\partial X[i][j]}
$$

Intuitively:

- $\frac{\partial Y[a][b]}{\partial X[i][j]}$ measures how $X[i][j]$ affects $Y[a][b]$
- $dY[a][b]$ measures how $Y[a][b]$ affects the loss $L$

Multiplying them gives the influence path

$$
X[i][j] \rightarrow Y[a][b] \rightarrow L
$$

Summing over all $a,b$ aggregates all such paths, showing the influence $X[i][j]$ on $L$.

---

# Expand the Forward Definition

From matrix multiplication:

$$
Y[a][b] =
\sum_{c=0}^{N-1}
W[a][c] \cdot X[c][b]
$$

That is:

$$
Y[a][b]=
W[a][0]X[0][b] + W[a][1]X[1][b] + \dots + W[a][N-1]X[N-1][b]
$$

---

# Dependency Observation

If $b \ne j$, then $Y[a][b]$ **does not depend on** $X[i][j]$. It only depends on row $a$ of $W$ and column $b$ of $X$.

Therefore:

$$
\frac{\partial Y[a][b]}{\partial X[i][j]} = 0
$$

Thus only terms where $b = j$ contribute.

The formula simplifies from

$$
dX[i][j]=
\sum_{a,b}
dY[a][b]
\cdot
\frac{\partial Y[a][b]}{\partial X[i][j]}
$$

to

$$
dX[i][j]=
\sum_{a=0}^{M-1}
dY[a][j]
\cdot
\frac{\partial Y[a][j]}{\partial X[i][j]}
$$

---

# Compute the Partial Derivative

Consider

$$
Y[a][j]=
W[a][0]X[0][j] + W[a][1]X[1][j] + \dots + W[a][i]X[i][j] + \dots
$$

The **only term involving $X[i][j]$** is

$$
W[a][i]X[i][j]
$$

Therefore

$$
\frac{\partial Y[a][j]}{\partial X[i][j]} = W[a][i]
$$

Then

$$
dX[i][j]=
\sum_{a=0}^{M-1}
dY[a][j]
\cdot
\frac{\partial Y[a][j]}{\partial X[i][j]}
$$

becomes

$$
dX[i][j]=
\sum_{a=0}^{M-1}
dY[a][j] \cdot W[a][i]
$$

then exchange position of two elements

$$
dX[i][j]=
\sum_{a=0}^{M-1}
W[a][i] \cdot dY[a][j]
$$

---

# Recognizing the Matrix Form

Notice

$$
W^T[i][a] = W[a][i]
$$

Thus

$$
dX[i][j]=
\sum_{a=0}^{M-1}
W^T[i][a] \cdot dY[a][j]
$$

This is exactly the definition of matrix multiplication

$$
dX = W^T dY
$$

---

# Final Result

For the forward operation

$$
Y = WX
$$

the backward gradients are

$$
\frac{\partial L}{\partial X} = W^T dY
$$

If we go through a very similar process, we could also prove

$$
\frac{\partial L}{\partial W} = dY X^T
$$

How to remember this: for Y=WX or Y=XW, when we want to compute dX given dY, we always just swap positions of X(dX) and Y(dY), and then just transpose W without changing its location in the equation.