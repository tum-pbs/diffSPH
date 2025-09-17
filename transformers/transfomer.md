# Transformers

## Input:

- Given a set of input points $p_I$ and output points $p_A$
- Each point has a set of features $f\in\mathbb{R}^n$ and a position $\mathbf{x}\in\mathbb{R}^d$. Note that for for structured data we can assume that $\mathbb{x}\in[0,\dots,w)\times[0,\dots,h)$ instead, and for token sequences $\mathbf{x}\in[0,s)$ for some sequence of length $s$.
- There is a connectivity from $p_I$ to $p_A$, $\mathcal{N}:p_I\times p_A$, that indicates which points are connected to which. For a normal transformer this is a densely connected adjacency, whereas for patched networks this is very different.

## Patching

With patching the adjacency matrix $\mathcal{N}$ is split based on some criterion. 

- For naive vision transformers the domain is split into regular sub images of fixed size $p\times p$ of size $W\times H$ such that $Wp = w$. Each patch is then treated independently of every other patch and there is no information exchange between patches.
- One option is to include a ghost layer around each patch to have information flow from a neighboring patch into the current patch, i.e., patches are effectively of size $p+g\times p+g$ with an overlap. The transformer then computes outputs only for the "real" entries, i.e., $p_I = p+g$ and $p_A = p$. But this may lead to awkward patch sizings
- An alternative is shifted windows. In one transformer layer attentions are computed on patches of size $p$ relative to $0,0$ but for the next layer the reference point of the windows is shifted to $s,s$. This creates a need for padding on the boundaries as the windows would go outside of the image, but directly communicates information between patches without awkward sizes. This is what SWIN does
- Another alternative is to patch as usual but then compute a reference token per crop/anchor, i.e., one could use an mlp that maps all $p\times p$ entries from a single crop into a $1\times 1$ value, using a U-shape network, and then an additional transformer layer is added on top that computes a fully connected attention between all crops. The $1\times 1$ would be a full U-shape but networks like P3D use a minimum size, e.g., $4\times 4$ skipping some of the deeper U-shape layers. AB-UPT always propagates to individual entries.

Patching for unstructured data is more complex. UPT choses a random subset of tokens as anchor tokens and agglomerates information only on those for efficiency, but we could also utilize SPH like biases (background grids with ghost cells for a natural intra-inter cell approach with some inductive bias from CFL conditions on the size of the patches)

## Input Encoder

Instead of directly using $f$, most transformer architectures use an encoded input set of features, e.g., using a dictionary value, that encodes the input features $f\in\mathbb{R}^n$ to a latent space $\tilde{f}\in\mathbb{R}^L$, for example using an MLP. This MLP could also be trained in coordination with a matching output network as an auto-encoder or end-to-end. 

## Absolute Position Bias (APB)

The original attention is all you need paper adds a position embedding to this encoder, i.e., an absolute position bias (APB). This works by concatenating $f$ with $\mathbf{x}$ before feeding them into an encoding layer, such as an MLP. These features are not necessary for the output decode layer and are not viable in physics problems as they violate translational invariance that is crucial for PDEs. These can also be mapped with a learned embedding, e.g., given the absolute position a small MLP could learn the embedded value, but this does not avoid the invariance issue. See later for relative position biases.

## Transformer Input

At this point the next steps are independent w.r.t. the application area, i.e., the crop based approach and patch based approaches all work on the same fundamental inputs. For convenience denote $f$ as the encoded features. The current state of inputs is:

- A set of output tokens $p_i\in p_A$ with features $f_i\in\mathbb{R}^L$ and positions $\mathbf{x}_i\in\mathbb{R}^d$
- A set of input tokens $p_j\in p_I$ with features $f_j\in\mathbb{R}^L$ and positions $\mathbf{x}_j\in\mathbb{R}^d$
- An adjacency matrix $\mathcal{N}$ that denotes pairs of tokens $ij$ that are connected
- A set of edge features $\epsilon_{ij}\in\mathbb{R}^d$ for the relative position of tokens based on the adjacency. For non point cloud problems this can be of an arbitrary dimension $\epsilon_{ij}\in\mathbb{R}^E$, e.g., in molecule graphs, but here we assume only position information on edges

## Query, Keys, Values

A standard Transformer now has three matrices $W_K$, $W_Q$ and $W_V$ for key, query and value information (note that GAT assumes them $W_K$ and $W_Q$ to be equal), which can be used to compute the Query values of $i$, which use $i$ as the query is what the output tokens look for, wheras the input tokens $j$ are what the input tokens provide, hence the keys. That means we get:

$$Q_i = W_Q \cdot f_i,\; K_j = W_K \cdot f_k,\;V_j = W_V \cdot f_j$$

## Attention Mechanism

For a normal attention mechanism we can then compute the pairwise attention score for every entry in $\mathcal{N}$ as

$$\alpha_{ij} = \langle Q_i, K_j \rangle $$

To normalize the attention we apply a softmax function across the different incoming connections per output token, i.e., over $j$ as

$$w_{ij} = \operatorname{softmax}_j \alpha_{ij}$$

Note that often times the attention value is scaled inversely proportional to the dimensionality, e.g., with $\frac{1}{\sqrt{d)_k}}$

## Message Passing

These attention scores are then used to weigh the different input values $V_j$ multiplicatively, which gives a set of messages

$$m_{ij} = w_{ij} V_j$$

These messages are then agglomerated, using generally summation operations but mean operations and concatenation (for fully connected or equal connected tokens), resulting in a set of outputs

$$o_i = \sum_j w_{ij} V_j$$

This output is then linearily transformed back to the input shape

$$\hat{o}_i = W_o o_i$$

Note that if there are no incoming edges then the common use of a skip connection still results in a valid entry on $\hat{o}_i$

## Feed Forward Network (FFN)

This also facilitates skip connections, e.g., adding $f_i$ at this point. The output of this step is then potentially normalized using a layer norm and fed through a feed forward network with potential skip connections, e.g.,

$$\hat{f}_i = \operatorname{LayerNorm}\left[\operatorname{MLP}(\hat{o}_i) + \hat{o}_i\right]$$

## Scaled Attention

A trivial extension to this is adding a scaling to the attention per output token, e.g., 

$$w_{ij} = s_i \operatorname{softmax}_j \alpha_{ij}$$

e.g., using $s_i = 1/\sum_j \frac{m_j}{\rho_j} W_{ij}$, which ensures that the message passing step is a mathematical convolution operation using a shepard filter.

## Multi-Head

Multi-Head attention is a simple mechanism that changes the computation of the attention mechanism from being compute simultaneously across all Q and K entries, to splitting this into multiple parts with independent attention scaling. For example, given a query and key feature dimension of 128, the normal attention mechanism computes one attention score for all 128 features. With $8$ multi-heads the input features are split into $8$ sets of $16$ features each, which each independently get an attention score, i.e., we get $8$ attention scores per connection. These are then also softmaxed and scaled only within each attention score, and not across them.

## Non-Linear attention

Instead of using $\alpha_{ij} = \langle Q_i, K_j \rangle$ a trivial extension of the attention mechanism is using an MLP that maps to a single attention score, e.g., 

$$ \alpha_{ij} = \operatorname{MLP}\left[Q_i, K_j \right],$$

which works by concatenating the query and key values together and feeding them into an MLP that produces a single output, e.g., $\mathbb{R}^{Q+K}\rightarrow\mathbb{R}$.

## Graph Attention Networks (GAT)

In a GAT a simplified form of the previous non-linear attention is used where the MLP is replaced with a single linear transformation, e.g., 

$$ \alpha_{ij} = W_a \left[Q_i, K_j\right],$$

with a learned attention matrix $W_a$. This can also be done using a bilinear form

$$ \alpha_{ij} = Q_i^T W^T W K_j$$

as GAT assume $W_Q = W_K$, which gives a scaled dot product for attention.

## Edge Attention

Including edge attention can be done either using an edge bias or as a direct component of the attention mechanism. For an edge bias we require some transformation function $e$ that given the edge features $\epsilon_{ij}$ produces a scalar value per edge, i.e., $\mathbb{R}^d \rightarrow \mathbb{R}$, which could for example be the vector length (with optional clamping for a maximum distance as done in language models), which gives

$\alpha_{ij} = \langle Q_i, K_j \rangle + e(\epsilon_{ij})$

This edge attention function $e$ can also be an arbitrary MLP, or learned weighting matrix, with an optional position encoder, e.g., using a logarithmic (SWIN v2) or nearest neighbor (SWIN v1) embedding (which is especially useful for cartesian data). 

## CConv attention

If we go with the CConv approach we could utilize a fourier, or polynomial, embedding of the relative distance between tokens and a learned weighting per embedding, i.e.,

$$b^x_{ij} = [1, \cos \pi \epsilon_{ij}^x, \sin \pi \epsilon_{ij}^x,\dots\sin k\pi\epsilon{ij}^x],$$

and analogously for the y-component, which are then combined in an outer product

$$B_{ij} = b^x \otimes b^y$$

And then weighed based on a learned matrix

$$ e_{ij} = \sum_x\sum_y W_C^{x,y} B^{x,y}_{ij}$$

Which can then be added on the attention score

## Windowed attention/Masking

In addition to a direct edge bias, we could also apply windowing functions, such as an SPH kernel, that reduces the attention naturally between distant nodes, e.g.,

$$\alpha_{ij} = \left(\langle Q_i, K_j \rangle\right)W(\epsilon_{ij},h) $$

based on a clamping distance/cutoff-radius after which the pairwise attention becomes zero. This would also naturally be reflected in the adjacency as this naturally reduces the adjacency to an SPH-like neighborhood.

This is similar to masking where certain components or features are set to have no influence on the attention, e.g., if an input channel has no meaning for this particular PDE.

## Biased Convolution

Another application of CConv like approaches is computing the Basis Tensor $B_{ij}$ as an input to an MLP (or linear layer) that produces $W_Q$ and $W_K$, i.e., conditioning the query and key matrices strongly on the relative positions in an encoded manner, or straight by using an MLP to map from $\epsilon_{ij}$ straight to $W_Q$ and $W_K$ (algebraically very similar to GNNs)

## Message Passing GAT

Another approach to include Edge information is after the attention mechanism, e.g., instead of using a message passing network that only considers $w_{ij}$ and $V_j$, multiplying the resulting message directly with a linear mapping of the edge features, e.g.,

$$m_{ij} = \left(W_e \epsilon_{ij}\right)\left(w_{ij} V_j\right)$$

Gives a direct influence of edge features on the messages. 

## Extensions

This could obviously be done in any normal GNN manner, e.g., using MLPs to compute the edge weighting, throwing everything into an MLP, using CConvs for the weighting, etc., which is a trivial extension but at least using CConv style approaches, or window functions, hasnt been done so far. Mathematically this might be convenient with a scaled attention using a shephard filter as this mathematically becomes very close to a proper convolutional operator that fulfills normal convolution properties.
