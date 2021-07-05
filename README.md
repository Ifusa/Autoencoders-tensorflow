# Autoencoders-tensorflow
An autoencoder is a type of artificial neural network used to learn efficient codings of unlabeled data (unsupervised learning).[1] The encoding is validated and refined by attempting to regenerate the input from the encoding. The autoencoder learns a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore insignificant data (“noise”). 
Autoencoders here are tried on mnist data.
An autoencoder has two main parts: an encoder that maps the input into the code, and a decoder that maps the code to a reconstruction of the input.

The simplest way to perform the copying task perfectly would be to duplicate the signal. Instead, autoencoders are typically forced to reconstruct the input approximately, preserving only the most relevant aspects of the data in the copy.

The idea of autoencoders has been popular for decades. The first applications date to the 1980s.[2][8][9] Their most traditional application was dimensionality reduction or feature learning, but the concept became widely used for learning generative models of data.[10][11] Some of the most powerful AIs in the 2010s involved autoencoders stacked inside deep neural networks.[12]
Schema of a basic Autoencoder

The simplest form of an autoencoder is a feedforward, non-recurrent neural network similar to single layer perceptrons that participate in multilayer perceptrons (MLP) – employing an input layer and an output layer connected by one or more hidden layers. The output layer has the same number of nodes (neurons) as the input layer. Its purpose is to reconstruct its inputs (minimizing the difference between the input and the output) instead of predicting a target value Y {\displaystyle Y} Y given inputs X {\displaystyle X} X. Therefore, autoencoders learn unsupervised.

An autoencoder consists of two parts, the encoder and the decoder, which can be defined as transitions ϕ {\displaystyle \phi } \phi and ψ , {\displaystyle \psi ,} {\displaystyle \psi ,} such that:

    ϕ : X → F {\displaystyle \phi :{\mathcal {X}}\rightarrow {\mathcal {F}}} {\displaystyle \phi :{\mathcal {X}}\rightarrow {\mathcal {F}}}
    ψ : F → X {\displaystyle \psi :{\mathcal {F}}\rightarrow {\mathcal {X}}} {\displaystyle \psi :{\mathcal {F}}\rightarrow {\mathcal {X}}}
    ϕ , ψ = a r g m i n ϕ , ψ ‖ X − ( ψ ∘ ϕ ) X ‖ 2 {\displaystyle \phi ,\psi ={\underset {\phi ,\psi }{\operatorname {arg\,min} }}\,\|{\mathcal {X}}-(\psi \circ \phi ){\mathcal {X}}\|^{2}} {\displaystyle \phi ,\psi ={\underset {\phi ,\psi }{\operatorname {arg\,min} }}\,\|{\mathcal {X}}-(\psi \circ \phi ){\mathcal {X}}\|^{2}}

In the simplest case, given one hidden layer, the encoder stage of an autoencoder takes the input x ∈ R d = X {\displaystyle \mathbf {x} \in \mathbb {R} ^{d}={\mathcal {X}}} {\displaystyle \mathbf {x} \in \mathbb {R} ^{d}={\mathcal {X}}} and maps it to h ∈ R p = F {\displaystyle \mathbf {h} \in \mathbb {R} ^{p}={\mathcal {F}}} {\displaystyle \mathbf {h} \in \mathbb {R} ^{p}={\mathcal {F}}}:

    h = σ ( W x + b ) {\displaystyle \mathbf {h} =\sigma (\mathbf {Wx} +\mathbf {b} )} {\displaystyle \mathbf {h} =\sigma (\mathbf {Wx} +\mathbf {b} )}

This image h {\displaystyle \mathbf {h} } {\mathbf {h}} is usually referred to as code, latent variables, or a latent representation. σ {\displaystyle \sigma } \sigma is an element-wise activation function such as a sigmoid function or a rectified linear unit. W {\displaystyle \mathbf {W} } \mathbf {W} is a weight matrix and b {\displaystyle \mathbf {b} } \mathbf {b} is a bias vector. Weights and biases are usually initialized randomly, and then updated iteratively during training through backpropagation. After that, the decoder stage of the autoencoder maps h {\displaystyle \mathbf {h} } {\mathbf {h}} to the reconstruction x ′ {\displaystyle \mathbf {x'} } \mathbf {x'} of the same shape as x {\displaystyle \mathbf {x} } \mathbf {x} :

    x ′ = σ ′ ( W ′ h + b ′ ) {\displaystyle \mathbf {x'} =\sigma '(\mathbf {W'h} +\mathbf {b'} )} {\displaystyle \mathbf {x'} =\sigma '(\mathbf {W'h} +\mathbf {b'} )}

where σ ′ , W ′ ,  and  b ′ {\displaystyle \mathbf {\sigma '} ,\mathbf {W'} ,{\text{ and }}\mathbf {b'} } {\displaystyle \mathbf {\sigma '} ,\mathbf {W'} ,{\text{ and }}\mathbf {b'} } for the decoder may be unrelated to the corresponding σ , W ,  and  b {\displaystyle \mathbf {\sigma } ,\mathbf {W} ,{\text{ and }}\mathbf {b} } {\displaystyle \mathbf {\sigma } ,\mathbf {W} ,{\text{ and }}\mathbf {b} } for the encoder.

Autoencoders are trained to minimise reconstruction errors (such as squared errors), often referred to as the "loss":

    L ( x , x ′ ) = ‖ x − x ′ ‖ 2 = ‖ x − σ ′ ( W ′ ( σ ( W x + b ) ) + b ′ ) ‖ 2 {\displaystyle {\mathcal {L}}(\mathbf {x} ,\mathbf {x'} )=\|\mathbf {x} -\mathbf {x'} \|^{2}=\|\mathbf {x} -\sigma '(\mathbf {W'} (\sigma (\mathbf {Wx} +\mathbf {b} ))+\mathbf {b'} )\|^{2}} {\displaystyle {\mathcal {L}}(\mathbf {x} ,\mathbf {x'} )=\|\mathbf {x} -\mathbf {x'} \|^{2}=\|\mathbf {x} -\sigma '(\mathbf {W'} (\sigma (\mathbf {Wx} +\mathbf {b} ))+\mathbf {b'} )\|^{2}}

where x {\displaystyle \mathbf {x} } \mathbf {x} is usually averaged over the training set.

As mentioned before, autoencoder training is performed through backpropagation of the error, just like other feedforward neural networks.

Should the feature space F {\displaystyle {\mathcal {F}}} {\mathcal {F}} have lower dimensionality than the input space X {\displaystyle {\mathcal {X}}} {\mathcal {X}}, the feature vector ϕ ( x ) {\displaystyle \phi (x)} \phi (x) can be regarded as a compressed representation of the input x {\displaystyle x} x. This is the case of undercomplete autoencoders. If the hidden layers are larger than (overcomplete), or equal to, the input layer, or the hidden units are given enough capacity, an autoencoder can potentially learn the identity function and become useless. However, experimental results found that overcomplete autoencoders might still learn useful features.[13] In the ideal setting, the code dimension and the model capacity could be set on the basis of the complexity of the data distribution to be modeled. One way to do so is to exploit the model variants known as Regularized Autoencoders.[2] 
