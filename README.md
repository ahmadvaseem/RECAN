# RECAN

we propose a new method that utilizes not only internal image features but also multi-level edge prior knowledge with richer information. Holding the 
intuition that edge information helps to deal with blurry edges and try to generate sharper results, we present a residual edge and channel attention super-
resolution network to handle LR images, named RECAN.
Our architecture consists of two basic modules: the first module is EdgeNet, which generates multi-level edge maps from the input image; and the second 
module takes advantage of significant information in input image along with edge maps, called SRNet. Specifically, the SRNet uses channel attention 
technique and spatial feature transform (SFT) layers to super-resolve an image. 
