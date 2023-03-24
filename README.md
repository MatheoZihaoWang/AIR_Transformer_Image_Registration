# AiR: Attention-based Deformable Image Registration using Transformer

This is a repository for the AiR method, an attention-based deformable image registration method that uses the Transformer framework. Image registration is an important basis in signal processing tasks, but often encounters problems with stability and efficiency. Non-learning registration approaches rely on optimizing the similarity metrics between the fix and moving images, which can be costly in terms of time and space complexity, especially for large images or severe deformations.

Recently, convolutional neural network (CNN) based image registration methods have been investigated and show promising effectiveness in overcoming the weaknesses of non-learning based methods. However, to explore advanced learning approaches for solving practical issues in image registration, we propose a method that introduces attention mechanisms in deformable image registration problems.

Our approach learns an unsupervised generated deformation map using the Transformer framework, which does not rely on the CNN but can be efficiently trained on GPGPU devices. In a more vivid interpretation, we treat the image registration problem as a language translation task and introduce a Transformer to tackle the problem. We test our method on two benchmark datasets and show its effectiveness.

## Data Setup
processor: Create an instance of Preprocessing.
alginment(): Align the fix and mov images for all stacks.
patchlization_for_training(): Create the patches for training.
patchlization_for_testing(): Create the patches for testing.
normalization(): Normalize the images.
tensor_fix_train, tensor_mov_train, tensor_fix_test, tensor_mov_test: Convert the numpy arrays to tensors.
train_pr, test_pr: Create instances of imageHandle with the training and testing tensors.


We would like to thank you for citing the paper:

@misc{wang2021attention,
      title={Attention for Image Registration (AiR): an unsupervised Transformer approach}, 
      author={Zihao Wang and Hervé Delingette},
      year={2021},
      eprint={2105.02282},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Wang, Zihao, and Hervé Delingette. "Attention for image registration (air): an unsupervised transformer approach." arXiv preprint arXiv:2105.02282 (2021).
Wang Z, Delingette H. Attention for image registration (air): an unsupervised transformer approach[J]. arXiv preprint arXiv:2105.02282, 2021. MLA	
