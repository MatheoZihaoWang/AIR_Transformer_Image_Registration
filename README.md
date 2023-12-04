# AiR: Attention-based Deformable Image Registration using Transformer

This is a repository for the AiR method, an attention-based deformable image registration method that uses the Transformer framework. Image registration is an important basis in signal processing tasks, but often encounters problems with stability and efficiency. Non-learning registration approaches rely on optimizing the similarity metrics between the fix and moving images, which can be costly in terms of time and space complexity, especially for large images or severe deformations.

Our approach learns an unsupervised generated deformation map using the Transformer framework, which does not rely on the CNN but can be efficiently trained on GPGPU devices. In a more vivid interpretation, we treat the image registration problem as a language translation task and introduce a Transformer to tackle the problem. We test our method on two benchmark datasets and show its effectiveness.

## Data Setup
processor: Create an instance of Preprocessing.

alginment(): Align the fix and mov images for all stacks.

patchlization_for_training(): Create the patches for training.

patchlization_for_testing(): Create the patches for testing.

normalization(): Normalize the images.

tensor_fix_train, tensor_mov_train, tensor_fix_test, tensor_mov_test: Convert the numpy arrays to tensors.

train_pr, test_pr: Create instances of imageHandle with the training and testing tensors.

```python main_brain.py```

### We would like to thank you for citing the paper:

@misc{wang2021attention,
      title={Attention for Image Registration (AiR): an unsupervised Transformer approach}, 
      author={Zihao Wang and Hervé Delingette},
      year={2021},
      eprint={2105.02282},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Wang, Zihao, and Hervé Delingette. "Attention for image registration (air): an unsupervised transformer approach." arXiv preprint arXiv:2105.02282 (2021).


A following up work is in 

Wang, Z., Yang, Y., Sermesant, M., Delingette, H. (2022). Unsupervised Echocardiography Registration Through Patch-Based MLPs and Transformers. In: Camara, O., et al. Statistical Atlases and Computational Models of the Heart. Regular and CMRxMotion Challenge Papers. STACOM 2022. Lecture Notes in Computer Science, vol 13593. Springer, Cham. https://doi.org/10.1007/978-3-031-23443-9_16

@InProceedings{10.1007/978-3-031-23443-9_16,
author="Wang, Zihao
and Yang, Yingyu
and Sermesant, Maxime
and Delingette, Herv{\'e}",
title="Unsupervised Echocardiography Registration Through Patch-Based MLPs and Transformers",
booktitle="Statistical Atlases and Computational Models of the Heart. Regular and CMRxMotion Challenge Papers",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="168--178",
isbn="978-3-031-23443-9"
}

Source code is in
https://gitlab.inria.fr/epione/mlp_transformer_registration
