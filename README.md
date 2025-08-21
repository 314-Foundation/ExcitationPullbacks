# Excitation Pullbacks - faithful? explanations of ReLU networks

This repository contains the source code for the paper [Tapping into the Black Box: Uncovering Aligned Representations in Pretrained Neural Networks](https://www.arxiv.org/abs/2507.22832). We describe a powerful novel explanation method for Deep Neural Networks (the so-called *Excitation Pullbacks*) and argue for it's faithfulness.

To recreate the results run the `pullbacks.ipynb` notebook in the adequate env, see `requirements.txt`.

## Demo app

Also see the publicitly available interactive app on Huggingface Spaces: [https://huggingface.co/spaces/msat/ExcitationPullbacks](https://huggingface.co/spaces/msat/ExcitationPullbacks) 

## Background

We argue that excitation pullbacks are gradients of a kernel machine that is implicitly learned by the network and encoded in its *highly activated paths*. We claim that this kernel machine mainly supports the network's decision boundary. For details see our paper.

## Visualising gradients for various pretrained architectures

To visualise the pullbacks we perform a rudimentary 5-step pixel-space gradient ascent guided by excitation pullbacks and vanilla gradients, respectively. We do this for 3 popular ImageNet-pretrained ReLU architectures: ResNet50, VGG11_BN and DenseNet121. While vanilla gradients are noisy, excitation pullbacks reveal compelling label-specific features that "just make sense". 

Specifically, in images below, each cell shows the difference between the perturbed and clean image, targeting the class in the column. Diagonal: original class; off-diagonal: counterfactuals. Last column: randomly selected extra label.

Notice that excitation pullbacks tend to highlight similar features across architectures, which suggests that the models learn comparable feature representations. Also, the structure of the excitation pullbacks intuitively reflects the internal organization of each network, reinforcing our hypothesis that they indeed faithfully capture the underlying decision process of the model.

Excitation pullbacks for ResNet50:
![img](./media/pullback_diff/resnet50_alpha_20_steps_5.jpg)

Excitation pullbacks for VGG11_BN:
![img](./media/pullback_diff/vgg11_bn_alpha_20_steps_5.jpg)

Excitation pullbacks for DenseNet121:
![img](./media/pullback_diff/densenet121_alpha_20_steps_5.jpg)

The vanilla gradients for all the models look like noise, e.g.

Vanilla gradients for ResNet50:
![img](./media/vanilla_grad_diff/resnet50_alpha_20_steps_5.jpg)

<!-- Excitation pullbacks for ResNet50:
![img](./media/pullback/resnet50_alpha_20_steps_10.jpg)

Excitation pullbacks for VGG11_BN:
![img](./media/pullback/vgg11_bn_alpha_20_steps_10.jpg)

Excitation pullbacks for DenseNet121:
![img](./media/pullback/densenet121_alpha_20_steps_10.jpg) -->