# Monocular Depth Project for CIL, ETH Zurich (Spring 2025)

For the following 5 models the provided example notebook on kaggle has been used as a base and we have mainly adjusted the model definition for the various experiments as well as added a function for the scale-invariant [RMSE loss](https://github.com/TheHummel/cil-monocular-depth/blob/master/training/loss.py). We now give a short description on the model, and where they are located.

**For the review: find links to the lines of code where the model definition is located. Main method, image transforms and training method remain largely unchanged in comparison to the kaggle template**

## Baseline model 1: Basic UNet

The basic Unet is a bigger version of the UNet given in the provided example on kaggle with 4 Encoder, Decoder blocks. The associated file can be found in the models section as [basic_unet.py](https://github.com/TheHummel/cil-monocular-depth/blob/master/models/basic_unet.py) with the model defintion starting on [line 127](https://github.com/TheHummel/cil-monocular-depth/blob/master/models/basic_unet.py#L127-L211). To run the training, simply execute this file.

## Baseline model 2: MiDaS Decoder Finetuned:

This is the pre-trained model from huggingface where we finetuned the decoder on the dataset. In our report we made use of both the MiDaS version with [ViT backbone](https://huggingface.co/Intel/dpt-hybrid-midas) and the [Swinv2 backbone](https://huggingface.co/Intel/dpt-swinv2-base-384). Simply execute the file [finetune_midas_decoder.py](https://github.com/TheHummel/cil-monocular-depth/blob/master/models/finetune_midas_decoder.py) in models to run the training for the swinv2 backbone. We omit the finetuned version for the ViT backbone since it is a bit worse than with the swinv2 backbone and we used the swinv2 tuned version as reference baseline in the report. Model definitions starts on [line 126](https://github.com/TheHummel/cil-monocular-depth/blob/master/models/finetune_midas_decoder.py#L126-L148).


## 1. Variant: UNet with MiDaS-Encoder features:

This model takes a 4 layer UNet as a base and additionaly runs the input through the frozen encoder layer of the MiDaS model and fuses MiDaS features from different encodder stages into the UNet decoder. To run the training with the Swinv2 backbone execute file [unet_plus_midas_swin.py](https://github.com/TheHummel/cil-monocular-depth/blob/master/models/unet_plus_midas_swin.py), located in the models. For the ViT backbone, run [unet_plus_midas_vit.py](https://github.com/TheHummel/cil-monocular-depth/blob/master/models/unet_plus_midas_vit.py). Model definitions for Swinv2 backbone is [here](https://github.com/TheHummel/cil-monocular-depth/blob/master/models/unet_plus_midas_swin.py#L129-L277) and for ViT backbone [here](https://github.com/TheHummel/cil-monocular-depth/blob/master/models/unet_plus_midas_vit.py#L128-L271). The following diagram depicts the model architecture (with tensor values in the swinv2 version) to help with understanding. 
![UNet+MiDaS-Enc](./images/UNetPlusMidas.png)

## 2. Variant: MiDaS with full skip connection network:

This is inspired by the paper [Rethinking Skip Connections in Encoder-decoder Networks for Monocular Depth Estimation](https://arxiv.org/abs/2208.13441). The implementation replaces the FusionBlocks in the MiDaS Neck layer (model.dpt.neck) with a custom FusionBlock that performs a more complex feature fusion of different encoder hidden_states. The model is located in [models/dpt_hybrid_midas](https://github.com/TheHummel/cil-monocular-depth/tree/master/models/dpt_hybrid_midas). To run this model execute the run_midas_fscn located in [slurm_scripts](https://github.com/TheHummel/cil-monocular-depth/blob/master/slurm_scripts/run_midas_fscn.sh)

## 3. Variant (Combination): UNet with MiDaS-Encoder features and FSCN:

This model is the combination of the two variants, where instead of just adding one hidden_state from the MiDaS encoder into the UNet decoder, we take all selected hidden_states from the MiDaS encoder and combine them before fusing them into the UNet pipeline. The code to train this model is located in [unet_plus_midas_vit_and_fscn.py](https://github.com/TheHummel/cil-monocular-depth/blob/master/models/unet_plus_midas_vit_and_fscn.py) under models. Model definitions are located on [line 144](https://github.com/TheHummel/cil-monocular-depth/blob/master/models/unet_plus_midas_vit_and_fscn.py#L144-L380) The following diagram gives a rough overview.
![UNet+Midas-Enc with FSCN](./images/UNetPlusMidasWithFSCN.png)
