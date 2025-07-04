<div align="center">

# No Time to Train!
## Training-Free Reference-Based Instance Segmentation

[Arxiv Paper](https://arxiv.org/abs/2507.02798) | [Project Page](https://miquel-espinosa.github.io/no-time-to-train/)



</div>

> _The performance of image segmentation models has historically been constrained by the high cost of collecting large-scale annotated data. The Segment Anything Model (SAM) alleviates this original problem through a promptable, semantics-agnostic, segmentation paradigm and yet still requires manual visual-prompts or complex domain-dependent prompt-generation rules to process a new image. Towards reducing this new burden, our work investigates the task of object segmentation when provided with, alternatively, only a small set of reference images. Our key insight is to leverage strong semantic priors, as learned by foundation models, to identify corresponding regions between a reference and a target image. We find that correspondences enable automatic generation of instance-level segmentation masks for downstream tasks and instantiate our ideas via a multi-stage, training-free method incorporating (1) memory bank construction; (2) representation aggregation and (3) semantic-aware feature matching. Our experiments show significant improvements on segmentation metrics, leading to state-of-the-art performance on COCO FSOD (36.8% nAP), PASCAL VOC Few-Shot (71.2% nAP50) and outperforming existing training-free approaches on the Cross-Domain FSOD benchmark (22.4% nAP)._


## Citation
If you find this work helpful please consider citing
```
@article{espinosa2025notimetotrain,
        title={No time to train! Training-Free Reference-Based Instance Segmentation},
        author={Miguel Espinosa and Chenhongyi Yang and Linus Ericsson and Steven McDonagh and Elliot J. Crowley},
        year={2025}
        journal={arXiv},
        primaryclass={cs.CV},
        url={https://arxiv.org/abs/2507.01300}
}
```
