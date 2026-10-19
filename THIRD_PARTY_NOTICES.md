# Third-party sources and attribution

## H-RDT

The action architecture in `src/ego_action_lab/model.py` is an MLX adaptation of concepts/code from [HongzheBi/H_RDT](https://github.com/HongzheBi/H_RDT), inspected at commit `8be02cb5e631898dfab12f9b6bbb77e32999547a`. The upstream project distributes its code under the Mozilla Public License 2.0; this repository retains MPL-2.0 in `LICENSE`.

Adaptations include a 131M configuration, MLX attention/normalization and module layout, masked 48-dimensional bimanual state/trajectory interfaces, frozen 196×384 image tokens, a learned task token, and residual Flow Matching training/evaluation. This is not a claim of equivalence to the official full model, official results, or robot transfer experiments. Upstream copyright and license rights are retained.

## DINOv3 / timm checkpoint

The frozen encoder implements DINOv3 ViT-S/16 inference with timm-compatible parameter names. References: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3), [timm checkpoint](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m). Reference checkpoint revision: `3bf4720a82ec2066db88137180ff1f83a675cef0`.

No DINOv3 weights, downloaded model card, or upstream DINOv3 repository are bundled. The repository's MPL-2.0 license does not license third-party model weights; obtain and use them under their own terms. The MLX inference implementation was extracted from the author's local experimental project.

## EgoDex

[EgoDex](https://github.com/apple-aiml-research/ml-egodex) supplies the data schema targeted by preprocessing. No dataset videos, motion annotations, participant data or official split files are redistributed. The demo generator creates artificial trajectories and color frames locally. Follow the original data terms when obtaining, using or sharing real data.

## Original research code

Geometry masks, sampling metrics, MLX model/encoder execution and checkpoint utilities were extracted from the author's EGO experiments and reorganized into a standalone package. Portable data preparation, configuration, CLI, synthetic examples, tests and documentation were added for this release. No private experiment logs, unpublished paper drafts, machine credentials, local workspace paths or model checkpoints are included.
