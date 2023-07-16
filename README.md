# Awesome-Forgetting-in-Deep-Learning

[![Awesome](https://awesome.re/badge.svg)]()
<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt=""/>

A comprehensive list of papers about **'Forgetting in Deep Learning'**.

## Abstract
> Forgetting refers to the loss or deterioration of previously acquired information or knowledge. While the existing surveys on forgetting have primarily focused on continual learning, forgetting exists in various other research areas within deep learning. Forgetting manifests in research fields such as generative models due to generator shifts, and federated learning due to heterogeneous data distributions across clients. Addressing forgetting encompasses several challenges, including balancing the retention of old task knowledge with fast learning of new tasks, managing task interference with conflicting goals, and preventing privacy leakage, etc.
Moreover, most existing surveys on continual learning implicitly assume that forgetting is always harmful. In contrast, our survey argues that forgetting is a double-edged sword and can be beneficial and desirable in certain cases, such as privacy-preserving scenarios. By exploring forgetting in a broader context, we aim to present a more nuanced understanding of this phenomenon and highlight its potential advantages. Through this comprehensive review, we aspire to uncover potential solutions by drawing upon approaches from various fields that have dealt with forgetting. By examining forgetting beyond its conventional boundaries, we hope to develop novel strategies for mitigating, harnessing, or even embracing forgetting in real applications.

## Citation

If you find this resource is helpful, please consider cite:  
```
@article{Forgetting_Survey_2023,
  title={A Comprehensive Survey of Forgetting in Deep Learning Beyond Continual Learning},
  author={Zhenyi Wang, Enneng Yang, Li Shen, Heng Huang},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```
Thanks!

******

## Framework

  * [Harmful Forgetting](#harmful-forgetting)
    + [Forgetting in Continual Learning](#forgetting-in-continual-learning)
      - [Task-aware CL](#task-aware-cl)
        * [Memory-based Methods](#memory-based-methods)
        * [Architecture-based Methods](#architecture-based-methods)
        * [Regularization-based Methods](#regularization-based-methods)
        * [Subspace-based Methods](#subspace-based-methods)
        * [Bayesian Methods](#bayesian-methods)
      - [Task-free CL](#task-free-cl)
      - [Online CL](#online-cl)
      - [Semi-supervised CL](#semi-supervised-cl)
      - [Few-shot CL](#few-shot-cl)
      - [Unsupervised CL](#unsupervised-cl)
      - [Theoretical Analysis](#theoretical-analysis)
    + [Forgetting in Foundation Models](#forgetting-in-foundation-models)
      - [Forgetting in Fine-Tuning Foundation Models](#forgetting-in-fine-tuning-foundation-models)
      - [Forgetting in One-Epoch Pre-training](#forgetting-in-one-epoch-pre-training)
      - [CL in Foundation Model](#cl-in-foundation-model)
    + [Forgetting in Domain Adaptation](#forgetting-in-domain-adaptation)
    + [Forgetting in Test-Time Adaptation](#forgetting-in-test-time-adaptation)
    + [Forgetting in Meta-Learning](#forgetting-in-meta-learning)
      - [Incremental Few-Shot Learning](#incremental-few-shot-learning)
      - [Continual Meta-Learning](#continual-meta-learning)
    + [Forgetting in Generative Models](#forgetting-in-generative-models)
      - [GAN Training is a Continual Learning Problem](#gan-training-is-a-continual-learning-problem)
      - [Lifelong Learning of Generative Models](#lifelong-learning-of-generative-models)
    + [Forgetting in Reinforcement Learning](#forgetting-in-reinforcement-learning)
    + [Forgetting in Federated Learning](#forgetting-in-federated-learning)
      - [Forgetting Due to Non-IID Data in FL  ](#forgetting-due-to-non-iid-data-in-fl)
      - [Federated Continual Learning](#federated-continual-learning)
  * [Beneficial Forgetting](#beneficial-forgetting)
    + [Forgetting Irrelevant Information to Achieve Better Performance](#forgetting-irrelevant-information-to-achieve-better-performance)
      - [Combat Overfitting Through Forgetting](#combat-overfitting-through-forgetting)
      - [Learning New Knowledge Through Forgetting Previous Knowledge](#learning-new-knowledge-through-forgetting-previous-knowledge)
    + [Machine Unlearning](#machine-unlearning)



******


## Harmful Forgetting

Harmful forgetting occurs when we desire the machine learning model to retain previously learned knowledge while adapting to new tasks, domains, or environments. In such cases, it is important to prevent and mitigate knowledge forgetting.

| **Problem Setting** | **Goal** | **Source of forgetting** |
| --------------- | :---- | :---- |
| Continual Learning | non-stationary data distribution without forgetting previous knowledge  | data-distribution shift during training |
| Foundation Models |unsupervised learning on large-scale unlabeled data | data-distribution shift in pre-training, fine-tuning  |
| Domain Adaptation | adapt to target domain while maintaining performance on source domain | target domain sequentially shift over time |
| Meta-Learning | learn adaptable knowledge to new tasks | incrementally meta-learn new classes / task-distribution shift  |
| Generative Model | learn a generator to appriximate real data distribution | generator shift/data-distribution shift |
| Self-Supervised Learning | unsupervised pre-training | data-distribution shift |
| Reinforcement Learning | aximize accumulate rewards | state, action, reward and state transition dynamics|
| Federated Learning | decentralized training without sharing data |  model average; non-i.i.d data; data-distribution shift |

**Links**:
<u> [Forgetting in Continual Learning](#forgetting-in-continual-learning) </u> |
<u> [Forgetting in Foundation Models](#forgetting-in-foundation-models) </u> |
<u> [Forgetting in Domain Adaptation](#forgetting-in-domain-adaptation)</u> |
<u> [Forgetting in Test-Time Adaptation](#forgetting-in-test-time-adaptation)</u> |  
<u> [Forgetting in Meta-Learning](#forgetting-in-meta-learning) </u>|  
<u> [Forgetting in Generative Models](#forgetting-in-generative-models) </u>|
<u> [Forgetting in Reinforcement Learning](#forgetting-in-reinforcement-learning)</u> |
<u> [Forgetting in Federated Learning](#forgetting-in-federated-learning)</u>


----------
### Forgetting in Continual Learning


> The goal of continual learning  (CL) is to learn on a sequence of tasks without forgetting the knowledge on previous tasks.

**Links**:
<u> [Task-aware CL](#task-aware-cl)  </u>|
<u> [Task-free CL](#task-free-cl)  </u>|
<u> [Online CL](#online-cl)  </u>|
<u> [Semi-supervised CL](#semi-supervised-cl)  </u>|
<u> [Few-shot CL](#few-shot-cl)  </u>|
<u> [Unsupervised CL](#unsupervised-cl)  </u>|
<u> [Theoretical Analysis](#theoretical-analysis) </u>

#### Task-aware CL
> Task-aware CL focuses on addressing scenarios where explicit task definitions, such as task IDs or labels, are available during the CL process. Existing methods on task-aware CL have explored five main branches:   [Memory-based Methods](#memory-based-methods) |
  [Architecture-based Methods](#architecture-based-methods) |
  [Regularization-based Methods](#regularization-based-methods) |
  [Subspace-based Methods](#subspace-based-methods) |
  [Bayesian Methods](#bayesian-methods).

#####  Memory-based Methods
> Memory-based method keeps a memory buffer that stores the examples/knowledges from previous tasks and replay those examples during learning new tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Error Sensitivity Modulation based Experience Replay: Mitigating Abrupt Representation Drift in Continual Learning](https://openreview.net/pdf?id=zlbci7019Z3) |2023 | ICLR
|[A Model or 603 Exemplars: Towards Memory-Efficient Class-Incremental Learning](https://arxiv.org/pdf/2205.13218.pdf)| 2023 | ICLR
| [DualHSIC: HSIC-Bottleneck and Alignment for Continual Learning](https://arxiv.org/pdf/2305.00380.pdf) | 2023 | ICML
| [Regularizing Second-Order Influences for Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Regularizing_Second-Order_Influences_for_Continual_Learning_CVPR_2023_paper.pdf) | 2023|CVPR
| [Class-Incremental Exemplar Compression for Class-Incremental Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Luo_Class-Incremental_Exemplar_Compression_for_Class-Incremental_Learning_CVPR_2023_paper.pdf) | 2023|CVPR
| [Class-Incremental Learning using Diffusion Model for Distillation and Replay](https://arxiv.org/pdf/2306.17560.pdf) | 2023 | Arxiv
| [On the Effectiveness of Lipschitz-Driven Rehearsal in Continual Learning](https://openreview.net/pdf?id=TThSwRTt4IB) | 2022 | NeurIPS
| [Exploring Example Influence in Continual Learning](https://openreview.net/pdf?id=u4dXcUEsN7B) | 2022 | NeurIPS
| [Navigating Memory Construction by Global Pseudo-Task Simulation for Continual Learning](https://openreview.net/pdf?id=tVbJdvMxK2-) | 2022 | NeurIPS
| [Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System](https://openreview.net/pdf?id=uxxFrDwrE7Y) | 2022 | ICLR
| [Information-theoretic Online Memory Selection for Continual Learning](https://openreview.net/pdf?id=IpctgL7khPp) | 2022 | ICLR
| [Memory Replay with Data Compression for Continual Learning](https://openreview.net/pdf?id=a7H7OucbWaU) | 2022 | ICLR
| [Improving Task-free Continual Learning by Distributionally Robust Memory Evolution](https://proceedings.mlr.press/v162/wang22v/wang22v.pdf) | 2022 | ICML
| [GCR: Gradient Coreset based Replay Buffer Selection for Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Tiwari_GCR_Gradient_Coreset_Based_Replay_Buffer_Selection_for_Continual_Learning_CVPR_2022_paper.pdf) | 2022 | CVPR
| [RMM: Reinforced Memory Management for Class-Incremental Learning](https://proceedings.neurips.cc/paper_files/paper/2021/file/1cbcaa5abbb6b70f378a3a03d0c26386-Paper.pdf) | 2021 | NeurIPS
| [Rainbow Memory: Continual Learning with a Memory of Diverse Samples](https://openaccess.thecvf.com/content/CVPR2021/papers/Bang_Rainbow_Memory_Continual_Learning_With_a_Memory_of_Diverse_Samples_CVPR_2021_paper.pdf) | 2021|CVPR
| [Always Be Dreaming: A New Approach for Data-Free Class-Incremental Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Smith_Always_Be_Dreaming_A_New_Approach_for_Data-Free_Class-Incremental_Learning_ICCV_2021_paper.pdf) | 2021 | ICCV
| [Using Hindsight to Anchor Past Knowledge in Continual Learning](https://arxiv.org/pdf/2002.08165.pdf) | 2021 | AAAI
| [Improved Schemes for Episodic Memory-based Lifelong Learning](https://proceedings.neurips.cc/paper/2020/file/0b5e29aa1acf8bdc5d8935d7036fa4f5-Paper.pdf) | 2020 |NeurIPS
| [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf) | 2020 |NeurIPS
| [La-MAML: Look-ahead Meta Learning for Continual Learning](https://proceedings.neurips.cc/paper/2020/file/85b9a5ac91cd629bd3afe396ec07270a-Paper.pdf) | 2020 | NeurIPS
| [Brain-inspired replay for continual learning with artificial neural networks](https://pubmed.ncbi.nlm.nih.gov/32792531/) |2020 |Nature Communications
| [LAMOL: LAnguage MOdeling for Lifelong Language Learning](https://openreview.net/pdf?id=Skgxcn4YDS) | 2020 |ICLR
| [Mnemonics Training: Multi-Class Incremental Learning without Forgetting](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Mnemonics_Training_Multi-Class_Incremental_Learning_Without_Forgetting_CVPR_2020_paper.pdf) | 2020 | CVPR
| [GDumb: A Simple Approach that Questions Our Progress in Continual Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf) |2020| ECCV
| [Continual Learning with Tiny Episodic Memories](https://arxiv.org/pdf/1902.10486.pdf) | 2019 | ICML |
| [Efficient lifelong learning with A-GEM](https://openreview.net/pdf?id=Hkf2_sC5FX) | 2019 | ICLR |
| [Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference](https://openreview.net/pdf?id=B1gTShAct7) | 2019 |ICLR
| [Large Scale Incremental Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf) | 2019 | CVPR
| [On Tiny Episodic Memories in Continual Learning](https://arxiv.org/pdf/1902.10486.pdf) | 2019 | Arxiv
| [Progress & Compress: A scalable framework for continual learning](https://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf) | 2018 | ICML
| [Gradient Episodic Memory for Continual Learning](https://proceedings.neurips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf) | 2017 |NeurIPS
| [Continual Learning with Deep Generative Replay](https://proceedings.neurips.cc/paper_files/paper/2017/file/0efbe98067c6c73dba1250d2beaa81f9-Paper.pdf) | 2017 |NeurIPS
| [iCaRL: Incremental Classifier and Representation Learning](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf) | 2017| CVPR



#####  Architecture-based Methods
> The architecture-based approach avoids forgetting by reducing parameter sharing between tasks or adding parameters to new tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Parameter-Level Soft-Masking for Continual Learning](https://openreview.net/pdf?id=wxFXvPdVqi) | 2023 | ICML
| [Continual Learning on Dynamic Graphs via Parameter Isolation](https://arxiv.org/pdf/2305.13825.pdf) | 2023 | SIGIR
| [Heterogeneous Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Madaan_Heterogeneous_Continual_Learning_CVPR_2023_paper.pdf) | 2023 | CVPR
| [Dense Network Expansion for Class Incremental Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Dense_Network_Expansion_for_Class_Incremental_Learning_CVPR_2023_paper.pdf) | 2023 | CVPR
| [Achieving a Better Stability-Plasticity Trade-off via Auxiliary Networks in Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_Achieving_a_Better_Stability-Plasticity_Trade-Off_via_Auxiliary_Networks_in_Continual_CVPR_2023_paper.pdf) |2023 | CVPR
| [Forget-free Continual Learning with Winning Subnetworks](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf) | 2022 | ICML
| [NISPA: Neuro-Inspired Stability-Plasticity Adaptation for Continual Learning in Sparse Networks](https://proceedings.mlr.press/v162/gurbuz22a/gurbuz22a.pdf) | 2022 | ICML
| [Continual Learning with Filter Atom Swapping](https://openreview.net/pdf?id=metRpM4Zrcb) | 2022 | ICLR
| [SparCL: Sparse Continual Learning on the Edge](https://proceedings.neurips.cc/paper_files/paper/2022/file/80133d0f6eccaace15508f91e3c5a93c-Paper-Conference.pdf) | 2022 | NeurIPS
| [Learning Bayesian Sparse Networks with Full Experience Replay for Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Yan_Learning_Bayesian_Sparse_Networks_With_Full_Experience_Replay_for_Continual_CVPR_2022_paper.pdf) | 2022 | CVPR
| [FOSTER: Feature Boosting and Compression for Class-Incremental Learning](https://arxiv.org/pdf/2204.04662.pdf) | 2022 | ECCV
| [BNS: Building Network Structures Dynamically for Continual Learning](https://openreview.net/pdf?id=2ybxtABV2Og) | 2021 | NeurIPS
| [DER: Dynamically Expandable Representation for Class Incremental Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_DER_Dynamically_Expandable_Representation_for_Class_Incremental_Learning_CVPR_2021_paper.pdf) | 2021 | CVPR
| [Adaptive Aggregation Networks for Class-Incremental Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Adaptive_Aggregation_Networks_for_Class-Incremental_Learning_CVPR_2021_paper.pdf) |2021 | CVPR
| [BatchEnsemble: an Alternative Approach to Efficient Ensemble and Lifelong Learning](https://openreview.net/pdf?id=Sklf1yrYDr) | 2020 | ICLR
| [Compacting, Picking and Growing for Unforgetting Continual Learning](https://proceedings.neurips.cc/paper/2019/file/3b220b436e5f3d917a1e649a0dc0281c-Paper.pdf) | 2019 | NeurIPS
| [Progress & Compress: A scalable framework for continual learning](https://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf) | 2018 | ICML
| [Overcoming Catastrophic Forgetting with Hard Attention to the Task](https://arxiv.org/pdf/1801.01423.pdf) |2018 | ICML
| [Lifelong Learning with Dynamically Expandable Networks ](https://openreview.net/pdf?id=Sk7KsfW0-) | 2018 | ICLR
| [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.pdf) | 2018 | CVPR
| [Expert Gate: Lifelong Learning with a Network of Experts](https://openaccess.thecvf.com/content_cvpr_2017/papers/Aljundi_Expert_Gate_Lifelong_CVPR_2017_paper.pdf) |2017 | CVPR
| [Progressive Neural Networks](https://arxiv.org/pdf/1606.04671.pdf) | 2016 | Arxiv

#####  Regularization-based Methods
> Regularization-based approaches avoid forgetting by penalizing updates of important parameters or distilling knowledge with previous model as a teacher.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation](https://openaccess.thecvf.com/content/CVPR2022/papers/Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response_CVPR_2022_paper.pdf) | 2022 | CVPR
| [Natural continual learning: success is a journey, not (just) a destination](https://proceedings.neurips.cc/paper/2021/file/ec5aa0b7846082a2415f0902f0da88f2-Paper.pdf) | 2021 | NeurIPS
| [CPR: Classifier-Projection Regularization for Continual Learning](https://openreview.net/pdf?id=F2v4aqEL6ze) | 2021 | ICLR
| [Continual Learning with Node-Importance based Adaptive Group Sparse Regularization](https://proceedings.neurips.cc/paper/2020/file/258be18e31c8188555c2ff05b4d542c3-Paper.pdf) | 2020 | NeurIPS
| [Uncertainty-based Continual Learning with Adaptive Regularization](https://proceedings.neurips.cc/paper_files/paper/2019/file/2c3ddf4bf13852db711dd1901fb517fa-Paper.pdf) | 2019 |NeurIPS
| [Efficient Lifelong Learning with A-GEM](https://openreview.net/pdf?id=Hkf2_sC5FX) | 2019| ICLR
| [Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence](https://openaccess.thecvf.com/content_ECCV_2018/papers/Arslan_Chaudhry__Riemannian_Walk_ECCV_2018_paper.pdf) | 2018 | ECCV
| [Memory Aware Synapses: Learning what (not) to forget](https://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf) | 2018 | ECCV
| [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/pdf/1612.00796v2.pdf) | 2017 | Arxiv
| [Continual Learning Through Synaptic Intelligence](https://dl.acm.org/doi/pdf/10.5555/3305890.3306093) | 2017 | ICML
| [Learning without Forgetting](https://ieeexplore.ieee.org/document/8107520) |2016 | TPAMI



#####  Subspace-based Methods
> Subspace-based methods perform CL in multiple disjoint subspaces to avoid interference between multiple tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Building a Subspace of Policies for Scalable Continual Learning](https://openreview.net/pdf?id=ZloanUtG4a) | 2023 | ICLR
| [Rethinking Gradient Projection Continual Learning: Stability / Plasticity Feature Space Decoupling](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Rethinking_Gradient_Projection_Continual_Learning_Stability__Plasticity_Feature_Space_CVPR_2023_paper.pdf) | 2023 | CVPR
| [Continual Learning with Scaled Gradient Projection](https://arxiv.org/pdf/2302.01386.pdf) | 2023 | AAAI
| [SketchOGD: Memory-Efficient Continual Learning](https://arxiv.org/pdf/2305.16424.pdf) | 2023 | Arxiv
| [Beyond Not-Forgetting: Continual Learning with Backward Knowledge Transfer](https://openreview.net/pdf?id=diV1PpaP33) | 2022 | NeurIPS
| [TRGP: Trust Region Gradient Projection for Continual Learning](https://openreview.net/pdf?id=iEvAf8i6JjO) | 2022 | ICLR
| [Continual Learning with Recursive Gradient Optimization](https://openreview.net/pdf?id=7YDLgf9_zgm) | 2022 | ICLR
| [Balancing Stability and Plasticity through Advanced Null Space in Continual Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860215.pdf) |  2022 | ECCV
| [Adaptive Orthogonal Projection for Batch and Online Continual Learning](https://ojs.aaai.org/index.php/AAAI/article/view/20634/20393) | 2022 | AAAI
| [Natural continual learning: success is a journey, not (just) a destination](https://proceedings.neurips.cc/paper/2021/file/ec5aa0b7846082a2415f0902f0da88f2-Paper.pdf) | 2021 | NeurIPS
| [Flattening Sharpness for Dynamic Gradient Projection Memory Benefits Continual Learning](https://proceedings.neurips.cc/paper/2021/file/9b16759a62899465ab21e2e79d2ef75c-Paper.pdf) | 2021 | NeurIPS
| [Gradient Projection Memory for Continual Learning](https://openreview.net/pdf?id=3AOj0RCNC2) |2021 | ICLR
| [Training Networks in Null Space of Feature Covariance for Continual Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Training_Networks_in_Null_Space_of_Feature_Covariance_for_Continual_CVPR_2021_paper.pdf) |2021 | CVPR
| [Generalisation Guarantees For Continual Learning With Orthogonal Gradient Descent](https://arxiv.org/pdf/2006.11942.pdf) | 2021 | Arxiv
| [Defeating Catastrophic Forgetting via Enhanced Orthogonal Weights Modification](https://arxiv.org/pdf/2111.10078.pdf) | 2021 | Arxiv
| [Continual Learning in Low-rank Orthogonal Subspaces](https://papers.nips.cc/paper/2020/file/70d85f35a1fdc0ab701ff78779306407-Paper.pdf) | 2020 | NeurIPS
| [Orthogonal Gradient Descent for Continual Learning](https://core.ac.uk/download/pdf/345075797.pdf) |  2020 | AISTATS
| [Generalisation Guarantees for Continual Learning with Orthogonal Gradient Descent](https://arxiv.org/pdf/2006.11942.pdf) | 2020 | Arxiv
| [Generative Feature Replay with Orthogonal Weight Modification for Continual Learning](https://arxiv.org/pdf/2005.03490.pdf) |2020 | Arxiv
| [Continual Learning of Context-dependent Processing in Neural Networks](https://www.nature.com/articles/s42256-019-0080-x)| 2019 | Nature Machine Intelligence


#####  Bayesian Methods
> Bayesian methods provide a principled probabilistic framework for addressing Forgetting.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [A Probabilistic Framework for Modular Continual Learning](https://arxiv.org/pdf/2306.06545.pdf) | 2023 |Arxiv
| [Online Continual Learning on Class Incremental Blurry Task Configuration with Anytime Inference](https://openreview.net/pdf?id=nrGGfMbY_qK) |2022 | ICLR
| [Continual Learning via Sequential Function-Space Variational Inference](https://proceedings.mlr.press/v162/rudner22a/rudner22a.pdf) | 2022 | ICML
| [Generalized Variational Continual Learning](https://openreview.net/pdf?id=_IM-AfFhna9) | 2021 | ICLR
| [Variational Auto-Regressive Gaussian Processes for Continual Learning](http://proceedings.mlr.press/v139/kapoor21b/kapoor21b.pdf)| 2021 | ICML
| [Bayesian Structural Adaptation for Continual Learning](http://proceedings.mlr.press/v139/kumar21a/kumar21a.pdf) | 2021 | ICML
| [Continual Learning using a Bayesian Nonparametric Dictionary of Weight Factors](http://proceedings.mlr.press/v130/mehta21a/mehta21a.pdf) | 2021 | AISTATS
| [Posterior Meta-Replay for Continual Learning](https://openreview.net/pdf?id=AhuVLaYp6gn) |2021 |NeurIPS
| [Natural continual learning: success is a journey, not (just) a destination](https://openreview.net/pdf?id=W9250bXDgpK) | 2021 |NeurIPS
| [Continual Learning with Adaptive Weights (CLAW)](https://openreview.net/pdf?id=Hklso24Kwr) | 2020 | ICLR
| [Uncertainty-guided Continual Learning with Bayesian Neural Networks](https://openreview.net/pdf?id=HklUCCVKDB) | 2020 | ICLR
| [Functional Regularisation for Continual Learning with Gaussian Processes](https://openreview.net/pdf?id=HkxCzeHFDB) | 2020 | ICLR
| [Continual Deep Learning by Functional Regularisation of Memorable Past](https://dl.acm.org/doi/pdf/10.5555/3495724.3496098)|2020| NeurIPS
| [Variational Continual Learning](https://openreview.net/pdf?id=BkQqq0gRb) | 2018 | ICLR
| [Online Structured Laplace Approximations for Overcoming Catastrophic Forgetting](https://proceedings.neurips.cc/paper_files/paper/2018/file/f31b20466ae89669f9741e047487eb37-Paper.pdf) | 2018| NeurIPS
| [Overcoming Catastrophic Forgetting by Incremental Moment Matching](https://proceedings.neurips.cc/paper_files/paper/2017/file/f708f064faaf32a43e4d3c784e6af9ea-Paper.pdf) | 2017| NeurIPS


#### Task-free CL
> Task-free CL refers to a specific scenario that the learning system does not have access to any explicit task information.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Online Bias Correction for Task-Free Continual Learning](https://openreview.net/pdf?id=18XzeuYZh_) | 2023 | ICLR
| [Task-Free Continual Learning via Online Discrepancy Distance Learning](https://openreview.net/pdf?id=UFTcdcJrIl2) | 2022 |NeurIPS
| [Improving Task-free Continual Learning by Distributionally Robust Memory Evolution](https://proceedings.mlr.press/v162/wang22v/wang22v.pdf) | 2022 | ICML
| [VariGrow: Variational architecture growing for task-agnostic continual learning based on Bayesian novelty](https://proceedings.mlr.press/v162/ardywibowo22a/ardywibowo22a.pdf) | 2022 | ICML
| [Gradient-based Editing of Memory Examples for Online Task-free Continual Learning](https://proceedings.neurips.cc/paper/2021/file/f45a1078feb35de77d26b3f7a52ef502-Paper.pdf) | 2021 |NeurIPS
| [Continuous Meta-Learning without Tasks](https://proceedings.neurips.cc/paper/2020/file/cc3f5463bc4d26bc38eadc8bcffbc654-Paper.pdf) | 2020 |NeurIPS
| [A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning](https://openreview.net/pdf?id=SJxSOJStPr) | 2020 | ICLR
| [Online Continual Learning with Maximally Interfered Retrieval](https://proceedings.neurips.cc/paper/2019/file/15825aee15eb335cc13f9b559f166ee8-Paper.pdf) | 2019 | NeurIPS
| [Gradient based sample selection for online continual learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/e562cd9c0768d5464b64cf61da7fc6bb-Paper.pdf) | 2019 | NeurIPS
| [Efficient lifelong learning with A-GEM](https://openreview.net/pdf?id=Hkf2_sC5FX) | 2019 | ICLR |
| [Task-Free Continual Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aljundi_Task-Free_Continual_Learning_CVPR_2019_paper.pdf) | 2019 | CVPR
| [Continual Learning with Tiny Episodic Memories](https://arxiv.org/pdf/1902.10486v1.pdf) | 2019 | Arxiv


#### Online CL
> In online CL, the learner is only allowed to process the data for each task once.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [New Insights for the Stability-Plasticity Dilemma in Online Continual Learning](https://openreview.net/pdf?id=fxC7kJYwA_a) | 2023 | ICLR
| [Real-Time Evaluation in Online Continual Learning: A New Hope](https://openaccess.thecvf.com/content/CVPR2023/papers/Ghunaim_Real-Time_Evaluation_in_Online_Continual_Learning_A_New_Hope_CVPR_2023_paper.pdf) | 2023 | CVPR
| [PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_PCR_Proxy-Based_Contrastive_Replay_for_Online_Class-Incremental_Continual_Learning_CVPR_2023_paper.pdf) |2023 | CVPR
| [Dealing with Cross-Task Class Discrimination in Online Continual Learning](https://arxiv.org/pdf/2305.14657.pdf) |2023 | CVPR
| [Online continual learning through mutual information maximization](https://proceedings.mlr.press/v162/guo22g/guo22g.pdf) | 2022 | ICML
| [Online Coreset Selection for Rehearsal-based Continual Learning](https://openreview.net/pdf?id=f9D-5WNG4Nv) | 2022 | ICLR
| [New Insights on Reducing Abrupt Representation Change in Online Continual Learning](https://openreview.net/pdf?id=N8MaByOzUfb) |2022 | ICLR
| [Online Continual Learning on Class Incremental Blurry Task Configuration with Anytime Inference](https://openreview.net/pdf?id=nrGGfMbY_qK) | 2022 | ICLR
| [Information-theoretic Online Memory Selection for Continual Learning](https://openreview.net/pdf?id=IpctgL7khPp) | 2022 | ICLR
| [Continual Normalization: Rethinking Batch Normalization for Online Continual Learning](https://openreview.net/pdf?id=vwLLQ-HwqhZ) | 2022 | ICLR
| [Navigating Memory Construction by Global Pseudo-Task Simulation for Continual Learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/3013680bf2d072b5f3851aec70b39a59-Paper-Conference.pdf) |  2022 |NeurIPS
| [Not Just Selection, but Exploration: Online Class-Incremental Continual Learning via Dual View Consistency](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Not_Just_Selection_but_Exploration_Online_Class-Incremental_Continual_Learning_via_CVPR_2022_paper.pdf) |2022 | CVPR
| [Online Task-free Continual Learning with Dynamic Sparse Distributed Memory](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850721.pdf) | 2022 | ECCV
| [Mitigating Forgetting in Online Continual Learning with Neuron Calibration](https://proceedings.neurips.cc/paper/2021/file/54ee290e80589a2a1225c338a71839f5-Paper.pdf) | 2021 | NeurIPS
| [Online class-incremental continual learning with adversarial shapley value](https://arxiv.org/pdf/2009.00093.pdf) | 2021 | AAAI
| [Online Continual Learning with Natural Distribution Shifts: An Empirical Study with Visual Data](https://openaccess.thecvf.com/content/ICCV2021/papers/Cai_Online_Continual_Learning_With_Natural_Distribution_Shifts_An_Empirical_Study_ICCV_2021_paper.pdf) |2021 | ICCV
| [Continual Prototype Evolution: Learning Online from Non-Stationary Data Streams](https://openaccess.thecvf.com/content/ICCV2021/papers/De_Lange_Continual_Prototype_Evolution_Learning_Online_From_Non-Stationary_Data_Streams_ICCV_2021_paper.pdf) | 2021 | ICCV
| [La-MAML: Look-ahead Meta Learning for Continual Learning](https://proceedings.neurips.cc/paper/2020/file/85b9a5ac91cd629bd3afe396ec07270a-Paper.pdf) | 2020 |NeurIPS
| [Online Learned Continual Compression with Adaptive Quantization Modules](http://proceedings.mlr.press/v119/caccia20a/caccia20a.pdf) | 2020 | ICML
| [Online Continual Learning under Extreme Memory Constraints](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730715.pdf) | 2020| ECCV
| [Online Continual Learning with Maximally Interfered Retrieval](https://proceedings.neurips.cc/paper/2019/file/15825aee15eb335cc13f9b559f166ee8-Paper.pdf) | 2019 | NeurIPS
| [Gradient based sample selection for online continual learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/e562cd9c0768d5464b64cf61da7fc6bb-Paper.pdf) | 2019 | NeurIPS
| [On Tiny Episodic Memories in Continual Learning](https://arxiv.org/pdf/1902.10486.pdf) | Arxiv | 2019
| [Progress & Compress: A scalable framework for continual learning](https://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf) | 2018 | ICML



The presence of **imbalanced data** streams in CL (especially online CL) has drawn significant attention, primarily due to its prevalence in real-world application scenarios.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Online Bias Correction for Task-Free Continual Learning](https://openreview.net/pdf?id=18XzeuYZh_) | 2023| ICLR
| [Information-theoretic Online Memory Selection for Continual Learning](https://openreview.net/pdf?id=IpctgL7khPp) | 2022 | ICLR
| [SS-IL: Separated Softmax for Incremental Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Ahn_SS-IL_Separated_Softmax_for_Incremental_Learning_ICCV_2021_paper.pdf) |2021 | ICCV
| [Online Continual Learning from Imbalanced Data](http://proceedings.mlr.press/v119/chrysakis20a/chrysakis20a.pdf) | 2020 | ICML
| [Maintaining Discrimination and Fairness in Class Incremental Learning]() |2020 | CVPR
| [Imbalanced Continual Learning with Partitioning Reservoir Sampling](https://arxiv.org/pdf/2009.03632.pdf) | 2020| ECCV
| [GDumb: A Simple Approach that Questions Our Progress in Continual Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf) |2020 | ECCV
| [Large scale incremental learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf) | 2019 | CVPR
| [IL2M: Class Incremental Learning With Dual Memory](https://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf) | 2019|ICCV
| [End-to-end incremental learning](https://arxiv.org/pdf/1807.09536.pdf) | 2018 | ECCV



#### Semi-supervised CL
> Semi-supervised CL is an extension of traditional CL that allows each task to incorporate unlabeled data as well.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Semi-supervised drifted stream learning with short lookback](https://arxiv.org/pdf/2205.13066.pdf) | 2022 | SIGKDD
| [Ordisco: Effective and efficient usage of incremental unlabeled data for semi-supervised continual learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_ORDisCo_Effective_and_Efficient_Usage_of_Incremental_Unlabeled_Data_for_CVPR_2021_paper.pdf) | 2021 | CVPR
| [Memory-Efficient Semi-Supervised Continual Learning: The World is its Own Replay Buffer](https://arxiv.org/pdf/2101.09536.pdf) | 2021| IJCNN
| [Overcoming Catastrophic Forgetting with Unlabeled Data in the Wild](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_Overcoming_Catastrophic_Forgetting_With_Unlabeled_Data_in_the_Wild_ICCV_2019_paper.pdf) | 2019 | ICCV


#### Few-shot CL
> Few-shot CL refers to the scenario where a model needs to learn new tasks with only a limited number of labeled examples per task while retaining knowledge from previously encountered tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Warping the Space: Weight Space Rotation for Class-Incremental Few-Shot Learning](https://openreview.net/pdf?id=kPLzOfPfA2l) |2023 | ICLR
| [Neural Collapse Inspired Feature-Classifier Alignment for Few-Shot Class Incremental Learning](https://openreview.net/pdf?id=y5W8tpojhtJ) |2023 | ICLR
| [Few-Shot Class-Incremental Learning by Sampling Multi-Phase Tasks](https://arxiv.org/pdf/2203.17030.pdf) | 2022 | TPAMI
| [Dynamic Support Network for Few-Shot Class Incremental Learning](https://ieeexplore.ieee.org/document/9779071) | 2022| TPAMI
| [Subspace Regularizers for Few-Shot Class Incremental Learning](https://openreview.net/pdf?id=boJy41J-tnQ) | 2022 | ICLR
| [MetaFSCIL: A Meta-Learning Approach for Few-Shot Class Incremental Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Chi_MetaFSCIL_A_Meta-Learning_Approach_for_Few-Shot_Class_Incremental_Learning_CVPR_2022_paper.pdf) | 2022 | CVPR
| [Forward Compatible Few-Shot Class-Incremental Learning](https://arxiv.org/pdf/2203.06953.pdf) | 2022 | CVPR
| [Constrained Few-shot Class-incremental Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Hersche_Constrained_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf) |  2022 | CVPR
| [Few-Shot Class-Incremental Learning via Entropy-Regularized Data-Free Replay](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840144.pdf) | 2022| ECCV
| [MgSvF: Multi-Grained Slow vs. Fast Framework for Few-Shot Class-Incremental Learning](https://ieeexplore.ieee.org/document/9645290) | 2021 | TPAMI
| [Semantic-aware Knowledge Distillation for Few-Shot Class-Incremental Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheraghian_Semantic-Aware_Knowledge_Distillation_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf) | 2021 | CVPR
| [Self-Promoted Prototype Refinement for Few-Shot Class-Incremental Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Self-Promoted_Prototype_Refinement_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf) | 2021|CVPR
| [Few-Shot Incremental Learning with Continually Evolved Classifiers](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Few-Shot_Incremental_Learning_With_Continually_Evolved_Classifiers_CVPR_2021_paper.pdf) | 2021| CVPR
| [Synthesized Feature based Few-Shot Class-Incremental Learning on a Mixture of Subspaces](https://openaccess.thecvf.com/content/ICCV2021/papers/Cheraghian_Synthesized_Feature_Based_Few-Shot_Class-Incremental_Learning_on_a_Mixture_of_ICCV_2021_paper.pdf) | 2021| ICCV
| [Few-Shot Lifelong Learning](https://arxiv.org/pdf/2103.00991.pdf) | 2021 | AAAI
| [Few-Shot Class-Incremental Learning via Relation Knowledge Distillation](https://ojs.aaai.org/index.php/AAAI/article/view/16213) | 2021 | AAAI
| [Few-shot Continual Learning: a Brain-inspired Approach](https://arxiv.org/pdf/2104.09034.pdf) | 2021 | Arxiv |
| [Few-Shot Class-Incremental Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tao_Few-Shot_Class-Incremental_Learning_CVPR_2020_paper.pdf) | 2020 | CVPR


#### Unsupervised CL
> Unsupervised CL (UCL) assumes that only unlabeled data is provided to the CL learner.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Unsupervised Continual Learning in Streaming Environments](https://ieeexplore.ieee.org/document/9756660) | 2023 | TNNLS
| [Representational Continuity for Unsupervised Continual Learning](https://openreview.net/pdf?id=9Hrka5PA7LW) | 2022 | ICLR
| [Probing Representation Forgetting in Supervised and Unsupervised Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Davari_Probing_Representation_Forgetting_in_Supervised_and_Unsupervised_Continual_Learning_CVPR_2022_paper.pdf) |2022 | CVPR
| [Unsupervised Continual Learning for Gradually Varying Domains](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Taufique_Unsupervised_Continual_Learning_for_Gradually_Varying_Domains_CVPRW_2022_paper.pdf) | 2022 | CVPRW
| [Co2L: Contrastive Continual Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Cha_Co2L_Contrastive_Continual_Learning_ICCV_2021_paper.pdf) |2021 | ICCV
| [Unsupervised Progressive Learning and the STAM Architecture](https://www.ijcai.org/proceedings/2021/0410.pdf) | 2021 | IJCAI
| [Continual Unsupervised Representation Learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/861578d797aeb0634f77aff3f488cca2-Paper.pdf) | 2019 | NeurIPS


#### Theoretical Analysis
> Theory or analysis of continual learning

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [The Ideal Continual Learner: An Agent That Never Forgets](https://openreview.net/pdf?id=o7BOzuqFi2) | 2023 | ICML
| [Continual Learning in Linear Classification on Separable Data](https://openreview.net/pdf?id=kkpIrMu3Vf) | 2023 | ICML
| [Theory on Forgetting and Generalization of Continual Learning](https://arxiv.org/pdf/2302.05836.pdf) |2023  | ArXiv
| [A Theoretical Study on Solving Continual Learning](https://openreview.net/pdf?id=bA8CYH5uEn_) | 2022 | NeurIPS
| [Learning Curves for Continual Learning in Neural Networks: Self-Knowledge Transfer and Forgetting](https://openreview.net/pdf?id=tFgdrQbbaa) |2022 | ICLR
| [Continual Learning in the Teacher-Student Setup: Impact of Task Similarity](https://arxiv.org/pdf/2107.04384.pdf) |2022 | ICML
| [Formalizing the Generalization-Forgetting Trade-off in Continual Learning](https://openreview.net/pdf?id=u1XV9BPAB9) | 2021 | NeurIPS
| [A PAC-Bayesian Bound for Lifelong Learning](http://proceedings.mlr.press/v32/pentina14.pdf) | 2014 | ICML


----------


### Forgetting in Foundation Models


> Foundation models are large machine learning models trained on a vast quantity of data at scale, such that they can be adapted to a wide range of downstream tasks.

**Links**: [Forgetting in Fine-Tuning Foundation Models](#forgetting-in-fine-tuning-foundation-models) | [Forgetting in One-Epoch Pre-training](#forgetting-in-one-epoch-pre-training) | [CL in Foundation Model](#cl-in-foundation-model)

#### Forgetting in Fine-Tuning Foundation Models
> When fine-tuning a foundation model, there is a tendency to forget the pre-trained knowledge, resulting in sub-optimal performance on downstream tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Improving Gender Fairness of Pre-Trained Language Models without Catastrophic Forgetting](https://arxiv.org/pdf/2110.05367.pdf) |2023 | ACL
| [On The Role of Forgetting in Fine-Tuning Reinforcement Learning Models](https://openreview.net/pdf?id=zmXJUKULDzh) | 2023 | ICLRW
| [Reinforcement Learning with Action-Free Pre-Training from Videos](https://proceedings.mlr.press/v162/seo22a/seo22a.pdf) | 2022 | ICML
| [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](https://openreview.net/pdf?id=AXDNM76T1nc) | 2022 |NeurIPS
| [How Should Pre-Trained Language Models Be Fine-Tuned Towards Adversarial Robustness?](https://proceedings.neurips.cc/paper/2021/file/22b1f2e0983160db6f7bb9f62f4dbb39-Paper.pdf) | 2021 | NeurIPS
| [Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models](https://openreview.net/pdf?id=HkgaETNtDB) | 2020 | ICLR
| [Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting](https://aclanthology.org/2020.emnlp-main.634.pdf) | 2020 | EMNLP
| [Universal Language Model Fine-tuning for Text Classification](https://aclanthology.org/P18-1031.pdf) | 2018 | ACL



#### Forgetting in One-Epoch Pre-training
> Foundation models often undergo training on a dataset for a single pass. As a result, the earlier examples encountered during pre-training may be overwritten or forgotten by the model more quickly than the later examples.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Measuring Forgetting of Memorized Training Examples](https://openreview.net/pdf?id=7bJizxLKrR) | 2023 | ICLR
| [Quantifying Memorization Across Neural Language Models](https://openreview.net/pdf?id=TatRHT_1cK) | 2023| ICLR
| [Analyzing leakage of personally identifiable information in language models](https://arxiv.org/pdf/2302.00539.pdf) | 2023|S\&P
| [How Well Does Self-Supervised Pre-Training Perform with Streaming Data?](https://arxiv.org/pdf/2104.12081.pdf) | 2022| ICLR
| [The challenges of continuous self-supervised learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860687.pdf) | 2022 | ECCV
| [Continual contrastive learning for image classification](https://ieeexplore.ieee.org/document/9859995) | 2022 | ICME


#### CL in Foundation Model
> By leveraging the powerful feature extraction capabilities of foundation models, researchers have been able to explore new avenues for advancing continual learning techniques.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Progressive Prompts: Continual Learning for Language Models](https://openreview.net/pdf?id=UJTgQBc91_) | 2023| ICLR
| [Continual Pre-training of Language Models](https://openreview.net/pdf?id=m_GDIItaI3o) | 2023 | ICLR
| [CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) | 2023 | CVPR
| [PIVOT: Prompting for Video Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Villa_PIVOT_Prompting_for_Video_Continual_Learning_CVPR_2023_paper.pdf) |2023 | CVPR
| [Do Pre-trained Models Benefit Equally in Continual Learning?](https://openaccess.thecvf.com/content/WACV2023/papers/Lee_Do_Pre-Trained_Models_Benefit_Equally_in_Continual_Learning_WACV_2023_paper.pdf) | 2023 | WACV
| [Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](https://arxiv.org/pdf/2303.07338.pdf) | 2023 | Arxiv
| [First Session Adaptation: A Strong Replay-Free Baseline for Class-Incremental Learning](https://arxiv.org/pdf/2303.13199.pdf) | 2023 | Arxiv
| [Memory Efficient Continual Learning with Transformers](https://openreview.net/pdf?id=U07d1Y-x2E) | 2022 | NeurIPS
| [S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning](https://openreview.net/pdf?id=ZVe_WeMold) |2022 | NeurIPS
| [Pretrained Language Model in Continual Learning: A Comparative Study](https://openreview.net/pdf?id=figzpGMrdD) | 2022 | ICLR
| [Effect of scale on catastrophic forgetting in neural networks](https://openreview.net/pdf?id=GhVS8_yPeEa) | 2022| ICLR
| [An Empirical Investigation of the Role of Pre-training in Lifelong Learning](https://arxiv.org/pdf/2112.09153.pdf) | 2022 | ICML
| [Learning to Prompt for Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.pdf) | 2022 | CVPR
| [Class-Incremental Learning with Strong Pre-trained Models](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Class-Incremental_Learning_With_Strong_Pre-Trained_Models_CVPR_2022_paper.pdf) |  2022|CVPR
| [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/pdf/2204.04799.pdf) |2022  |ECCV
| [ELLE: Efficient Lifelong Pre-training for Emerging Data](https://aclanthology.org/2022.findings-acl.220.pdf) | 2022 | ACL
| [Fine-tuned Language Models are Continual Learners](https://aclanthology.org/2022.emnlp-main.410.pdf) | 2022 | EMNLP
| [Continual Training of Language Models for Few-Shot Learning](https://aclanthology.org/2022.emnlp-main.695.pdf) | 2022 | EMNLP
| [Continual Learning with Foundation Models: An Empirical Study of Latent Replay](https://arxiv.org/pdf/2205.00329.pdf) |2022 | Conference on Lifelong Learning Agents
| [Achieving Forgetting Prevention and Knowledge Transfer in Continual Learning](https://openreview.net/pdf?id=RJ7XFI15Q8f) | 2021 |NeurIPS



### Forgetting in Domain Adaptation


> The goal of domain adaptation is to transfer the knowledge from a source domain to a target domain.


| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Continual Source-Free Unsupervised Domain Adaptation](https://arxiv.org/pdf/2304.07374.pdf) | 2023 | International Conference on Image Analysis and Processing
| [CoSDA: Continual Source-Free Domain Adaptation](https://arxiv.org/pdf/2304.06627.pdf) | 2023| Arxiv
| [Lifelong Domain Adaptation via Consolidated Internal Distribution](https://proceedings.neurips.cc/paper_files/paper/2021/file/5caf41d62364d5b41a893adc1a9dd5d4-Paper.pdf) |2022 |NeurIPS
| [Online Domain Adaptation for Semantic Segmentation in Ever-Changing Conditions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940125.pdf) | 2022| ECCV
| [FRIDA -- Generative Feature Replay for Incremental Domain Adaptation](https://arxiv.org/pdf/2112.14316.pdf) | 2022 | CVIU
| [Unsupervised Continual Learning for Gradually Varying Domains](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Taufique_Unsupervised_Continual_Learning_for_Gradually_Varying_Domains_CVPRW_2022_paper.pdf) |2022 | CVPRW
| [Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning](https://arxiv.org/pdf/2012.04324.pdf) | 2021|CVPR
| [Gradient Regularized Contrastive Learning for Continual Domain Adaptation](https://ojs.aaai.org/index.php/AAAI/article/view/16370/16177) |2021 | AAAI
| [Learning to Adapt to Evolving Domains](https://proceedings.neurips.cc/paper/2020/file/fd69dbe29f156a7ef876a40a94f65599-Paper.pdf) | 2020 | NeurIPS
| [AdaGraph: Unifying Predictive and Continuous Domain Adaptation through Graphs](https://openaccess.thecvf.com/content_CVPR_2019/papers/Mancini_AdaGraph_Unifying_Predictive_and_Continuous_Domain_Adaptation_Through_Graphs_CVPR_2019_paper.pdf) | 2019|CVPR
| [ACE: Adapting to Changing Environments for Semantic Segmentation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_ACE_Adapting_to_Changing_Environments_for_Semantic_Segmentation_ICCV_2019_paper.pdf) | 2019| ICCV
| [Adapting to Continuously Shifting Domains](https://openreview.net/pdf?id=BJsBjPJvf) | 2018 | ICLRW


----------


### Forgetting in Test-Time Adaptation
<!-- <u>[Click back to content outline](#framework)</u> -->

> Test time adaptation (TTA) refers to the process of adapting a pre-trained model on-the-fly to unlabeled test data during inference or testin.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [MECTA: Memory-Economic Continual Test-Time Model Adaptation](https://openreview.net/pdf?id=N92hjSf5NNh) | 2023|ICLR
| [Decorate the Newcomers: Visual Domain Prompt for Continual Test Time Adaptation](https://arxiv.org/pdf/2212.04145.pdf) |2023 | AAAI (Outstanding Student Paper Award)
| [Robust Mean Teacher for Continual and Gradual Test-Time Adaptation](https://openaccess.thecvf.com/content/CVPR2023/papers/Dobler_Robust_Mean_Teacher_for_Continual_and_Gradual_Test-Time_Adaptation_CVPR_2023_paper.pdf) | 2023|CVPR
| [A Probabilistic Framework for Lifelong Test-Time Adaptation](https://openaccess.thecvf.com/content/CVPR2023/papers/Brahma_A_Probabilistic_Framework_for_Lifelong_Test-Time_Adaptation_CVPR_2023_paper.pdf) | 2023 | CVPR
| [AUTO: Adaptive Outlier Optimization for Online Test-Time OOD Detection](https://arxiv.org/pdf/2303.12267.pdf) |2023 |Arxiv
| [Efficient Test-Time Model Adaptation without Forgetting](https://proceedings.mlr.press/v162/niu22a/niu22a.pdf) | 2022| ICML
| [MEMO: Test time robustness via adaptation and augmentation](https://openreview.net/pdf?id=vn74m_tWu8O) | 2022|NeurIPS
| [Continual Test-Time Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.pdf) | 2022|CVPR
| [Improving test-time adaptation via shift-agnostic weight regularization and nearest source prototypes](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930433.pdf) |2022 | ECCV
| [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://openreview.net/pdf?id=uXl3bZLkr3c) | 2021 | ICLR


----------

### Forgetting in Meta-Learning


> Meta-learning, also known as learning to learn, focuses on developing algorithms and models that can learn from previous learning experiences to improve their ability to learn new tasks or adapt to new domains more efficiently and effectively.

**Links**:
[Incremental Few-Shot Learning](#incremental-few-shot-learning) |
[Continual Meta-Learning](#continual-meta-learning)


#### Incremental Few-Shot Learning
> Incremental few-shot learning (IFSL) focuses on the challenge of learning new categories with limited labeled data while retaining knowledge about previously learned categories.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Constrained Few-shot Class-incremental Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Hersche_Constrained_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf) | 2022 | CVPR
| [Meta-Learning with Less Forgetting on Large-Scale Non-Stationary Task Distributions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800211.pdf) | 2022 | ECCV
| [Overcoming Catastrophic Forgetting in Incremental Few-Shot Learning by Finding Flat Minima](https://proceedings.neurips.cc/paper/2021/file/357cfba15668cc2e1e73111e09d54383-Paper.pdf) | 2021 | NeurIPS
| [Incremental Few-shot Learning via Vector Quantization in Deep Embedded Space](https://openreview.net/pdf?id=3SV-ZePhnZM) |2021 | ICLR
| [XtarNet: Learning to Extract Task-Adaptive Representation for Incremental Few-Shot Learning](http://proceedings.mlr.press/v119/yoon20b/yoon20b.pdf) |2020 | ICML
| [Incremental Few-Shot Learning with Attention Attractor Networks](https://proceedings.neurips.cc/paper_files/paper/2019/file/e833e042f509c996b1b25324d56659fb-Paper.pdf) |2019 | NeurIPS
| [Dynamic Few-Shot Visual Learning without Forgetting](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gidaris_Dynamic_Few-Shot_Visual_CVPR_2018_paper.pdf) | 2018| CVPR



#### Continual Meta-Learning
> The goal of continual meta-learning (CML) is to address the challenge of forgetting in non-stationary task distributions.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Adaptive Compositional Continual Meta-Learning](https://proceedings.mlr.press/v202/wu23d/wu23d.pdf) | 2023|ICML
| [Learning to Learn and Remember Super Long Multi-Domain Task Sequence](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_To_Learn_and_Remember_Super_Long_Multi-Domain_Task_Sequence_CVPR_2022_paper.pdf) | 2022|CVPR
| [Meta-Learning with Less Forgetting on Large-Scale Non-Stationary Task Distributions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800211.pdf) | 2022|ECCV
| [Variational Continual Bayesian Meta-Learning](https://proceedings.neurips.cc/paper_files/paper/2021/file/cdd0500dc0ef6682fa6ec6d2e6b577c4-Paper.pdf) | 2021|NeurIPS
| [Meta Learning on a Sequence of Imbalanced Domains with Difficulty Awareness](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Meta_Learning_on_a_Sequence_of_Imbalanced_Domains_With_Difficulty_ICCV_2021_paper.pdf) | 2021|ICCV
| [Addressing Catastrophic Forgetting in Few-Shot Problems](http://proceedings.mlr.press/v139/yap21a/yap21a.pdf) |2020 | ICML
| [Continuous meta-learning without tasks](https://proceedings.neurips.cc/paper/2020/file/cc3f5463bc4d26bc38eadc8bcffbc654-Paper.pdf) | 2020|NeurIPS
| [Reconciling meta-learning and continual learning with online mixtures of tasks](https://proceedings.neurips.cc/paper/2019/file/7a9a322cbe0d06a98667fdc5160dc6f8-Paper.pdf) |2019 |NeurIPS
| [Fast Context Adaptation via Meta-Learning](http://proceedings.mlr.press/v97/zintgraf19a/zintgraf19a.pdf) |2019 | ICML
| [Online meta-learning](http://proceedings.mlr.press/v97/finn19a/finn19a.pdf) | 2019| ICML




----------

### Forgetting in Generative Models

> The goal of a generative model is to learn a generator that can generate samples from a target distribution.

**Links**:
 [GAN Training is a Continual Learning Problem](#gan-training-is-a-continual-learning-problem) |
 [Lifelong Learning of Generative Models](#lifelong-learning-of-generative-models)


#### GAN Training is a Continual Learning Problem
> Treating GAN training as a continual learning problem.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Learning to Retain while Acquiring: Combating Distribution-Shift in Adversarial Data-Free Knowledge Distillation](https://openaccess.thecvf.com/content/CVPR2023/papers/Patel_Learning_To_Retain_While_Acquiring_Combating_Distribution-Shift_in_Adversarial_Data-Free_CVPR_2023_paper.pdf) | 2023|CVPR
| [Momentum Adversarial Distillation: Handling Large Distribution Shifts in Data-Free Knowledge Distillation](https://proceedings.neurips.cc/paper_files/paper/2022/file/41128e5b3a7622da5b17588757599077-Paper-Conference.pdf) | 2022 |NeurIPS
| [Robust and Resource-Efficient Data-Free Knowledge Distillation by Generative Pseudo Replay](https://ojs.aaai.org/index.php/AAAI/article/view/20556/20315) | 2022|AAAI
| [Preventing Catastrophic Forgetting and Distribution Mismatch in Knowledge Distillation via Synthetic Data](https://openaccess.thecvf.com/content/WACV2022/papers/Binici_Preventing_Catastrophic_Forgetting_and_Distribution_Mismatch_in_Knowledge_Distillation_via_WACV_2022_paper.pdf) | 2022 | WACV
| [On Catastrophic Forgetting and Mode Collapse in Generative Adversarial Networks](https://arxiv.org/pdf/1807.04015.pdf) | 2020| IJCNN
| [Generative adversarial network training is a continual learning problem](https://arxiv.org/pdf/1811.11083.pdf) | 2018|ArXiv


#### Lifelong Learning of Generative Models
> The goal is to develop generative models that can continually generate high-quality samples for both new and previously encountered tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models](https://arxiv.org/pdf/2303.17591.pdf) | 2023|Arxiv
| [Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models](https://arxiv.org/pdf/2305.10120.pdf) | 2023|Arxiv
| [Lifelong Generative Modelling Using Dynamic Expansion Graph Model](https://ojs.aaai.org/index.php/AAAI/article/view/20867/20626) | 2022|AAAI
| [Continual Variational Autoencoder Learning via Online Cooperative Memorization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830515.pdf) |2022 |ECCV
| [Hyper-LifelongGAN: Scalable Lifelong Learning for Image Conditioned Generation](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhai_Hyper-LifelongGAN_Scalable_Lifelong_Learning_for_Image_Conditioned_Generation_CVPR_2021_paper.pdf) | 2021|CVPR
| [Lifelong Twin Generative Adversarial Networks](https://ieeexplore.ieee.org/document/9506116) |2021 | ICIP
| [Lifelong Mixture of Variational Autoencoders](https://arxiv.org/pdf/2107.04694.pdf) |2021 | TNNLS
| [Lifelong Generative Modeling](https://arxiv.org/pdf/1705.09847.pdf) | 2020 |Neurocomputing
| [Lifelong GAN: Continual Learning for Conditional Image Generation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhai_Lifelong_GAN_Continual_Learning_for_Conditional_Image_Generation_ICCV_2019_paper.pdf) |2019 | ICCV




----------

### Forgetting in Reinforcement Learning


> Reinforcement learning is a machine learning technique that allows an agent to learn how to behave in an environment by trial and error, through rewards and punishments.


| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Building a Subspace of Policies for Scalable Continual Learning](https://openreview.net/pdf?id=ZloanUtG4a) |2023 | ICLR
| [Modular Lifelong Reinforcement Learning via Neural Composition](https://openreview.net/pdf?id=5XmLzdslFNN) |2022 |ICLR
| [Towards continual reinforcement learning: A review and perspectives](https://arxiv.org/pdf/2012.13490.pdf) | 2022 | Journal of Artificial Intelligence Research
| [Model-Free Generative Replay for Lifelong Reinforcement Learning: Application to Starcraft-2](https://proceedings.mlr.press/v199/daniels22a/daniels22a.pdf) | 2022|Conference on Lifelong Learning Agents
| [Transient Non-stationarity and Generalisation in Deep Reinforcement Learning](https://openreview.net/pdf?id=Qun8fv4qSby) | 2021 | ICLR
| [Sharing Less is More: Lifelong Learning in Deep Networks with Selective Layer Transfer](http://proceedings.mlr.press/v139/lee21a/lee21a.pdf) | 2021| ICML
| [Pseudo-rehearsal: Achieving deep reinforcement learning without catastrophic forgetting](https://arxiv.org/pdf/1812.02464.pdf) | 2021|Neurocomputing
| [Exploiting Hierarchy for Learning and Transfer in KL-regularized RL](https://openreview.net/pdf?id=CCs4iXw4KJ-) | 2019|Arxiv
| [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf) | 2017| ICML
| [Progressive neural networks](https://arxiv.org/pdf/1606.04671.pdf) |2016 | Arxiv
| [Learning a synaptic learning rule](https://ieeexplore.ieee.org/document/155621) |1991 | IJCNN

----------

### Forgetting in Federated Learning


> Federated learning (FL) is a decentralized machine learning approach where the training process takes place on local devices or edge servers instead of a centralized server.

**Links**:
 [Forgetting Due to Non-IID Data in FL  ](#forgetting-due-to-non-iid-data-in-fl) |
 [Federated Continual Learning](#federated-continual-learning)

#### Forgetting Due to Non-IID Data in FL  

> This branch pertains to the forgetting problem caused by the inherent non-IID (not identically and independently distributed) data among different clients participating in FL.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [GradMA: A Gradient-Memory-based Accelerated Federated Learning with Alleviated Catastrophic Forgetting](https://openaccess.thecvf.com/content/CVPR2023/papers/Luo_GradMA_A_Gradient-Memory-Based_Accelerated_Federated_Learning_With_Alleviated_Catastrophic_Forgetting_CVPR_2023_paper.pdf) |2023 | CVPR
| [Acceleration of Federated Learning with Alleviated Forgetting in Local Training](https://openreview.net/pdf?id=541PxiEKN3F) |2022 |ICLR
| [Preservation of the Global Knowledge by Not-True Distillation in Federated Learning](https://openreview.net/pdf?id=qw3MZb1Juo) | 2022 |NeurIPS
| [Learn from Others and Be Yourself in Heterogeneous Federated Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf) |2022 |CVPR
| [Rethinking Architecture Design for Tackling Data Heterogeneity in Federated Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Qu_Rethinking_Architecture_Design_for_Tackling_Data_Heterogeneity_in_Federated_Learning_CVPR_2022_paper.pdf) | 2022|CVPR
| [Model-Contrastive Federated Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.pdf) | 2021|CVPR
| [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf) | 2020| ICML
| [Overcoming Forgetting in Federated Learning on Non-IID Data]() | 2019|NeurIPSW


#### Federated Continual Learning

> This branch addresses the issue of continual learning within each individual client in the federated learning process, which results in forgetting at the overall FL level.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [FedET: A Communication-Efficient Federated Class-Incremental Learning Framework Based on Enhanced Transformer](https://arxiv.org/pdf/2306.15347.pdf) | 2023| IJCAI
| [Better Generative Replay for Continual Federated Learning](https://openreview.net/pdf?id=cRxYWKiTan) |2023 | ICLR
| [Don’t Memorize; Mimic The Past: Federated Class Incremental Learning Without Episodic Memory](https://arxiv.org/pdf/2307.00497.pdf) |2023 | ICMLW
| [Addressing Catastrophic Forgetting in Federated Class-Continual Learning](https://arxiv.org/pdf/2303.06937.pdf) | 2023|Arxiv
| [Federated Class-Incremental Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Federated_Class-Incremental_Learning_CVPR_2022_paper.pdf) |2022 | CVPR
| [Continual Federated Learning Based on Knowledge Distillation](https://www.ijcai.org/proceedings/2022/0303.pdf) |2022 | IJCAI
| [Federated Continual Learning with Weighted Inter-client Transfer](http://proceedings.mlr.press/v139/yoon21b/yoon21b.pdf) | 2021| ICML
| [A distillation-based approach integrating continual learning and federated learning for pervasive services](https://arxiv.org/pdf/2109.04197.pdf) | 2021 |Arxiv



******


## Beneficial Forgetting

Beneficial forgetting arises when the model contains private information that could lead to privacy breaches or when irrelevant information hinders the learning of new tasks. In these situations, forgetting becomes desirable as it helps protect privacy and facilitate efficient learning by discarding unnecessary information.

| **Problem Setting** | **Goal** |
| --------------- | :---- |
| Mitigate Overfitting | mitigate memorization of training data through selective forgetting
|Debias and Forget Irrelevant Information | forget biased information to achieve better performance or remove  irrelevant information to learn new tasks
| Machine Unlearning | forget some specified training data to protect user privacy

**Links**:
<u>[Combat Overfitting Through Forgetting](#combat-overfitting-through-forgetting)</u> |
<u>[Learning New Knowledge Through Forgetting Previous Knowledge](#learning-new-knowledge-through-forgetting-previous-knowledge)</u> |
<u>[Machine Unlearning](#machine-unlearning)</u>

### Forgetting Irrelevant Information to Achieve Better Performance


####  Combat Overfitting Through Forgetting
> Overfitting in neural networks occurs when the model excessively memorizes the training data, leading to poor generalization. To address overfitting, it is necessary to selectively forget irrelevant or noisy information.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier](https://openreview.net/pdf?id=OpC-9aBBVJe) | 2023|ICLR
| [The Primacy Bias in Deep Reinforcement Learning](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf) | 2022|ICML
| [Learning with Selective Forgetting](https://www.ijcai.org/proceedings/2021/0137.pdf) | 2021|IJCAI
| [SIGUA: Forgetting May Make Learning with Noisy Labels More Robust](https://arxiv.org/pdf/1809.11008.pdf) | 2020|ICML
| [Invariant Representations through Adversarial Forgetting](https://ojs.aaai.org/index.php/AAAI/article/view/5850/5706) |2020 |AAAI
| [Forget a Bit to Learn Better: Soft Forgetting for CTC-based Automatic Speech Recognition](https://www.isca-speech.org/archive_v0/Interspeech_2019/pdfs/2841.pdf) | 2019 |Interspeech



####  Learning New Knowledge Through Forgetting Previous Knowledge
> "Learning to forget" suggests that not all previously acquired prior knowledge is helpful for learning new tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [ReFactor GNNs: Revisiting Factorisation-based Models from a Message-Passing Perspective](https://openreview.net/pdf?id=81LQV4k7a7X) | 2022|NeurIPS
| [Fortuitous Forgetting in Connectionist Networks](https://openreview.net/pdf?id=ei3SY1_zYsE) | 2022|ICLR
| [Skin Deep Unlearning: Artefact and Instrument Debiasing in the Context of Melanoma Classification](https://proceedings.mlr.press/v162/bevan22a/bevan22a.pdf) |2022 |ICML
| [Near-Optimal Task Selection for Meta-Learning with Mutual Information and Online Variational Bayesian Unlearning](https://proceedings.mlr.press/v151/chen22h/chen22h.pdf) |2022 |AISTATS
| [AFEC: Active Forgetting of Negative Transfer in Continual Learning](https://proceedings.neurips.cc/paper/2021/file/bc6dc48b743dc5d013b1abaebd2faed2-Paper.pdf) |2021 |NeurIPS
| [Active Forgetting: Adaptation of Memory by Prefrontal Control](https://www.annualreviews.org/doi/10.1146/annurev-psych-072720-094140) | 2021|Annual Review of Psychology
| [Learning to Forget for Meta-Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Baik_Learning_to_Forget_for_Meta-Learning_CVPR_2020_paper.pdf) | 2020|CVPR
| [The Forgotten Part of Memory](https://www.nature.com/articles/d41586-019-02211-5) |2019 |Nature
| [Learning Not to Learn: Training Deep Neural Networks with Biased Data](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Learning_Not_to_Learn_Training_Deep_Neural_Networks_With_Biased_CVPR_2019_paper.pdf) | 2019| CVPR
| [Inhibiting your native language: the role of retrieval-induced forgetting during second-language acquisition](https://pubmed.ncbi.nlm.nih.gov/17362374/) | 2007|Psychological Science


----------

### Machine Unlearning


> Machine unlearning, a recent area of research, addresses the need to forget previously learned training data in order to protect user data privacy.


| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Deep Unlearning via Randomized Conditionally Independent Hessians](https://openaccess.thecvf.com/content/CVPR2022/papers/Mehta_Deep_Unlearning_via_Randomized_Conditionally_Independent_Hessians_CVPR_2022_paper.pdf) | 2022|CVPR
| [Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Golatkar_Eternal_Sunshine_of_the_Spotless_Net_Selective_Forgetting_in_Deep_CVPR_2020_paper.pdf) |2022 | CVPR
| [PUMA: Performance Unchanged Model Augmentation for Training Data Removal](https://ojs.aaai.org/index.php/AAAI/article/view/20846/20605) | 2022|AAAI
| [ARCANE: An Efficient Architecture for Exact Machine Unlearning](https://www.ijcai.org/proceedings/2022/0556.pdf) | 2022 | IJCAI
| [Learn to Forget: Machine Unlearning via Neuron Masking](https://arxiv.org/pdf/2003.10933.pdf) |2022 |IEEE TDSC
| [Backdoor Defense with Machine Unlearning](https://dl.acm.org/doi/abs/10.1109/INFOCOM48880.2022.9796974) | 2022|IEEE INFOCOM
| [Markov Chain Monte Carlo-Based Machine Unlearning: Unlearning What Needs to be Forgotten](https://dl.acm.org/doi/abs/10.1145/3488932.3517406) | 2022|ASIA CCS
| [Machine Unlearning](https://arxiv.org/pdf/1912.03817.pdf) |2021 |SSP
| [Remember What You Want to Forget: Algorithms for Machine Unlearning](https://proceedings.neurips.cc/paper/2021/file/9627c45df543c816a3ddf2d8ea686a99-Paper.pdf) |2021 |NeurIPS
| [Machine Unlearning via Algorithmic Stability](https://proceedings.mlr.press/v134/ullah21a/ullah21a.pdf) | 2021|COLT
| [Variational Bayesian Unlearning](https://proceedings.neurips.cc/paper/2020/file/b8a6550662b363eb34145965d64d0cfb-Paper.pdf) |2020 |NeurIPS
| [Rapid retraining of machine learning models](http://proceedings.mlr.press/v119/wu20b/wu20b.pdf) |2020 |ICML
| [Certified Data Removal from Machine Learning Models](http://proceedings.mlr.press/v119/guo20c/guo20c.pdf) | 2020 |ICML
| [Making AI Forget You: Data Deletion in Machine Learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/cb79f8fa58b91d3af6c9c991f63962d3-Paper.pdf) | 2019|NeurIPS
| [Lifelong Anomaly Detection Through Unlearning](https://dl.acm.org/doi/10.1145/3319535.3363226) |2019 |CCS
| [The EU Proposal for a General Data Protection Regulation and the Roots of the ‘Right to Be Forgotten’](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2473151) | 2013|Computer Law & Security Review


******

**Contact**

We welcome all researchers to contribute to this repository **'forgetting in deep learning'**.

Email: wangzhenyineu@gmail.com | ennengyang@stumail.neu.edu.cn
