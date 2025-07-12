# Awesome-Forgetting-in-Deep-Learning

[![Awesome](https://awesome.re/badge.svg)]()
<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt=""/>

A comprehensive list of papers about **'[A Comprehensive Survey of Forgetting in Deep Learning Beyond Continual Learning](https://arxiv.org/abs/2307.09218)'**.

## Abstract
> Forgetting refers to the loss or deterioration of previously acquired information or knowledge. While the existing surveys on forgetting have primarily focused on continual learning, forgetting is a prevalent phenomenon observed in various other research domains within deep learning. Forgetting manifests in research fields such as generative models due to generator shifts, and federated learning due to heterogeneous data distributions across clients. Addressing forgetting encompasses several challenges, including balancing the retention of old task knowledge with fast learning of new tasks, managing task interference with conflicting goals, and preventing privacy leakage, etc. Moreover, most existing surveys on continual learning implicitly assume that forgetting is always harmful. In contrast, our survey argues that forgetting is a double-edged sword and can be beneficial and desirable in certain cases, such as privacy-preserving scenarios. By exploring forgetting in a broader context, we aim to present a more nuanced understanding of this phenomenon and highlight its potential advantages. Through this comprehensive survey, we aspire to uncover potential solutions by drawing upon ideas and approaches from various fields that have dealt with forgetting. By examining forgetting beyond its conventional boundaries, in future work, we hope to encourage the development of novel strategies for mitigating, harnessing, or even embracing forgetting in real applications.

## Citation

If you find our paper or this resource helpful, please consider citing:
```
@article{Forgetting_Survey_2024,
  title={A Comprehensive Survey of Forgetting in Deep Learning Beyond Continual Learning},
  author={Wang, Zhenyi and Yang, Enneng and Shen, Li and Huang, Heng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```
Thanks!

******

## Framework

  * [Harmful Forgetting](#harmful-forgetting)
    + [Forgetting in Continual Learning](#forgetting-in-continual-learning)
      - [Survey and Book](#survey-and-book)
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
| Continual Learning | learn non-stationary data distribution without forgetting previous knowledge  | data-distribution shift during training |
| Foundation Model |unsupervised learning on large-scale unlabeled data | data-distribution shift in pre-training, fine-tuning  |
| Domain Adaptation | adapt to target domain while maintaining performance on source domain | target domain sequentially shift over time |
| Test-time Adaptation |mitigate the distribution gap between training and testing | adaptation to the test data distribution during testing|
| Meta-Learning | learn adaptable knowledge to new tasks | incrementally meta-learn new classes / task-distribution shift  |
| Generative Model | learn a generator to appriximate real data distribution | generator shift/data-distribution shift |
| Reinforcement Learning | maximize accumulate rewards | state, action, reward and state transition dynamics|
| Federated Learning | decentralized training without sharing data |  model average; non-i.i.d data; data-distribution shift |

<!-- | Self-Supervised Learning | unsupervised pre-training | data-distribution shift | -->

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
<a href="#top">[Back to top]</a>

> The goal of continual learning  (CL) is to learn on a sequence of tasks without forgetting the knowledge on previous tasks.

**Links**:
<u> [Task-aware CL](#task-aware-cl)  </u>|
<u> [Task-free CL](#task-free-cl)  </u>|
<u> [Online CL](#online-cl)  </u>|
<u> [Semi-supervised CL](#semi-supervised-cl)  </u>|
<u> [Few-shot CL](#few-shot-cl)  </u>|
<u> [Unsupervised CL](#unsupervised-cl)  </u>|
<u> [Theoretical Analysis](#theoretical-analysis) </u>


#### Survey and Book
| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [A Survey of Continual Reinforcement Learning](https://arxiv.org/pdf/2506.21872) | 2025  | TPAMI
| [Advancements and Challenges in Continual Reinforcement Learning: A Comprehensive Review](https://arxiv.org/pdf/2506.21899)| 2025  | Arxiv
| [SWE-Bench-CL: Continual Learning for Coding Agents](https://arxiv.org/pdf/2507.00014)| 2025  | Arxiv
| [A Comprehensive Survey on Continual Learning in Generative Models](https://arxiv.org/pdf/2506.13045)| 2025  | Arxiv
| [The Future of Continual Learning in the Era of Foundation Models: Three Key Directions](https://arxiv.org/pdf/2506.03320)| 2025  | TCAI workshop 2025
| [Parameter-Efficient Continual Fine-Tuning: A Survey](https://arxiv.org/pdf/2504.13822)| 2025  | Arxiv
| [Latest Advancements Towards Catastrophic Forgetting under Data Scarcity: A Comprehensive Survey on Few-Shot Class Incremental Learning](https://arxiv.org/pdf/2502.08181)| 2025  | Arxiv
| [Federated Continual Learning: Concepts, Challenges, and Solutions](https://arxiv.org/pdf/2502.07059)| 2025  | Arxiv
| [Online Continual Learning: A Systematic Literature Review of Approaches, Challenges, and Benchmarks](https://arxiv.org/pdf/2501.04897)| 2025  | Arxiv
| [A Comprehensive Survey of Forgetting in Deep Learning Beyond Continual Learning](https://arxiv.org/pdf/2307.09218.pdf) | 2024  | TPAMI
| [Class-Incremental Learning: A Survey](https://arxiv.org/pdf/2302.03648.pdf) | 2024  | TPAMI
| [A Comprehensive Survey of Continual Learning: Theory, Method and Application](https://arxiv.org/pdf/2302.00487.pdf) | 2024  | TPAMI
| [Unleashing the Power of Continual Learning on Non-Centralized Devices: A Survey](https://arxiv.org/pdf/2412.13840)| 2024  | Arxiv
| [Federated Continual Learning for Edge-AI: A Comprehensive Survey](https://arxiv.org/pdf/2411.13740)| 2024  | Arxiv
| [Continual Learning with Neuromorphic Computing: Theories, Methods, and Applications](https://arxiv.org/pdf/2410.09218)| 2024  | Arxiv
| [Recent Advances of Multimodal Continual Learning: A Comprehensive Survey](https://arxiv.org/pdf/2410.05352)| 2024  | Arxiv
| [Towards General Industrial Intelligence: A Survey on Industrial IoT-Enhanced Continual Large Models](https://arxiv.org/pdf/2409.01207v1)| 2024  | Arxiv
| [Towards Lifelong Learning of Large Language Models: A Survey](https://arxiv.org/pdf/2406.06391)| 2024  | Arxiv
| [Recent Advances of Foundation Language Models-based Continual Learning: A Survey](https://arxiv.org/pdf/2405.18653)| 2024  | Arxiv
| [Continual Learning of Large Language Models: A Comprehensive Survey](https://arxiv.org/pdf/2404.16789) | 2024  | Arxiv
| [Continual Learning on Graphs: Challenges, Solutions, and Opportunities](https://arxiv.org/pdf/2402.11565.pdf)| 2024  | Arxiv
| [Continual Learning on Graphs: A Survey](https://arxiv.org/pdf/2402.06330.pdf)| 2024  | Arxiv
| [Continual Learning for Large Language Models: A Survey](https://arxiv.org/pdf/2402.01364.pdf)| 2024 | Arxiv
| [Continual Learning with Pre-Trained Models: A Survey](https://arxiv.org/pdf/2401.16386.pdf) | 2024 | IJCAI
| [A Survey on Few-Shot Class-Incremental Learning](https://www.sciencedirect.com/science/article/pii/S0893608023006019?via%3Dihub) | 2024 | Neural Networks
| [Sharpness and Gradient Aware Minimization for Memory-based Continual Learning](https://dl.acm.org/doi/10.1145/3628797.3629000) | 2023 | SOICT
| [A Survey on Incremental Update for Neural Recommender Systems](https://arxiv.org/pdf/2303.02851.pdf) | 2023  | Arxiv
| [Continual Graph Learning: A Survey](https://arxiv.org/pdf/2301.12230.pdf) | 2023  | Arxiv
| [Towards Label-Efficient Incremental Learning: A Survey](https://arxiv.org/pdf/2302.00353.pdf) |    2023  | Arxiv
| [Advancing continual lifelong learning in neural information retrieval: definition, dataset, framework, and empirical evaluation](https://arxiv.org/pdf/2308.08378.pdf) |   2023  | Arxiv
| [How to Reuse and Compose Knowledge for a Lifetime of Tasks: A Survey on Continual Learning and Functional Composition](https://arxiv.org/pdf/2207.07730.pdf) | 2023  |  Transactions on Machine Learning Research
| [Online Continual Learning in Image Classification: An Empirical Survey](https://arxiv.org/pdf/2101.10423.pdf) | 2022  |Neurocomputing
| [Class-incremental learning: survey and performance evaluation on image classification](https://arxiv.org/pdf/2010.15277.pdf) | 2022  |TPAMI
| [Towards Continual Reinforcement Learning: A Review and Perspectives](https://arxiv.org/pdf/2012.13490.pdf) | 2022  |Journal of Artificial Intelligence Research
| [An Introduction to Lifelong Supervised Learning](https://arxiv.org/pdf/2207.04354.pdf) | 2022 | Arxiv
| [Continual Learning for Real-World Autonomous Systems: Algorithms, Challenges and Frameworks](https://arxiv.org/pdf/2105.12374.pdf) |  2022 | Arxiv
| [A continual learning survey: Defying forgetting in classification tasks](https://arxiv.org/pdf/1909.08383.pdf) |  2021 | TPAMI
| [Recent Advances of Continual Learning in Computer Vision: An Overview](https://arxiv.org/pdf/2109.11369.pdf) | 2021  | Arxiv
| [Continual Lifelong Learning in Natural Language Processing: A Survey](https://aclanthology.org/2020.coling-main.574/) | 2020  |COLING
| [A Comprehensive Study of Class Incremental Learning Algorithms for Visual Tasks](https://arxiv.org/pdf/2011.01844.pdf) | 2020 | Neural Networks
| [Continual Lifelong Learning with Neural Networks: A Review](https://www.sciencedirect.com/science/article/pii/S0893608019300231) | 2019   | Neural Networks
| [Three scenarios for continual learning](https://arxiv.org/pdf/1904.07734.pdf) | 2018  | NeurIPSW
| [Lifelong Machine Learning](https://link.springer.com/book/10.1007/978-3-031-01575-5) | 2016  | Book

#### Task-aware CL
<a href="#top">[Back to top]</a>
> Task-aware CL focuses on addressing scenarios where explicit task definitions, such as task IDs or labels, are available during the CL process. Existing methods on task-aware CL have explored five main branches:   [Memory-based Methods](#memory-based-methods) |
  [Architecture-based Methods](#architecture-based-methods) |
  [Regularization-based Methods](#regularization-based-methods) |
  [Subspace-based Methods](#subspace-based-methods) |
  [Bayesian Methods](#bayesian-methods).

#####  Memory-based Methods
<a href="#top">[Back to top]</a>
> Memory-based (or Rehearsal-based) method keeps a memory buffer that stores the examples/knowledge from previous tasks and replays those examples during learning new tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Unlocking the Power of Rehearsal in Continual Learning: A Theoretical Perspective](https://arxiv.org/pdf/2506.00205)| 2025 | ICML
| [Autoencoder-Based Hybrid Replay for Class-Incremental Learning](https://arxiv.org/pdf/2505.05926)| 2025 | ICML
| [Do Your Best and Get Enough Rest for Continual Learning](https://arxiv.org/pdf/2503.18371)| 2025 | CVPR
| [Reducing Class-wise Confusion for Incremental Learning with Disentangled Manifolds](https://arxiv.org/pdf/2503.17677)| 2025 | CVPR
| [Towards Experience Replay for Class-Incremental Learning in Fully-Binary Networks](https://arxiv.org/pdf/2503.07107)| 2025 | Arxiv
| [Generative Binary Memory: Pseudo-Replay Class-Incremental Learning on Binarized Embeddings](https://arxiv.org/pdf/2503.10333)| 2025 | Arxiv
| [Sample Compression for Continual Learning](https://arxiv.org/pdf/2503.10503)| 2025 | Arxiv
| [STAR: Stability-Inducing Weight Perturbation for Continual Learning](https://arxiv.org/pdf/2503.01595)| 2025 | ICLR
| [Prior-free Balanced Replay: Uncertainty-guided Reservoir Sampling for Long-Tailed Continual Learning](https://arxiv.org/pdf/2408.14976)| 2024 | MM
| [FTF-ER: Feature-Topology Fusion-Based Experience Replay Method for Continual Graph Learning](https://arxiv.org/pdf/2407.19429)| 2024 | MM
| [Multi-layer Rehearsal Feature Augmentation for Class-Incremental Learning](https://openreview.net/pdf?id=aksdU1KOpT) | 2024 | ICML
| [Gradual Divergence for Seamless Adaptation: A Novel Domain Incremental Learning Method](https://arxiv.org/pdf/2406.16231) | 2024 | ICML
| [Accelerating String-Key Learned Index Structures via Memoization based Incremental Training](https://arxiv.org/pdf/2403.11472.pdf) | 2024 | VLDB
| [DSLR: Diversity Enhancement and Structure Learning for Rehearsal-based Graph Continual Learning](https://arxiv.org/pdf/2402.13711.pdf) | 2024 | WWW
| [Exemplar-based Continual Learning via Contrastive Learning](https://ieeexplore.ieee.org/abstract/document/10411956) | 2024 | IEEE Transactions on Artificial Intelligence
| [Saving 100x Storage: Prototype Replay for Reconstructing Training Sample Distribution in Class-Incremental Semantic Segmentation](https://openreview.net/pdf?id=Ct0zPIe3xs) |  2023 | NeurIPS
| [Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models](https://arxiv.org/pdf/2305.10120.pdf) | 2023 | NeurIPS
| [A Unified Approach to Domain Incremental Learning with Memory: Theory and Algorithm](https://arxiv.org/pdf/2310.12244.pdf)|  2023 | NeurIPS
| [An Efficient Dataset Condensation Plugin and Its Application to Continual Learning](https://openreview.net/pdf?id=Murj6wcjRw) | 2023 | NeurIPS
| [Augmented Memory Replay-based Continual Learning Approaches for Network Intrusion Detection](https://openreview.net/pdf?id=yGLokEhdh9) | 2023 | NeurIPS
| [Bilevel Coreset Selection in Continual Learning: A New Formulation and Algorithm](https://openreview.net/pdf?id=2dtU9ZbgSN) | 2023 | NeurIPS
| [FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning](https://arxiv.org/pdf/2309.14062.pdf) | 2023 | NeurIPS
| [Distributionally Robust Memory Evolution with Generalized Divergence for Continual Learning](https://www.computer.org/csdl/journal/tp/5555/01/10258417/1QEwVQys7ok) | 2023 | TPAMI
| [Improving Replay Sample Selection and Storage for Less Forgetting in Continual Learning](https://arxiv.org/pdf/2308.01895.pdf) | 2023 | ICCV
| [Masked Autoencoders are Efficient Class Incremental Learners](https://arxiv.org/pdf/2308.12510.pdf) | 2023 | ICCV
| [Error Sensitivity Modulation based Experience Replay: Mitigating Abrupt Representation Drift in Continual Learning](https://openreview.net/pdf?id=zlbci7019Z3) |2023 | ICLR
| [A Model or 603 Exemplars: Towards Memory-Efficient Class-Incremental Learning](https://arxiv.org/pdf/2205.13218.pdf)| 2023 | ICLR
| [DualHSIC: HSIC-Bottleneck and Alignment for Continual Learning](https://arxiv.org/pdf/2305.00380.pdf) | 2023 | ICML
| [DDGR: Continual Learning with Deep Diffusion-based Generative Replay](https://openreview.net/pdf?id=RlqgQXZx6r) | 2023 | ICML
| [BiRT: Bio-inspired Replay in Vision Transformers for Continual Learning](https://arxiv.org/pdf/2305.04769.pdf)| 2023 | ICML
| [Neuro-Symbolic Continual Learning: Knowledge, Reasoning Shortcuts and Concept Rehearsal](https://proceedings.mlr.press/v202/marconato23a/marconato23a.pdf)| 2023 | ICML
| [Poisoning Generative Replay in Continual Learning to Promote Forgetting](https://proceedings.mlr.press/v202/kang23c/kang23c.pdf)| 2023 | ICML
| [Regularizing Second-Order Influences for Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Regularizing_Second-Order_Influences_for_Continual_Learning_CVPR_2023_paper.pdf) | 2023|CVPR
| [Class-Incremental Exemplar Compression for Class-Incremental Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Luo_Class-Incremental_Exemplar_Compression_for_Class-Incremental_Learning_CVPR_2023_paper.pdf) | 2023|CVPR
| [A closer look at rehearsal-free continual learning](https://openaccess.thecvf.com/content/CVPR2023W/CLVision/papers/Smith_A_Closer_Look_at_Rehearsal-Free_Continual_Learning_CVPRW_2023_paper.pdf)| 2023|CVPRW
| [Continual Learning by Modeling Intra-Class Variation](https://openreview.net/forum?id=iDxfGaMYVr)| 2023 | TMLR
| [Class-Incremental Learning using Diffusion Model for Distillation and Replay](https://arxiv.org/pdf/2306.17560.pdf) | 2023 | Arxiv
| [On the Effectiveness of Lipschitz-Driven Rehearsal in Continual Learning](https://openreview.net/pdf?id=TThSwRTt4IB) | 2022 | NeurIPS
| [Exploring Example Influence in Continual Learning](https://openreview.net/pdf?id=u4dXcUEsN7B) | 2022 | NeurIPS
| [Navigating Memory Construction by Global Pseudo-Task Simulation for Continual Learning](https://openreview.net/pdf?id=tVbJdvMxK2-) | 2022 | NeurIPS
| [Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System](https://openreview.net/pdf?id=uxxFrDwrE7Y) | 2022 | ICLR
| [Information-theoretic Online Memory Selection for Continual Learning](https://openreview.net/pdf?id=IpctgL7khPp) | 2022 | ICLR
| [Memory Replay with Data Compression for Continual Learning](https://openreview.net/pdf?id=a7H7OucbWaU) | 2022 | ICLR
| [Improving Task-free Continual Learning by Distributionally Robust Memory Evolution](https://proceedings.mlr.press/v162/wang22v/wang22v.pdf) | 2022 | ICML
| [GCR: Gradient Coreset based Replay Buffer Selection for Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Tiwari_GCR_Gradient_Coreset_Based_Replay_Buffer_Selection_for_Continual_Learning_CVPR_2022_paper.pdf) | 2022 | CVPR
| [On the Convergence of Continual Learning with Adaptive Methods](https://proceedings.mlr.press/v216/han23a/han23a.pdf) | 2022 | UAI
| [RMM: Reinforced Memory Management for Class-Incremental Learning](https://proceedings.neurips.cc/paper_files/paper/2021/file/1cbcaa5abbb6b70f378a3a03d0c26386-Paper.pdf) | 2021 | NeurIPS
| [Rainbow Memory: Continual Learning with a Memory of Diverse Samples](https://openaccess.thecvf.com/content/CVPR2021/papers/Bang_Rainbow_Memory_Continual_Learning_With_a_Memory_of_Diverse_Samples_CVPR_2021_paper.pdf) | 2021|CVPR
| [Prototype Augmentation and Self-Supervision for Incremental Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf) | 2021|CVPR
| [Class-incremental experience replay for continual learning under concept drift](https://openaccess.thecvf.com/content/CVPR2021W/CLVision/papers/Korycki_Class-Incremental_Experience_Replay_for_Continual_Learning_Under_Concept_Drift_CVPRW_2021_paper.pdf) | 2021|CVPRW
| [Always Be Dreaming: A New Approach for Data-Free Class-Incremental Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Smith_Always_Be_Dreaming_A_New_Approach_for_Data-Free_Class-Incremental_Learning_ICCV_2021_paper.pdf) | 2021 | ICCV
| [Using Hindsight to Anchor Past Knowledge in Continual Learning](https://arxiv.org/pdf/2002.08165.pdf) | 2021 | AAAI
| [Improved Schemes for Episodic Memory-based Lifelong Learning](https://proceedings.neurips.cc/paper/2020/file/0b5e29aa1acf8bdc5d8935d7036fa4f5-Paper.pdf) | 2020 |NeurIPS
| [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf) | 2020 |NeurIPS
| [La-MAML: Look-ahead Meta Learning for Continual Learning](https://proceedings.neurips.cc/paper/2020/file/85b9a5ac91cd629bd3afe396ec07270a-Paper.pdf) | 2020 | NeurIPS
| [GAN Memory with No Forgetting](https://arxiv.org/pdf/2006.07543)| 2020 | NeurIPS
| [Brain-inspired replay for continual learning with artificial neural networks](https://pubmed.ncbi.nlm.nih.gov/32792531/) |2020 |Nature Communications
| [LAMOL: LAnguage MOdeling for Lifelong Language Learning](https://openreview.net/pdf?id=Skgxcn4YDS) | 2020 |ICLR
| [Mnemonics Training: Multi-Class Incremental Learning without Forgetting](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Mnemonics_Training_Multi-Class_Incremental_Learning_Without_Forgetting_CVPR_2020_paper.pdf) | 2020 | CVPR
| [GDumb: A Simple Approach that Questions Our Progress in Continual Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf) |2020| ECCV
| [Episodic Memory in Lifelong Language Learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/f8d2e80c1458ea2501f98a2cafadb397-Paper.pdf) | 2019 | NeurIPS
| [Continual Learning with Tiny Episodic Memories](https://arxiv.org/pdf/1902.10486.pdf) | 2019 | ICML |
| [Efficient lifelong learning with A-GEM](https://openreview.net/pdf?id=Hkf2_sC5FX) | 2019 | ICLR |
| [Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference](https://openreview.net/pdf?id=B1gTShAct7) | 2019 |ICLR
| [Large Scale Incremental Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf) | 2019 | CVPR
| [On Tiny Episodic Memories in Continual Learning](https://arxiv.org/pdf/1902.10486.pdf) | 2019 | Arxiv
| [Memory Replay GANs: learning to generate images from new categories without forgetting](https://proceedings.neurips.cc/paper_files/paper/2018/file/a57e8915461b83adefb011530b711704-Paper.pdf) | 2018 |NeurIPS
| [Progress & Compress: A scalable framework for continual learning](https://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf) | 2018 | ICML
| [Gradient Episodic Memory for Continual Learning](https://proceedings.neurips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf) | 2017 |NeurIPS
| [Continual Learning with Deep Generative Replay](https://proceedings.neurips.cc/paper_files/paper/2017/file/0efbe98067c6c73dba1250d2beaa81f9-Paper.pdf) | 2017 |NeurIPS
| [iCaRL: Incremental Classifier and Representation Learning](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf) | 2017| CVPR
| [Catastrophic forgetting, rehearsal and pseudorehearsal](https://www.tandfonline.com/doi/abs/10.1080/09540099550039318) | 1995 | Connection Science


#####  Architecture-based Methods
<a href="#top">[Back to top]</a>
> The architecture-based approach avoids forgetting by reducing parameter sharing between tasks or adding parameters to new tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [The Importance of Being Lazy: Scaling Limits of Continual Learning](https://arxiv.org/pdf/2506.16884)| 2025| ICML
| [Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective](https://arxiv.org/pdf/2506.03951)| 2025| ICML
| [Prototype Augmented Hypernetworks for Continual Learning](https://arxiv.org/pdf/2505.07450) | 2025| CVPR
| [Self-Controlled Dynamic Expansion Model for Continual Learning](https://arxiv.org/pdf/2504.10561)| 2025 | Arxiv
| [KAC: Kolmogorov-Arnold Classifier for Continual Learning](https://arxiv.org/pdf/2503.21076)| 2024 | CVPR
| [CEAT: Continual Expansion and Absorption Transformer for Non-Exemplar Class-Incremental Learning](https://arxiv.org/pdf/2403.06670)| 2024 | TCSVT
| [Harnessing Neural Unit Dynamics for Effective and Scalable Class-Incremental Learning](https://arxiv.org/pdf/2406.02428) | 2024 | ICML
| [Revisiting Neural Networks for Continual Learning: An Architectural Perspective](https://arxiv.org/pdf/2404.14829) | 2024 | IJCAI
| [Recall-Oriented Continual Learning with Generative Adversarial Meta-Model](https://arxiv.org/pdf/2403.03082.pdf) | 2024 | AAAI
| [Divide and not forget: Ensemble of selectively trained experts in Continual Learning](https://openreview.net/pdf?id=sSyytcewxe) | 2024 | ICLR
| [A Probabilistic Framework for Modular Continual Learning](https://openreview.net/pdf?id=MVe2dnWPCu) | 2024 | ICLR
| [Incorporating neuro-inspired adaptability for continual learning in artificial intelligence](https://www.nature.com/articles/s42256-023-00747-w) | 2023 | Nature Machine Intelligence
| [TriRE: A Multi-Mechanism Learning Paradigm for Continual Knowledge Retention and Promotion](https://arxiv.org/pdf/2310.08217.pdf) | 2023 | NeurIPS
| [ScrollNet: Dynamic Weight Importance for Continual Learning](https://arxiv.org/pdf/2308.16567.pdf) | 2023 | ICCV
| [CLR: Channel-wise Lightweight Reprogramming for Continual Learning](https://arxiv.org/pdf/2307.11386.pdf) | 2023 | ICCV
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
| [Calibrating CNNs for Lifelong Learning](https://proceedings.neurips.cc/paper_files/paper/2020/file/b3b43aeeacb258365cc69cdaf42a68af-Paper.pdf) | 2020 | NeurIPS
| [Continual Learning of a Mixed Sequence of Similar and Dissimilar Tasks](https://proceedings.neurips.cc/paper/2020/file/d7488039246a405baf6a7cbc3613a56f-Paper.pdf) | 2020 | NeurIPS
| [Compacting, Picking and Growing for Unforgetting Continual Learning](https://proceedings.neurips.cc/paper/2019/file/3b220b436e5f3d917a1e649a0dc0281c-Paper.pdf) | 2019 | NeurIPS
| [Superposition of many models into one](https://papers.nips.cc/paper_files/paper/2019/file/4c7a167bb329bd92580a99ce422d6fa6-Paper.pdf)  | 2019 | NeurIPS
| [Reinforced Continual Learning](https://proceedings.neurips.cc/paper_files/paper/2018/file/cee631121c2ec9232f3a2f028ad5c89b-Paper.pdf) | 2018 | NeurIPS
| [Progress & Compress: A scalable framework for continual learning](https://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf) | 2018 | ICML
| [Overcoming Catastrophic Forgetting with Hard Attention to the Task](https://arxiv.org/pdf/1801.01423.pdf) |2018 | ICML
| [Lifelong Learning with Dynamically Expandable Networks ](https://openreview.net/pdf?id=Sk7KsfW0-) | 2018 | ICLR
| [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.pdf) | 2018 | CVPR
| [Expert Gate: Lifelong Learning with a Network of Experts](https://openaccess.thecvf.com/content_cvpr_2017/papers/Aljundi_Expert_Gate_Lifelong_CVPR_2017_paper.pdf) |2017 | CVPR
| [Progressive Neural Networks](https://arxiv.org/pdf/1606.04671.pdf) | 2016 | Arxiv

#####  Regularization-based Methods
<a href="#top">[Back to top]</a>
> Regularization-based approaches avoid forgetting by penalizing updates of important parameters or distilling knowledge with previous model as a teacher.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Parabolic Continual Learning](https://arxiv.org/pdf/2503.02117)|2025 | Arxiv
| [No Forgetting Learning: Memory-free Continual Learning](https://arxiv.org/pdf/2503.04638)|2025 | Arxiv
| [On the Computation of the Fisher Information in Continual Learning](https://arxiv.org/pdf/2502.11756)|2025 | ICLR
| [Rehearsal-Free Continual Federated Learning with Synergistic Regularization](https://arxiv.org/pdf/2412.13779)|2024 | Arxiv
| [A Statistical Theory of Regularization-Based Continual Learning](https://arxiv.org/pdf/2406.06213)|2024 | ICML
| [IMEX-Reg: Implicit-Explicit Regularization in the Function Space for Continual Learning](https://arxiv.org/pdf/2404.18161)| 2024 | TMLR
| [Contrastive Continual Learning with Importance Sampling and Prototype-Instance Relation Distillation](https://arxiv.org/pdf/2403.04599.pdf) | 2024 | AAAI
| [Elastic Feature Consolidation for Cold Start Exemplar-free Incremental Learning](https://arxiv.org/pdf/2402.03917.pdf) | 2024 | ICLR
| [Fine-Grained Knowledge Selection and Restoration for Non-Exemplar Class Incremental Learning](https://arxiv.org/pdf/2312.12722.pdf) | 2024 | AAAI
| [Prototype-Sample Relation Distillation: Towards Replay-Free Continual Learning](https://arxiv.org/pdf/2303.14771.pdf) | 2023 | ICML
| [Continual Learning via Sequential Function-Space Variational Inference](https://arxiv.org/pdf/2312.17210.pdf) | 2022 | ICML
| [Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation](https://openaccess.thecvf.com/content/CVPR2022/papers/Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response_CVPR_2022_paper.pdf) | 2022 | CVPR
| [Class-Incremental Learning by Knowledge Distillation with Adaptive Feature Consolidation](https://openaccess.thecvf.com/content/CVPR2022/papers/Kang_Class-Incremental_Learning_by_Knowledge_Distillation_With_Adaptive_Feature_Consolidation_CVPR_2022_paper.pdf) | 2022 | CVPR
| [Class-Incremental Learning via Knowledge Amalgamation](https://arxiv.org/pdf/2209.02112)| 2022 | PKDD
| [Natural continual learning: success is a journey, not (just) a destination](https://proceedings.neurips.cc/paper/2021/file/ec5aa0b7846082a2415f0902f0da88f2-Paper.pdf) | 2021 | NeurIPS
| [Distilling Causal Effect of Data in Class-Incremental Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_Distilling_Causal_Effect_of_Data_in_Class-Incremental_Learning_CVPR_2021_paper.pdf) | 2021 | CVPR
| [On Learning the Geodesic Path for Incremental Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Simon_On_Learning_the_Geodesic_Path_for_Incremental_Learning_CVPR_2021_paper.pdf) | 2021 | CVPR
| [CPR: Classifier-Projection Regularization for Continual Learning](https://openreview.net/pdf?id=F2v4aqEL6ze) | 2021 | ICLR
| [Few-Shot Class-Incremental Learning via Relation Knowledge Distillation](https://ojs.aaai.org/index.php/AAAI/article/view/16213) | 2021 | AAAI
| [Continual Learning with Node-Importance based Adaptive Group Sparse Regularization](https://proceedings.neurips.cc/paper/2020/file/258be18e31c8188555c2ff05b4d542c3-Paper.pdf) | 2020 | NeurIPS
| [PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650086.pdf) | 2020 | ECCV
| [Topology-Preserving Class-Incremental Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640256.pdf) | 2020 | ECCV
| [Uncertainty-based Continual Learning with Adaptive Regularization](https://proceedings.neurips.cc/paper_files/paper/2019/file/2c3ddf4bf13852db711dd1901fb517fa-Paper.pdf) | 2019 |NeurIPS
| [Learning a Unified Classifier Incrementally via Rebalancing](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf) | 2019 | CVPR
| [Learning Without Memorizing](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dhar_Learning_Without_Memorizing_CVPR_2019_paper.pdf) | 2019 | CVPR
| [Efficient Lifelong Learning with A-GEM](https://openreview.net/pdf?id=Hkf2_sC5FX) | 2019| ICLR
| [Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence](https://openaccess.thecvf.com/content_ECCV_2018/papers/Arslan_Chaudhry__Riemannian_Walk_ECCV_2018_paper.pdf) | 2018 | ECCV
| [Lifelong Learning via Progressive Distillation and Retrospection](https://openaccess.thecvf.com/content_ECCV_2018/papers/Saihui_Hou_Progressive_Lifelong_Learning_ECCV_2018_paper.pdf) | 2018 | ECCV
| [Memory Aware Synapses: Learning what (not) to forget](https://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf) | 2018 | ECCV
| [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/pdf/1612.00796v2.pdf) | 2017 | Arxiv
| [Continual Learning Through Synaptic Intelligence](https://dl.acm.org/doi/pdf/10.5555/3305890.3306093) | 2017 | ICML
| [Learning without Forgetting](https://ieeexplore.ieee.org/document/8107520) |2017 | TPAMI



#####  Subspace-based Methods
<a href="#top">[Back to top]</a>
> Subspace-based methods perform CL in multiple disjoint subspaces to avoid interference between multiple tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Continuous Subspace Optimization for Continual Learning](https://arxiv.org/pdf/2505.11816)| 2025 | Arxiv
| [Geodesic-Aligned Gradient Projection for Continual Task Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10934731)|2025|TIP
| [Revisiting Flatness-aware Optimization in Continual Learning with Orthogonal Gradient Projection](https://ieeexplore.ieee.org/abstract/document/10874188)|2025|TPAMI
| [Introducing Common Null Space of Gradients for Gradient Projection Methods in Continual Learning](https://openreview.net/pdf?id=N3yngE4fTy)| 2024 | ACM MM
| [Improving Data-aware and Parameter-aware Robustness for Continual Learning](https://arxiv.org/abs/2405.17054v1)| 2024 | Arxiv
| [Prompt Gradient Projection for Continual Learning](https://openreview.net/pdf?id=EH2O3h7sBI) | 2024 | ICLR
| [Hebbian Learning based Orthogonal Projection for Continual Learning of Spiking Neural Networks](https://openreview.net/pdf?id=MeB86edZ1P) | 2024 | ICLR
| [Towards Continual Learning Desiderata via HSIC-Bottleneck Orthogonalization and Equiangular Embedding](https://arxiv.org/pdf/2401.09067.pdf)  | 2024 | AAAI
| [Orthogonal Subspace Learning for Language Model Continual Learning](https://arxiv.org/pdf/2310.14152.pdf) | 2023 | EMNLP
| [Data Augmented Flatness-aware Gradient Projection for Continual Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Data_Augmented_Flatness-aware_Gradient_Projection_for_Continual_Learning_ICCV_2023_paper.pdf) | 2023 | ICCV
| [Rethinking Gradient Projection Continual Learning: Stability / Plasticity Feature Space Decoupling](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Rethinking_Gradient_Projection_Continual_Learning_Stability__Plasticity_Feature_Space_CVPR_2023_paper.pdf) | 2023 | CVPR
| [Building a Subspace of Policies for Scalable Continual Learning](https://openreview.net/pdf?id=ZloanUtG4a) | 2023 | ICLR
| [Continual Learning with Scaled Gradient Projection](https://arxiv.org/pdf/2302.01386.pdf) | 2023 | AAAI
| [SketchOGD: Memory-Efficient Continual Learning](https://arxiv.org/pdf/2305.16424.pdf) | 2023 | Arxiv
| [Continual Learning through Networks Splitting and Merging with Dreaming-Meta Weighted Model Fusion](https://arxiv.org/pdf/2312.07082)| 2023 | Arxiv
| [Beyond Not-Forgetting: Continual Learning with Backward Knowledge Transfer](https://openreview.net/pdf?id=diV1PpaP33) | 2022 | NeurIPS
| [TRGP: Trust Region Gradient Projection for Continual Learning](https://openreview.net/pdf?id=iEvAf8i6JjO) | 2022 | ICLR
| [Continual Learning with Recursive Gradient Optimization](https://openreview.net/pdf?id=7YDLgf9_zgm) | 2022 | ICLR
| [Class Gradient Projection For Continual Learning](https://dl.acm.org/doi/pdf/10.1145/3503161.3548054) | 2022 | MM
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
<a href="#top">[Back to top]</a>
> Bayesian methods provide a principled probabilistic framework for addressing Forgetting.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Bayesian continual learning and forgetting in neural networks](https://arxiv.org/pdf/2504.13569)| 2025 |Arxiv
| [Learning to Continually Learn with the Bayesian Principle](https://arxiv.org/pdf/2405.18758) |2024 | ICML
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
<a href="#top">[Back to top]</a>
> Task-free CL refers to a specific scenario that the learning system does not have access to any explicit task information.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Task-Free Continual Generation and Representation Learning via Dynamic Expansionable Memory Cluster](https://ojs.aaai.org/index.php/AAAI/article/view/29582) | 2024 | AAAI
| [Task-Free Dynamic Sparse Vision Transformer for Continual Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29581) | 2024 | AAAI
| [Doubly Perturbed Task-Free Continual Learning](https://arxiv.org/pdf/2312.13027.pdf) | 2024 | AAAI
| [Loss Decoupling for Task-Agnostic Continual Learning](https://openreview.net/pdf?id=9Oi3YxIBSa) | 2023 | NeurIPS
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
<a href="#top">[Back to top]</a>
> In online CL, the learner is only allowed to process the data for each task once.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning](https://arxiv.org/pdf/2505.19680)|2025|ICML
| [Amphibian: A Meta-Learning Framework for Rehearsal-Free, Fast Online Continual Learning](https://openreview.net/forum?id=n4AaKOBWbB)|2025|TMLR
| [Ferret: An Efficient Online Continual Learning Framework under Varying Memory Constraints](https://arxiv.org/pdf/2503.12053)|2025|CVPR
| [Alchemist: Towards the Design of Efficient Online Continual Learning System](https://arxiv.org/pdf/2503.01066) |2025|Arxiv
| [Online Curvature-Aware Replay: Leveraging 2nd Order Information for Online Continual Learning](https://arxiv.org/pdf/2502.01866) |2025|Arxiv
| [Dealing with Synthetic Data Contamination in Online Continual Learning](https://arxiv.org/pdf/2411.13852)| 2024 | NeurIPS
| [Random Representations Outperform Online Continually Learned Representations](https://cdn.iiit.ac.in/cdn/precog.iiit.ac.in/pubs/NeurIPS-RanDumb.pdf)| 2024 | NeurIPS
| [Forgetting, Ignorance or Myopia: Revisiting Key Challenges in Online Continual Learning](https://arxiv.org/pdf/2409.19245)| 2024 | NeurIPS
| [Mitigating Catastrophic Forgetting in Online Continual Learning by Modeling Previous Task Interrelations via Pareto Optimization](https://openreview.net/pdf?id=olbTrkWo1D)| 2024 | ICML
| [ER-FSL: Experience Replay with Feature Subspace Learning for Online Continual Learning](https://arxiv.org/pdf/2407.12279)| 2024 | MM
| [Dual-Enhanced Coreset Selection with Class-wise Collaboration for Online Blurry Class Incremental Learning](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_Dual-Enhanced_Coreset_Selection_with_Class-wise_Collaboration_for_Online_Blurry_Class_CVPR_2024_paper.pdf)|2024 | CVPR
| [Orchestrate Latent Expertise: Advancing Online Continual Learning with Multi-Level Supervision and Reverse Self-Distillation](https://arxiv.org/pdf/2404.00417.pdf)|2024 | CVPR
| [Learning Equi-angular Representations for Online Continual Learning](https://arxiv.org/pdf/2404.01628.pdf)|2024 | CVPR
| [Online Continual Learning For Interactive Instruction Following Agents](https://arxiv.org/pdf/2403.07548.pdf)|2024 | ICLR
| [Online Continual Learning for Interactive Instruction Following Agents](https://openreview.net/pdf?id=7M0EzjugaN)|2024 | ICLR
| [Summarizing Stream Data for Memory-Constrained Online Continual Learning](https://arxiv.org/pdf/2305.16645.pdf)|2024 | AAAI
| [Online Class-Incremental Learning For Real-World Food Image Classification](https://openaccess.thecvf.com/content/WACV2024/papers/Raghavan_Online_Class-Incremental_Learning_for_Real-World_Food_Image_Classification_WACV_2024_paper.pdf)|2024 | WACV
| [Rapid Adaptation in Online Continual Learning: Are We Evaluating It Right?](https://openaccess.thecvf.com/content/ICCV2023/papers/Al_Kader_Hammoud_Rapid_Adaptation_in_Online_Continual_Learning_Are_We_Evaluating_It_ICCV_2023_paper.pdf) |2023 | ICCV
| [CBA: Improving Online Continual Learning via Continual Bias Adaptor](https://arxiv.org/pdf/2308.06925.pdf) | 2023 | ICCV
| [Online Continual Learning on Hierarchical Label Expansion](https://arxiv.org/pdf/2308.14374.pdf) | 2023 | ICCV
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
| [Replay Without Saving: Prototype Derivation and Distribution Rebalance for Class-Incremental Semantic Segmentation](https://ieeexplore.ieee.org/document/10904177?denied=)| 2025 | TPAMI
| [Towards Macro-AUC oriented Imbalanced Multi-Label Continual Learning](https://arxiv.org/pdf/2412.18231) | 2025 | IJCAI
| [Joint Input and Output Coordination for Class-Incremental Learning](https://www.ijcai.org/proceedings/2024/0565.pdf)| 2024 | IJCAI
| [Imbalance Mitigation for Continual Learning via Knowledge Decoupling and Dual Enhanced Contrastive Learning](https://ieeexplore.ieee.org/abstract/document/10382590) | 2024 | TNNLS
| [Overcoming Recency Bias of Normalization Statistics in Continual Learning: Balance and Adaptation](https://openreview.net/pdf?id=Ph65E1bE6A) | 2023 | NeurIPS
| [Online Bias Correction for Task-Free Continual Learning](https://openreview.net/pdf?id=18XzeuYZh_) | 2023| ICLR
| [Information-theoretic Online Memory Selection for Continual Learning](https://openreview.net/pdf?id=IpctgL7khPp) | 2022 | ICLR
| [SS-IL: Separated Softmax for Incremental Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Ahn_SS-IL_Separated_Softmax_for_Incremental_Learning_ICCV_2021_paper.pdf) |2021 | ICCV
| [Online Continual Learning from Imbalanced Data](http://proceedings.mlr.press/v119/chrysakis20a/chrysakis20a.pdf) | 2020 | ICML
| [Maintaining Discrimination and Fairness in Class Incremental Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Maintaining_Discrimination_and_Fairness_in_Class_Incremental_Learning_CVPR_2020_paper.pdf) |2020 | CVPR
| [Semantic Drift Compensation for Class-Incremental Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Semantic_Drift_Compensation_for_Class-Incremental_Learning_CVPR_2020_paper.pdf) |2020 | CVPR
| [Imbalanced Continual Learning with Partitioning Reservoir Sampling](https://arxiv.org/pdf/2009.03632.pdf) | 2020| ECCV
| [GDumb: A Simple Approach that Questions Our Progress in Continual Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf) |2020 | ECCV
| [Large scale incremental learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf) | 2019 | CVPR
| [IL2M: Class Incremental Learning With Dual Memory](https://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf) | 2019|ICCV
| [End-to-end incremental learning](https://arxiv.org/pdf/1807.09536.pdf) | 2018 | ECCV



#### Semi-supervised CL
<a href="#top">[Back to top]</a>
> Semi-supervised CL is an extension of traditional CL that allows each task to incorporate unlabeled data as well.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Continual Learning on a Diet: Learning from Sparsely Labeled Streams Under Constrained Computation](https://openreview.net/pdf?id=Xvfz8NHmCj)| 2024 | ICLR
| [Dynamic Sub-graph Distillation for Robust Semi-supervised Continual Learning](https://arxiv.org/pdf/2312.16409.pdf) | 2024 | AAAI
| [Semi-supervised drifted stream learning with short lookback](https://arxiv.org/pdf/2205.13066.pdf) | 2022 | SIGKDD
| [Ordisco: Effective and efficient usage of incremental unlabeled data for semi-supervised continual learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_ORDisCo_Effective_and_Efficient_Usage_of_Incremental_Unlabeled_Data_for_CVPR_2021_paper.pdf) | 2021 | CVPR
| [Memory-Efficient Semi-Supervised Continual Learning: The World is its Own Replay Buffer](https://arxiv.org/pdf/2101.09536.pdf) | 2021| IJCNN
| [Overcoming Catastrophic Forgetting with Unlabeled Data in the Wild](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_Overcoming_Catastrophic_Forgetting_With_Unlabeled_Data_in_the_Wild_ICCV_2019_paper.pdf) | 2019 | ICCV


#### Few-shot CL
<a href="#top">[Back to top]</a>
> Few-shot CL refers to the scenario where a model needs to learn new tasks with only a limited number of labeled examples per task while retaining knowledge from previously encountered tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [A New Benchmark for Few-Shot Class-Incremental Learning: Redefining the Upper Bound](https://arxiv.org/pdf/2503.10003) | 2025 | Arxiv |
| [Improving Open-world Continual Learning under the Constraints of Scarce Labeled Data](https://arxiv.org/pdf/2502.20974) | 2025 | Arxiv |
| [Wearable Sensor-Based Few-Shot Continual Learning on Hand Gestures for Motor-Impaired Individuals via Latent Embedding Exploitation](https://arxiv.org/pdf/2405.08969) | 2024 | IJCAI
| [A Bag of Tricks for Few-Shot Class-Incremental Learning](https://arxiv.org/pdf/2403.14392.pdf)| 2024 | Arxiv
| [Analogical Learning-Based Few-Shot Class-Incremental Learning](https://ieeexplore.ieee.org/abstract/document/10382651) | 2024 | IEEE TCSVT
| [Few-Shot Class-Incremental Learning via Training-Free Prototype Calibration](https://openreview.net/pdf?id=8NAxGDdf7H) |2023 | NeurIPS
| [Few-shot Class-incremental Learning: A Survey](https://arxiv.org/pdf/2308.06764.pdf) | 2023 | Arxiv
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

<!-- | [Neural Collapse Terminus: A Unified Solution for Class Incremental Learning and Its Variants](https://arxiv.org/pdf/2308.01746.pdf) | 2023 | Arxiv -->

#### Unsupervised CL
<a href="#top">[Back to top]</a>
> Unsupervised CL (UCL) assumes that only unlabeled data is provided to the CL learner.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Class-Incremental Unsupervised Domain Adaptation via Pseudo-Label Distillation](https://pubmed.ncbi.nlm.nih.gov/38285572/) | 2024 | TIP
| [Plasticity-Optimized Complementary Networks for Unsupervised Continual](https://arxiv.org/pdf/2309.06086.pdf) | 2024 | WACV
| [Unsupervised Continual Learning in Streaming Environments](https://ieeexplore.ieee.org/document/9756660) | 2023 | TNNLS
| [Representational Continuity for Unsupervised Continual Learning](https://openreview.net/pdf?id=9Hrka5PA7LW) | 2022 | ICLR
| [Probing Representation Forgetting in Supervised and Unsupervised Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Davari_Probing_Representation_Forgetting_in_Supervised_and_Unsupervised_Continual_Learning_CVPR_2022_paper.pdf) |2022 | CVPR
| [Unsupervised Continual Learning for Gradually Varying Domains](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Taufique_Unsupervised_Continual_Learning_for_Gradually_Varying_Domains_CVPRW_2022_paper.pdf) | 2022 | CVPRW
| [Co2L: Contrastive Continual Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Cha_Co2L_Contrastive_Continual_Learning_ICCV_2021_paper.pdf) |2021 | ICCV
| [Unsupervised Progressive Learning and the STAM Architecture](https://www.ijcai.org/proceedings/2021/0410.pdf) | 2021 | IJCAI
| [Continual Unsupervised Representation Learning](https://proceedings.neurips.cc/paper_files/paper/2019/file/861578d797aeb0634f77aff3f488cca2-Paper.pdf) | 2019 | NeurIPS


#### Theoretical Analysis
<a href="#top">[Back to top]</a>
> Theory or analysis of continual learning

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Convergence and Implicit Bias of Gradient Descent on Continual Linear Classification](https://arxiv.org/pdf/2504.12712)|2025 | Arxiv
| [Theoretical Insights into Overparameterized Models in Multi-Task and Replay-Based Continual Learning](https://arxiv.org/pdf/2408.16939)|2024 | Arxiv
| [An analysis of best-practice strategies for replay and rehearsal in continual learning](https://openaccess.thecvf.com/content/CVPR2024W/CLVISION/papers/Krawczyk_An_Analysis_of_Best-practice_Strategies_for_Replay_and_Rehearsal_in_CVPRW_2024_paper.pdf)|2024 | CVPRW
| [Provable Contrastive Continual Learning](https://arxiv.org/pdf/2405.18756)|2024 | ICML
| [A Statistical Theory of Regularization-Based Continual Learning](https://arxiv.org/pdf/2406.06213)|2024 | ICML
| [Efficient Continual Finite-Sum Minimization](https://arxiv.org/pdf/2406.04731)| 2024 | ICLR
| [Provable Contrastive Continual Learning](https://arxiv.org/pdf/2405.18756)| 2024 | ICLR
| [Understanding Forgetting in Continual Learning with Linear Regression: Overparameterized and Underparameterized Regimes](https://arxiv.org/pdf/2405.17583)| 2024 | ICLR
| [The Joint Effect of Task Similarity and Overparameterization on Catastrophic Forgetting -- An Analytical Model](https://arxiv.org/pdf/2401.12617.pdf)| 2024 | ICLR
| [A Unified and General Framework for Continual Learning](https://openreview.net/pdf?id=BE5aK0ETbp)| 2024 | ICLR
| [Continual Learning in the Presence of Spurious Correlations: Analyses and a Simple Baseline](https://openreview.net/pdf?id=3Y7r6xueJJ)| 2024 | ICLR
| [On the Convergence of Continual Learning with Adaptive Methods](https://arxiv.org/pdf/2404.05555.pdf)| 2023 | UAI
| [Does Continual Learning Equally Forget All Parameters? ](https://proceedings.mlr.press/v202/zhao23n/zhao23n.pdf)| 2023 | ICML
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
<a href="#top">[Back to top]</a>

> Foundation models are large machine learning models trained on a vast quantity of data at scale, such that they can be adapted to a wide range of downstream tasks.

**Links**: [Forgetting in Fine-Tuning Foundation Models](#forgetting-in-fine-tuning-foundation-models) | [Forgetting in One-Epoch Pre-training](#forgetting-in-one-epoch-pre-training) | [CL in Foundation Model](#cl-in-foundation-model)

#### Forgetting in Fine-Tuning Foundation Models
<a href="#top">[Back to top]</a>
> When fine-tuning a foundation model, there is a tendency to forget the pre-trained knowledge, resulting in sub-optimal performance on downstream tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Upweighting Easy Samples in Fine-Tuning Mitigates Forgetting](https://arxiv.org/pdf/2502.02797)| 2025 | Arxiv
| [AURORA-M: Open Source Continual Pre-training for Multilingual Language and Code](https://aclanthology.org/2025.coling-industry.56.pdf)| 2025| Coling
| [Continual Learning Using a Kernel-Based Method Over Foundation Models](https://arxiv.org/pdf/2412.15571)| 2024| Arxiv
| [A Practitioner’s Guide to Continual Multimodal Pretraining](https://arxiv.org/pdf/2408.14471)| 2024| Arxiv
| [SLCA++: Unleash the Power of Sequential Fine-tuning for Continual Learning with Pre-training](https://arxiv.org/pdf/2408.08295)| 2024| Arxiv
| [MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning](https://arxiv.org/pdf/2407.20999)| 2024 | Arxiv
| [Towards Effective and Efficient Continual Pre-training of Large Language Models](https://arxiv.org/pdf/2407.18743)| 2024 | Arxiv
| [Revisiting Catastrophic Forgetting in Large Language Model Tuning](https://arxiv.org/abs/2406.04836)| 2024 | Arxiv
| [D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models](https://arxiv.org/pdf/2406.01375) | 2024 | Arxiv
| [Dissecting learning and forgetting in language model finetuning](https://openreview.net/forum?id=tmsqb6WpLz)| 2024 | ICLR
| [Understanding Catastrophic Forgetting in Language Models via Implicit Inference](https://openreview.net/attachment?id=VrHiF2hsrm&name=pdf)| 2024 | ICLR
| [Two-stage LLM Fine-tuning with Less Specialization and More Generalization](https://arxiv.org/abs/2211.00635)| 2024 | ICLR
| [What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement](https://arxiv.org/pdf/2402.01865.pdf)| 2024 | Arxiv
| [Scaling Laws for Forgetting When Fine-Tuning Large Language Models](https://arxiv.org/pdf/2401.05605.pdf)| 2024 | Arxiv
| [TOFU: A Task of Fictitious Unlearning for LLMs](https://arxiv.org/pdf/2401.06121.pdf) | 2024 | Arxiv
| [Self-regulating Prompts: Foundational Model Adaptation without Forgetting](https://openaccess.thecvf.com/content/ICCV2023/papers/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.pdf) | 2023| ICCV
| [Speciality vs Generality: An Empirical Study on Catastrophic Forgetting in Fine-tuning Foundation Models](https://arxiv.org/pdf/2309.06256.pdf) | 2023 | Arxiv
| [Continual Pre-Training of Large Language Models: How to (re)warm your model?](https://arxiv.org/pdf/2308.04014.pdf) | 2023 | ICMLW
| [Improving Gender Fairness of Pre-Trained Language Models without Catastrophic Forgetting](https://arxiv.org/pdf/2110.05367.pdf) |2023 | ACL
| [On The Role of Forgetting in Fine-Tuning Reinforcement Learning Models](https://openreview.net/pdf?id=zmXJUKULDzh) | 2023 | ICLRW
| [Retentive or Forgetful? Diving into the Knowledge Memorizing Mechanism of Language Models](https://arxiv.org/pdf/2305.09144.pdf)| 2023 | Arxiv
| [Reinforcement Learning with Action-Free Pre-Training from Videos](https://proceedings.mlr.press/v162/seo22a/seo22a.pdf) | 2022 | ICML
| [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](https://openreview.net/pdf?id=AXDNM76T1nc) | 2022 |NeurIPS
| [On Reinforcement Learning and Distribution Matching for Fine-Tuning Language Models with no Catastrophic Forgetting](https://proceedings.neurips.cc/paper_files/paper/2022/file/67496dfa96afddab795530cc7c69b57a-Paper-Conference.pdf) | 2022 |NeurIPS
| [How Should Pre-Trained Language Models Be Fine-Tuned Towards Adversarial Robustness?](https://proceedings.neurips.cc/paper/2021/file/22b1f2e0983160db6f7bb9f62f4dbb39-Paper.pdf) | 2021 | NeurIPS
| [Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models](https://openreview.net/pdf?id=HkgaETNtDB) | 2020 | ICLR
| [Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting](https://aclanthology.org/2020.emnlp-main.634.pdf) | 2020 | EMNLP
| [Universal Language Model Fine-tuning for Text Classification](https://aclanthology.org/P18-1031.pdf) | 2018 | ACL



#### Forgetting in One-Epoch Pre-training
<a href="#top">[Back to top]</a>
> Foundation models often undergo training on a dataset for a single pass. As a result, the earlier examples encountered during pre-training may be overwritten or forgotten by the model more quickly than the later examples.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Efficient Continual Pre-training of LLMs for Low-resource Languages](https://arxiv.org/pdf/2412.10244)|2024 | Arxiv
| [Exploring Forgetting in Large Language Model Pre-Training](https://arxiv.org/pdf/2410.17018)|2024 | Arxiv
| [Measuring Forgetting of Memorized Training Examples](https://openreview.net/pdf?id=7bJizxLKrR) | 2023 | ICLR
| [Quantifying Memorization Across Neural Language Models](https://openreview.net/pdf?id=TatRHT_1cK) | 2023| ICLR
| [Analyzing leakage of personally identifiable information in language models](https://arxiv.org/pdf/2302.00539.pdf) | 2023|S\&P
| [How Well Does Self-Supervised Pre-Training Perform with Streaming Data?](https://arxiv.org/pdf/2104.12081.pdf) | 2022| ICLR
| [The challenges of continuous self-supervised learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860687.pdf) | 2022 | ECCV
| [Continual contrastive learning for image classification](https://ieeexplore.ieee.org/document/9859995) | 2022 | ICME


#### CL in Foundation Model or Pretrained Model
<a href="#top">[Back to top]</a>
> By leveraging the powerful feature extraction capabilities of foundation models, researchers have been able to explore new avenues for advancing continual learning techniques.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training](https://arxiv.org/pdf/2507.05386)| 2025 | Arxiv
| [Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](https://arxiv.org/pdf/2507.02503)| 2025 | ACL
| [TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree](https://arxiv.org/pdf/2506.10355)| 2025 | ICML
| [Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning](https://arxiv.org/pdf/2506.03850)| 2025 | ICML
| [Proxy-FDA: Proxy-based Feature Distribution Alignment for Fine-tuning Vision Foundation Models without Forgetting](https://arxiv.org/pdf/2505.24088)| 2025 | ICML
| [Breaking Data Silos: Towards Open and Scalable Mobility Foundation Models via Generative Continual Learning](https://arxiv.org/pdf/2506.06694)| 2025 | Arxiv
| [Leveraging Pre-Trained Models for Multimodal Class-Incremental Learning under Adaptive Fusion](https://arxiv.org/pdf/2506.09999)| 2025 | Arxiv
| [Can LLMs Alleviate Catastrophic Forgetting in Graph Continual Learning? A Systematic Study](https://arxiv.org/pdf/2505.18697)| 2025 | Arxiv
| [Dynamic Mixture of Progressive Parameter-Efficient Expert Library for Lifelong Robot Learning](https://arxiv.org/pdf/2506.05985)| 2025 | Arxiv
| [Beyond Freezing: Sparse Tuning Enhances Plasticity in Continual Learning with Pre-Trained Models](https://arxiv.org/pdf/2505.19943)| 2025 | Arxiv
| [SplitLoRA: Balancing Stability and Plasticity in Continual Learning Through Gradient Space Splitting](https://arxiv.org/pdf/2505.22370)| 2025 | Arxiv
| [Budget-Adaptive Adapter Tuning in Orthogonal Subspaces for Continual Learning in LLMs](https://arxiv.org/pdf/2505.22358)| 2025 | Arxiv
| [LADA: Scalable Label-Specific CLIP Adapter for Continual Learning](https://arxiv.org/pdf/2505.23271)| 2025 | ICML
| [Efficient Federated Class-Incremental Learning of Pre-Trained Models via Task-agnostic Low-rank Residual Adaptation](https://arxiv.org/pdf/2505.12318)| 2025 | Arxiv
| [Parameter Efficient Continual Learning with Dynamic Low-Rank Adaptation](https://arxiv.org/pdf/2505.11998)| 2025 | Arxiv
| [Beyond CLIP Generalization: Against Forward&Backward Forgetting Adapter for Continual Learning of Vision-Language Models](https://arxiv.org/pdf/2505.07690)| 2025 | Arxiv
| [Componential Prompt-Knowledge Alignment for Domain Incremental Learning](https://arxiv.org/pdf/2505.04575)| 2025 | ICML
| [Enhancing Pre-Trained Model-Based Class-Incremental Learning through Neural Collapse](https://arxiv.org/pdf/2504.18437)| 2025 | Arxiv
| [SPECI: Skill Prompts based Hierarchical Continual Imitation Learning for Robot Manipulation](https://arxiv.org/pdf/2504.15561)| 2025 | Arxiv
| [Domain-Adaptive Continued Pre-Training of Small Language Models](https://arxiv.org/pdf/2504.09687)| 2025 | Arxiv
| [SEE: Continual Fine-tuning with Sequential Ensemble of Experts](https://arxiv.org/abs/2504.06664)| 2025 | Arxiv
| [Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning](https://arxiv.org/pdf/2504.07097)| 2025 | Arxiv
| [Dynamic Adapter Tuning for Long-Tailed Class-Incremental Learning](https://ieeexplore.ieee.org/abstract/document/10943390)| 2025 | WACV |
| [LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning](https://arxiv.org/pdf/2503.18985)| 2025 | CVPR |
| [HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model](https://arxiv.org/pdf/2503.12941)| 2025 | Arxiv |
| [Analytic Subspace Routing: How Recursive Least Squares Works in Continual Learning of Large Language Model](https://arxiv.org/pdf/2503.13575)| 2025 | Arxiv |
| [TMs-TSCIL Pre-Trained Models Based Class-Incremental Learning](https://arxiv.org/pdf/2503.07153)| 2025 | Arxiv |
| [Merge then Realign: Simple and Effective Modality-Incremental Continual Learning for Multimodal LLMs](https://arxiv.org/pdf/2503.07663)| 2025 | Arxiv |
| [CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering](https://arxiv.org/pdf/2503.00413)| 2025 | CVPR |
| [CLDyB: Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models](https://arxiv.org/pdf/2503.04655)| 2025 | ICLR |
| [Beyond Cosine Decay: On the effectiveness of Infinite Learning Rate Schedule for Continual Pre-training](https://arxiv.org/pdf/2503.02844)| 2025 | Arxiv |
| [Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](https://arxiv.org/pdf/2503.04229)| 2025 | Arxiv |
| [Recurrent Knowledge Identification and Fusion for Language Model Continual Learning](https://arxiv.org/pdf/2502.17510)| 2025 | Arxiv |
| [An Empirical Analysis of Forgetting in Pre-trained Models with Incremental Low-Rank Updates](https://proceedings.mlr.press/v274/soutif25a.html)| 2025 | Conference on Lifelong Learning Agents |
| [Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning](https://arxiv.org/pdf/2502.11019)| 2025 | ICLR |
| [Mitigating Visual Knowledge Forgetting in MLLM Instruction-tuning via Modality-decoupled Gradient Descent](https://arxiv.org/pdf/2502.11740)| 2025 | Arxiv |
| [How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](https://arxiv.org/pdf/2502.11196)| 2025 | Arxiv |
| [DATA: Decomposed Attention-based Task Adaptation for Rehearsal-Free Continual Learning](https://arxiv.org/pdf/2502.11482)| 2025 | Arxiv |
| [SPARC: Subspace-Aware Prompt Adaptation for Robust Continual Learning in LLMs](https://arxiv.org/pdf/2502.02909)| 2025 | Arxiv |
| [Sculpting [CLS] Features for Pre-Trained Model-Based Class-Incremental Learning](https://arxiv.org/pdf/2502.14762)| 2025 | Arxiv |
| [Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning](https://arxiv.org/pdf/2502.11019)|2025 | ICLR
| [S-LoRA: Scalable Low-Rank Adaptation for Class Incremental Learning](https://arxiv.org/pdf/2501.13198)|2025 | ICLR
| [Spurious Forgetting in Continual Learning of Language Models](https://arxiv.org/pdf/2501.13453)|2025 | ICLR
| [Practical Continual Forgetting for Pre-trained Vision Models](https://arxiv.org/pdf/2501.09705)|2025 | Arxiv
| [PEARL: Input-Agnostic Prompt Enhancement with Negative Feedback Regulation for Class-Incremental Learning](https://arxiv.org/pdf/2412.10900))|2025 | AAAI
| [MOS: Model Surgery for Pre-Trained Model-Based Class-Incremental Learning](https://arxiv.org/pdf/2412.09441)|2025 | AAAI
| [Learn or Recall? Revisiting Incremental Learning with Pre-trained Language Models](https://arxiv.org/pdf/2312.07887)|2024 | ACL
| [Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal](https://aclanthology.org/2024.acl-long.77.pdf)|2024 | ACL
| [Mixture of Experts Meets Prompt-Based Continual Learning](https://arxiv.org/pdf/2405.14124)|2024 | NeurIPS
| [SAFE: Slow and Fast Parameter-Efficient Tuning for Continual Learning with Pre-Trained Mode](https://arxiv.org/pdf/2411.02175)|2024 | NeurIPS
| [Incremental Learning of Retrievable Skills For Efficient Continual Task Adaptation](https://arxiv.org/pdf/2410.22658)|2024 | NeurIPS
| [Vector Quantization Prompting for Continual Learning](https://arxiv.org/pdf/2410.20444)|2024 | NeurIPS
| [Dual Low-Rank Adaptation for Continual Learning with Pre-Trained Models](https://arxiv.org/pdf/2411.00623)|2024 | Arxiv
| [Is Parameter Collision Hindering Continual Learning in LLMs](https://arxiv.org/pdf/2410.10179)|2024 | Arxiv
| [Does RoBERTa Perform Better than BERT in Continual Learning: An Attention Sink Perspective](https://arxiv.org/pdf/2410.05648)|2024 | COLM
| [Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning](https://arxiv.org/pdf/2410.00911)|2024 | Arxiv
| [ICL-TSVD: Bridging Theory and Practice in Continual Learning with Pre-trained Models](https://arxiv.org/pdf/2410.00645)|2024 | Arxiv
| [Adaptive Adapter Routing for Long-Tailed Class-Incremental Learning](https://arxiv.org/pdf/2311.03396) | 2024 | Machine Learning Journal
| [CoIN: A Benchmark of Continual Instruction tuNing for Multimodel Large Language Model](https://arxiv.org/pdf/2403.08350)|2024 | Arxiv
| [Continual Instruction Tuning for Large Multimodal Models](https://arxiv.org/pdf/2311.16206)|2024 | Arxiv
| [Parameter-Efficient Fine-Tuning for Continual Learning: A Neural Tangent Kernel Perspective](https://arxiv.org/pdf/2407.17120)|2024 | Arxiv
| [Class-Incremental Learning with CLIP: Adaptive Representation Adjustment and Parameter Fusion](https://arxiv.org/pdf/2407.14143) | 2024 | ECCV
| [Model Tailor: Mitigating Catastrophic Forgetting in Multi-modal Large Language Models](https://arxiv.org/pdf/2402.12048) | 2024 | ICML
| [One Size Fits All for Semantic Shifts: Adaptive Prompt Tuning for Continual Learning](https://arxiv.org/pdf/2311.12048) | 2024 | ICML
| [HiDe-PET: Continual Learning via Hierarchical Decomposition of Parameter-Efficient Tuning](https://arxiv.org/pdf/2407.05229)|2024 | Arxiv
| [Domain Adaptation of Llama3-70B-Instruct through Continual Pre-Training and Model Merging: A Comprehensive Evaluation](https://arxiv.org/pdf/2406.14971)|2024 | Arxiv
| [Mitigate Negative Transfer with Similarity Heuristic Lifelong Prompt Tuning](https://arxiv.org/pdf/2406.12251)|2024 | ACL
| [Reflecting on the State of Rehearsal-free Continual Learning with Pretrained Models](https://arxiv.org/pdf/2406.09384)|2024 | CoLLAs
| [Choice of PEFT Technique in Continual Learning: Prompt Tuning is Not All You Need](https://arxiv.org/pdf/2406.03216)|2024 | Arxiv
| [Disperse-Then-Merge: Pushing the Limits of Instruction Tuning via Alignment Tax Reduction](https://arxiv.org/pdf/2405.13432)|2024 | ACL
| [Gradient Projection For Parameter-Efficient Continual Learning](https://arxiv.org/pdf/2405.13383) |2024 | Arxiv
| [Continual Learning of Large Language Models: A Comprehensive Survey](https://arxiv.org/pdf/2404.16789) |2024 | Arxiv
| [Prompt Customization for Continual Learning](https://arxiv.org/abs/2404.18060) |2024 | MM
| [Dynamically Anchored Prompting for Task-Imbalanced Continual Learning](https://arxiv.org/pdf/2404.14721) |2024 |IJCAI
| [Continual Forgetting for Pre-trained Vision Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Continual_Forgetting_for_Pre-trained_Vision_Models_CVPR_2024_paper.pdf#:~:text=We%20propose%20a%20new%20problem%20termed%20continual%20forgetting%2C,model%20while%20preserving%20the%20performance%20of%20the%20rest)|2024 |CVPR
| [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](https://arxiv.org/pdf/2404.00228.pdf) |2024 |CVPR
| [Semantically-Shifted Incremental Adapter-Tuning is A Continual ViTransformer](https://arxiv.org/pdf/2403.19979.pdf) |2024 |CVPR
| [Evolving Parameterized Prompt Memory for Continual Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29231) |2024 |AAAI
| [Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning](https://arxiv.org/pdf/2403.12030.pdf)| 2024 | CVPR
| [Consistent Prompting for Rehearsal-Free Continual Learning](https://arxiv.org/pdf/2403.08568.pdf)| 2024 | CVPR
| [Interactive Continual Learning: Fast and Slow Thinking](https://arxiv.org/pdf/2403.02628.pdf)| 2024 | CVPR
| [HOP to the Next Tasks and Domains for Continual Learning in NLP](https://arxiv.org/pdf/2402.18449.pdf)| 2024 | AAAI
| [OVOR: OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning](https://arxiv.org/pdf/2402.04129.pdf)| 2024 | ICLR
| [Continual Learning for Large Language Models: A Survey](https://arxiv.org/pdf/2402.01364.pdf)| 2024 | Arxiv
| [Continual Learning with Pre-Trained Models: A Survey](https://arxiv.org/pdf/2401.16386.pdf) | 2024 | Arxiv
| [INCPrompt: Task-Aware incremental Prompting for Rehearsal-Free Class-incremental Learning](https://arxiv.org/pdf/2401.11667.pdf) | 2024 | ICASSP
| [P2DT: Mitigating Forgetting in task-incremental Learning with progressive prompt Decision Transformer](https://arxiv.org/pdf/2401.11666.pdf) | 2024 | ICASSP
| [Scalable Language Model with Generalized Continual Learning](https://openreview.net/pdf?id=mz8owj4DXu) | 2024 | ICLR
| [Prompt Gradient Projection for Continual Learning](https://openreview.net/pdf?id=EH2O3h7sBI) | 2024 | ICLR
| [TiC-CLIP: Continual Training of CLIP Models](https://openreview.net/pdf?id=TLADT8Wrhn)| 2024 | ICLR
| [Hierarchical Prompts for Rehearsal-free Continual Learning](https://arxiv.org/pdf/2401.11544.pdf)| 2024 | Arxiv
| [KOPPA: Improving Prompt-based Continual Learning with Key-Query Orthogonal Projection and Prototype-based One-Versus-All](https://arxiv.org/pdf/2311.15414.pdf) | 2023 | Arxiv
| [RanPAC: Random Projections and Pre-trained Models for Continual Learning](https://arxiv.org/pdf/2307.02251.pdf) | 2023 | NeurIPS
| [Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality](https://arxiv.org/pdf/2310.07234.pdf) | 2023 | NeurIPS
| [A Unified Continual Learning Framework with General Parameter-Efficient Tuning](https://arxiv.org/pdf/2303.10070) |2023 | ICCV
| [Generating Instance-level Prompts for Rehearsal-free Continual Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Jung_Generating_Instance-level_Prompts_for_Rehearsal-free_Continual_Learning_ICCV_2023_paper.pdf) | 2023| ICCV
| [Introducing Language Guidance in Prompt-based Continual Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Khan_Introducing_Language_Guidance_in_Prompt-based_Continual_Learning_ICCV_2023_paper.pdf) | 2023| ICCV
| [Generating Instance-level Prompts for Rehearsal-free Continual Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Jung_Generating_Instance-level_Prompts_for_Rehearsal-free_Continual_Learning_ICCV_2023_paper.pdf) | 2023| ICCV
| [Space-time Prompting for Video Class-incremental Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Pei_Space-time_Prompting_for_Video_Class-incremental_Learning_ICCV_2023_paper.pdf) | 2023| ICCV
| [When Prompt-based Incremental Learning Does Not Meet Strong Pretraining](https://openaccess.thecvf.com/content/ICCV2023/papers/Tang_When_Prompt-based_Incremental_Learning_Does_Not_Meet_Strong_Pretraining_ICCV_2023_paper.pdf) | 2023| ICCV
| [Online Class Incremental Learning on Stochastic Blurry Task Boundary via Mask and Visual Prompt Tuning](https://openaccess.thecvf.com/content/ICCV2023/papers/Moon_Online_Class_Incremental_Learning_on_Stochastic_Blurry_Task_Boundary_via_ICCV_2023_paper.pdf) | 2023| ICCV
| [SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model](https://arxiv.org/pdf/2303.05118.pdf)| 2023| ICCV
| [Progressive Prompts: Continual Learning for Language Models](https://openreview.net/pdf?id=UJTgQBc91_) | 2023| ICLR
| [Continual Pre-training of Language Models](https://openreview.net/pdf?id=m_GDIItaI3o) | 2023 | ICLR
| [Continual Learning of Language Models](https://openreview.net/pdf?id=m_GDIItaI3o)| 2023 | ICLR
| [CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) | 2023 | CVPR
| [PIVOT: Prompting for Video Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Villa_PIVOT_Prompting_for_Video_Continual_Learning_CVPR_2023_paper.pdf) |2023 | CVPR
| [Do Pre-trained Models Benefit Equally in Continual Learning?](https://openaccess.thecvf.com/content/WACV2023/papers/Lee_Do_Pre-Trained_Models_Benefit_Equally_in_Continual_Learning_WACV_2023_paper.pdf) | 2023 | WACV
| [Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](https://arxiv.org/pdf/2303.07338.pdf) | 2023 | Arxiv
| [First Session Adaptation: A Strong Replay-Free Baseline for Class-Incremental Learning](https://arxiv.org/pdf/2303.13199.pdf) | 2023 | Arxiv
| [Memory Efficient Continual Learning with Transformers](https://openreview.net/pdf?id=U07d1Y-x2E) | 2022 | NeurIPS
| [S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning](https://openreview.net/pdf?id=ZVe_WeMold) |2022 | NeurIPS
| [Pretrained Language Model in Continual Learning: A Comparative Study](https://openreview.net/pdf?id=figzpGMrdD) | 2022 | ICLR
| [Effect of scale on catastrophic forgetting in neural networks](https://openreview.net/pdf?id=GhVS8_yPeEa) | 2022| ICLR
| [LFPT5: A Unified Framework for Lifelong Few-shot Language Learning Based on Prompt Tuning of T5](https://openreview.net/pdf?id=HCRVf71PMF) | 2022| ICLR
| [Learning to Prompt for Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.pdf) | 2022 | CVPR
| [Class-Incremental Learning with Strong Pre-trained Models](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Class-Incremental_Learning_With_Strong_Pre-Trained_Models_CVPR_2022_paper.pdf) |  2022|CVPR
| [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/pdf/2204.04799.pdf) |2022  |ECCV
| [ELLE: Efficient Lifelong Pre-training for Emerging Data](https://aclanthology.org/2022.findings-acl.220.pdf) | 2022 | ACL
| [Fine-tuned Language Models are Continual Learners](https://aclanthology.org/2022.emnlp-main.410.pdf) | 2022 | EMNLP
| [Continual Training of Language Models for Few-Shot Learning](https://aclanthology.org/2022.emnlp-main.695.pdf) | 2022 | EMNLP
| [Continual Learning with Foundation Models: An Empirical Study of Latent Replay](https://arxiv.org/pdf/2205.00329.pdf) |2022 | Conference on Lifelong Learning Agents
| [Rational LAMOL: A Rationale-Based Lifelong Learning Framework](https://aclanthology.org/2021.acl-long.229.pdf) | 2021 | ACL
| [Achieving Forgetting Prevention and Knowledge Transfer in Continual Learning](https://openreview.net/pdf?id=RJ7XFI15Q8f) | 2021 |NeurIPS
| [An Empirical Investigation of the Role of Pre-training in Lifelong Learning](https://arxiv.org/pdf/2112.09153.pdf) | 2021 | Arxiv
| [LAnguage MOdeling for Lifelong Language Learning](https://openreview.net/pdf?id=Skgxcn4YDS) | 2020 | ICLR


### Forgetting in Domain Adaptation
<a href="#top">[Back to top]</a>

> The goal of domain adaptation is to transfer the knowledge from a source domain to a target domain.


| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Towards Cross-Domain Continual Learning](https://arxiv.org/pdf/2402.12490)| 2024| ICDE
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
<a href="#top">[Back to top]</a>
<!-- <u>[Click back to content outline](#framework)</u> -->

> Test time adaptation (TTA) refers to the process of adapting a pre-trained model on-the-fly to unlabeled test data during inference or testing.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Ranked Entropy Minimization for Continual Test-Time Adaptation](https://arxiv.org/pdf/2505.16441)| 2025 | ICML
| [FoCTTA: Low-Memory Continual Test-Time Adaptation with Focus](https://arxiv.org/pdf/2502.20677)| 2025 | Arxiv
| [Conformal Uncertainty Indicator for Continual Test-Time Adaptation](https://arxiv.org/pdf/2502.02998)| 2025 | Arxiv
| [PCoTTA: Continual Test-Time Adaptation for Multi-Task Point Cloud Understanding](https://arxiv.org/pdf/2411.00632)| 2024 | NeurIPS
| [Adaptive Cascading Network for Continual Test-Time Adaptation](https://arxiv.org/pdf/2407.12240)| 2024 | CIKM
| [Controllable Continual Test-Time Adaptation](https://arxiv.org/pdf/2405.14602)| 2024 | Arxiv
| [ViDA: Homeostatic Visual Domain Adapter for Continual Test Time Adaptation](https://openreview.net/pdf?id=sJ88Wg5Bp5)| 2024 | ICLR
| [Continual Momentum Filtering on Parameter Space for Online Test-time Adaptation](https://openreview.net/pdf?id=BllUWdpIOA)| 2024 | ICLR
| [A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts](https://arxiv.org/pdf/2303.15361.pdf) | 2023 | Arxiv
| [MECTA: Memory-Economic Continual Test-Time Model Adaptation](https://openreview.net/pdf?id=N92hjSf5NNh) | 2023|ICLR
| [Decorate the Newcomers: Visual Domain Prompt for Continual Test Time Adaptation](https://arxiv.org/pdf/2212.04145.pdf) |2023 | AAAI (Outstanding Student Paper Award)
| [Robust Mean Teacher for Continual and Gradual Test-Time Adaptation](https://openaccess.thecvf.com/content/CVPR2023/papers/Dobler_Robust_Mean_Teacher_for_Continual_and_Gradual_Test-Time_Adaptation_CVPR_2023_paper.pdf) | 2023|CVPR
| [A Probabilistic Framework for Lifelong Test-Time Adaptation](https://openaccess.thecvf.com/content/CVPR2023/papers/Brahma_A_Probabilistic_Framework_for_Lifelong_Test-Time_Adaptation_CVPR_2023_paper.pdf) | 2023 | CVPR
| [EcoTTA: Memory-Efficient Continual Test-time Adaptation via Self-distilled Regularization](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_EcoTTA_Memory-Efficient_Continual_Test-Time_Adaptation_via_Self-Distilled_Regularization_CVPR_2023_paper.pdf) |  2023 | CVPR
| [AUTO: Adaptive Outlier Optimization for Online Test-Time OOD Detection](https://arxiv.org/pdf/2303.12267.pdf) |2023 |Arxiv
| [Efficient Test-Time Model Adaptation without Forgetting](https://proceedings.mlr.press/v162/niu22a/niu22a.pdf) | 2022| ICML
| [MEMO: Test time robustness via adaptation and augmentation](https://openreview.net/pdf?id=vn74m_tWu8O) | 2022|NeurIPS
| [Continual Test-Time Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.pdf) | 2022|CVPR
| [Improving test-time adaptation via shift-agnostic weight regularization and nearest source prototypes](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930433.pdf) |2022 | ECCV
| [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://openreview.net/pdf?id=uXl3bZLkr3c) | 2021 | ICLR


----------

### Forgetting in Meta-Learning
<a href="#top">[Back to top]</a>

> Meta-learning, also known as learning to learn, focuses on developing algorithms and models that can learn from previous learning experiences to improve their ability to learn new tasks or adapt to new domains more efficiently and effectively.

**Links**:
[Incremental Few-Shot Learning](#incremental-few-shot-learning) |
[Continual Meta-Learning](#continual-meta-learning)


#### Incremental Few-Shot Learning
<a href="#top">[Back to top]</a>
> Incremental few-shot learning (IFSL) focuses on the challenge of learning new categories with limited labeled data while retaining knowledge about previously learned categories.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [An experimental approach on Few Shot Class Incremental Learning](https://arxiv.org/pdf/2503.11349)| 2025 | Arxiv
| [Controllable Forgetting Mechanism for Few-Shot Class-Incremental Learning](https://arxiv.org/pdf/2501.15998)| 2025 | ICASSP
| [AnchorInv: Few-Shot Class-Incremental Learning of Physiological Signals via Feature Space-Guided Inversion](https://arxiv.org/pdf/2412.13714)| 2024 | Arxiv
| [On Distilling the Displacement Knowledge for Few-Shot Class-Incremental Learning](https://arxiv.org/pdf/2412.11017) | 2024 | Arxiv
| [Few-Shot Class-Incremental Learning via Training-Free Prototype Calibration](https://arxiv.org/pdf/2312.05229.pdf) | 2023|NeurIPS
| [Constrained Few-shot Class-incremental Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Hersche_Constrained_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.pdf) | 2022 | CVPR
| [Meta-Learning with Less Forgetting on Large-Scale Non-Stationary Task Distributions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800211.pdf) | 2022 | ECCV
| [Overcoming Catastrophic Forgetting in Incremental Few-Shot Learning by Finding Flat Minima](https://proceedings.neurips.cc/paper/2021/file/357cfba15668cc2e1e73111e09d54383-Paper.pdf) | 2021 | NeurIPS
| [Incremental Few-shot Learning via Vector Quantization in Deep Embedded Space](https://openreview.net/pdf?id=3SV-ZePhnZM) |2021 | ICLR
| [XtarNet: Learning to Extract Task-Adaptive Representation for Incremental Few-Shot Learning](http://proceedings.mlr.press/v119/yoon20b/yoon20b.pdf) |2020 | ICML
| [Incremental Few-Shot Learning with Attention Attractor Networks](https://proceedings.neurips.cc/paper_files/paper/2019/file/e833e042f509c996b1b25324d56659fb-Paper.pdf) |2019 | NeurIPS
| [Dynamic Few-Shot Visual Learning without Forgetting](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gidaris_Dynamic_Few-Shot_Visual_CVPR_2018_paper.pdf) | 2018| CVPR



#### Continual Meta-Learning
<a href="#top">[Back to top]</a>
> The goal of continual meta-learning (CML) is to address the challenge of forgetting in non-stationary task distributions.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Meta Continual Learning Revisited: Implicitly Enhancing Online Hessian Approximation via Variance Reduction](https://openreview.net/forum?id=TpD2aG1h0D) | 2024 | ICLR
| [Recasting Continual Learning as Sequence Modeling](https://arxiv.org/pdf/2310.11952.pdf) | 2023 | NeurIPS
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
<a href="#top">[Back to top]</a>
> The goal of a generative model is to learn a generator that can generate samples from a target distribution.

**Links**:
 [GAN Training is a Continual Learning Problem](#gan-training-is-a-continual-learning-problem) |
 [Lifelong Learning of Generative Models](#lifelong-learning-of-generative-models)


#### GAN Training is a Continual Learning Problem
<a href="#top">[Back to top]</a>
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
<a href="#top">[Back to top]</a>
> The goal is to develop generative models that can continually generate high-quality samples for both new and previously encountered tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [KFC: Knowledge Reconstruction and Feedback Consolidation Enable Efficient and Effective Continual Generative Learning](https://openreview.net/pdf?id=pVTcR8ig3R) | 2024 | ICLR
| [The Curse of Recursion: Training on Generated Data Makes Models Forget](https://arxiv.org/pdf/2305.17493.pdf)| 2023|Arxiv
| [Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models](https://arxiv.org/pdf/2303.17591.pdf) | 2023|Arxiv
| [Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models](https://arxiv.org/pdf/2305.10120.pdf) | 2023|Arxiv
| [Lifelong Generative Modelling Using Dynamic Expansion Graph Model](https://ojs.aaai.org/index.php/AAAI/article/view/20867/20626) | 2022|AAAI
| [Continual Variational Autoencoder Learning via Online Cooperative Memorization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830515.pdf) |2022 |ECCV
| [Hyper-LifelongGAN: Scalable Lifelong Learning for Image Conditioned Generation](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhai_Hyper-LifelongGAN_Scalable_Lifelong_Learning_for_Image_Conditioned_Generation_CVPR_2021_paper.pdf) | 2021|CVPR
| [Lifelong Twin Generative Adversarial Networks](https://ieeexplore.ieee.org/document/9506116) |2021 | ICIP
| [Lifelong Mixture of Variational Autoencoders](https://arxiv.org/pdf/2107.04694.pdf) |2021 | TNNLS
| [Lifelong Generative Modeling](https://arxiv.org/pdf/1705.09847.pdf) | 2020 |Neurocomputing
| [GAN Memory with No Forgetting](https://arxiv.org/pdf/2006.07543)| 2020 | NeurIPS
| [Lifelong GAN: Continual Learning for Conditional Image Generation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhai_Lifelong_GAN_Continual_Learning_for_Conditional_Image_Generation_ICCV_2019_paper.pdf) |2019 | ICCV




----------

### Forgetting in Reinforcement Learning
<a href="#top">[Back to top]</a>

> Reinforcement learning is a machine learning technique that allows an agent to learn how to behave in an environment by trial and error, through rewards and punishments.


| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Rethinking the Foundations for Continual Reinforcement Learning](https://arxiv.org/pdf/2504.08161)| 2025 | RLDM
| [Mastering Continual Reinforcement Learning through Fine-Grained Sparse Network Allocation and Dormant Neuron Exploration Notebook](https://arxiv.org/pdf/2503.05246)| 2025 | Arxiv
| [An Autonomous Network Orchestration Framework Integrating Large Language Models with Continual Reinforcement Learning](https://arxiv.org/pdf/2502.16198)| 2025 | Arxiv
| [Self-Composing Policies for Scalable Continual Reinforcement Learning](https://arxiv.org/pdf/2506.14811)| 2024 | ICML
| [Continual Deep Reinforcement Learning with Task-Agnostic Policy Distillation](https://arxiv.org/pdf/2411.16532)| 2024 | Arxiv
| [Data Augmentation for Continual RL via Adversarial Gradient Episodic Memory](https://arxiv.org/pdf/2408.13452)| 2024 | Arxiv
| [Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem](https://arxiv.org/pdf/2402.02868.pdf)| 2024 | Arxiv
| [Hierarchical Continual Reinforcement Learning via Large Language Model](https://arxiv.org/pdf/2401.15098.pdf)| 2024 | Arxiv
| [Augmenting Replay in World Models for Continual Reinforcement Learning](https://arxiv.org/pdf/2401.16650.pdf) | 2024 | Arxiv
| [CPPO: Continual Learning for Reinforcement Learning with Human Feedback](https://openreview.net/pdf?id=86zAUE80pP) | 2024 | ICLR
| [Prediction and Control in Continual Reinforcement Learning](https://arxiv.org/pdf/2312.11669.pdf) | 2023 | NeurIPS
| [Replay-enhanced Continual Reinforcement Learning](https://arxiv.org/pdf/2311.11557.pdf)| 2023 | TMLR
| [A Definition of Continual Reinforcement Learning](https://arxiv.org/pdf/2307.11046.pdf) | 2023 | Arxiv
| [Continual Task Allocation in Meta-Policy Network via Sparse Prompting](https://openreview.net/pdf?id=IqI8074rFu) |2023 | ICML
| [Building a Subspace of Policies for Scalable Continual Learning](https://openreview.net/pdf?id=ZloanUtG4a) |2023 | ICLR
| [Continual Model-based Reinforcement Learning for Data Efficient Wireless Network Optimisation](https://arxiv.org/pdf/2404.19462) | 2023  | ECML
| [Modular Lifelong Reinforcement Learning via Neural Composition](https://openreview.net/pdf?id=5XmLzdslFNN) |2022 |ICLR
| [Disentangling Transfer in Continual Reinforcement Learning](https://openreview.net/pdf?id=pgF-N1YORd)|2022 |NeurIPS
| [Towards continual reinforcement learning: A review and perspectives](https://arxiv.org/pdf/2012.13490.pdf) | 2022 | Journal of Artificial Intelligence Research
| [Reinforced continual learning for graphs](https://arxiv.org/pdf/2209.01556)| 2022 | CIKM
| [Model-Free Generative Replay for Lifelong Reinforcement Learning: Application to Starcraft-2](https://proceedings.mlr.press/v199/daniels22a/daniels22a.pdf) | 2022|Conference on Lifelong Learning Agents
| [Transient Non-stationarity and Generalisation in Deep Reinforcement Learning](https://openreview.net/pdf?id=Qun8fv4qSby) | 2021 | ICLR
| [Sharing Less is More: Lifelong Learning in Deep Networks with Selective Layer Transfer](http://proceedings.mlr.press/v139/lee21a/lee21a.pdf) | 2021| ICML
| [Pseudo-rehearsal: Achieving deep reinforcement learning without catastrophic forgetting](https://arxiv.org/pdf/1812.02464.pdf) | 2021|Neurocomputing
| [Lifelong Policy Gradient Learning of Factored Policies for Faster Training Without Forgetting](https://arxiv.org/pdf/2007.07011.pdf)|2020 |NeurIPS
| [Policy Consolidation for Continual Reinforcement Learning](http://proceedings.mlr.press/v97/kaplanis19a/kaplanis19a.pdf)| 2019| ICML
| [Exploiting Hierarchy for Learning and Transfer in KL-regularized RL](https://openreview.net/pdf?id=CCs4iXw4KJ-) | 2019|Arxiv
| [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf) | 2017| ICML
| [Progressive neural networks](https://arxiv.org/pdf/1606.04671.pdf) |2016 | Arxiv
| [Learning a synaptic learning rule](https://ieeexplore.ieee.org/document/155621) |1991 | IJCNN

----------

### Forgetting in Federated Learning
<a href="#top">[Back to top]</a>

> Federated learning (FL) is a decentralized machine learning approach where the training process takes place on local devices or edge servers instead of a centralized server.

**Links**:
 [Forgetting Due to Non-IID Data in FL  ](#forgetting-due-to-non-iid-data-in-fl) |
 [Federated Continual Learning](#federated-continual-learning)

#### Forgetting Due to Non-IID Data in FL  
<a href="#top">[Back to top]</a>
> This branch pertains to the forgetting problem caused by the inherent non-IID (not identically and independently distributed) data among different clients participating in FL.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [Exemplar-condensed Federated Class-incremental Learning](https://arxiv.org/pdf/2412.18926)| 2024 | Arxiv
| [Flashback: Understanding and Mitigating Forgetting in Federated Learning](https://arxiv.org/pdf/2402.05558.pdf) | 2024 | Arxiv
| [How to Forget Clients in Federated Online Learning to Rank?](https://arxiv.org/pdf/2401.13410.pdf) | 2024 | ECIR
| [GradMA: A Gradient-Memory-based Accelerated Federated Learning with Alleviated Catastrophic Forgetting](https://openaccess.thecvf.com/content/CVPR2023/papers/Luo_GradMA_A_Gradient-Memory-Based_Accelerated_Federated_Learning_With_Alleviated_Catastrophic_Forgetting_CVPR_2023_paper.pdf) |2023 | CVPR
| [Acceleration of Federated Learning with Alleviated Forgetting in Local Training](https://openreview.net/pdf?id=541PxiEKN3F) |2022 |ICLR
| [Preservation of the Global Knowledge by Not-True Distillation in Federated Learning](https://openreview.net/pdf?id=qw3MZb1Juo) | 2022 |NeurIPS
| [Learn from Others and Be Yourself in Heterogeneous Federated Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf) |2022 |CVPR
| [Rethinking Architecture Design for Tackling Data Heterogeneity in Federated Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Qu_Rethinking_Architecture_Design_for_Tackling_Data_Heterogeneity_in_Federated_Learning_CVPR_2022_paper.pdf) | 2022|CVPR
| [Model-Contrastive Federated Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.pdf) | 2021|CVPR
| [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf) | 2020| ICML
| [Overcoming Forgetting in Federated Learning on Non-IID Data](http://www.edgify.ai/wp-content/uploads/2020/04/Overcoming-Forgetting-in-Federated-Learning-on-Non-IID-Data.pdf) | 2019|NeurIPSW


#### Federated Continual Learning
<a href="#top">[Back to top]</a>
> This branch addresses the issue of continual learning within each individual client in the federated learning process, which results in forgetting at the overall FL level.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [SacFL: Self-Adaptive Federated Continual Learning for Resource-Constrained End Devices](https://arxiv.org/pdf/2505.00365)|2025|TNNLS
| [Dynamic Allocation Hypernetwork with Adaptive Model Recalibration for Federated Continual Learning](https://arxiv.org/pdf/2503.20808)|2025|IPMI
| [Federated Class-Incremental Learning: A Hybrid Approach Using Latent Exemplars and Data-Free Techniques to Address Local and Global Forgetting](https://arxiv.org/pdf/2501.15356)|2025|ICLR
| [Resource-Constrained Federated Continual Learning: What Does Matter?](https://arxiv.org/pdf/2501.08737)| 2025 | Arxiv
| [Accurate Forgetting for Heterogeneous Federated Continual Learning](https://arxiv.org/pdf/2502.14205)| 2024 | ICLR
| [Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning](https://arxiv.org/pdf/2409.01128)| 2024 | ECCV
| [PIP: Prototypes-Injected Prompt for Federated Class Incremental](https://arxiv.org/pdf/2407.20705)| 2024 | CIKM
| [Personalized Federated Continual Learning via Multi-granularity Prompt](https://arxiv.org/pdf/2407.00113)| 2024 | KDD
| [Federated Continual Learning via Prompt-based Dual Knowledge Transfer](https://openreview.net/pdf?id=Kqa5JakTjB)| 2024 | ICML
| [Text-Enhanced Data-free Approach for Federated Class-Incremental Learning](https://arxiv.org/pdf/2403.14101.pdf)| 2024 | CVPR
| [Federated Continual Learning via Knowledge Fusion: A Survey](https://ieeexplore.ieee.org/abstract/document/10423871) | 2024 | TKDE
| [Accurate Forgetting for Heterogeneous Federated Continual Learning](https://openreview.net/pdf?id=ShQrnAsbPI) | 2024 | ICLR
| [Federated Orthogonal Training: Mitigating Global Catastrophic Forgetting in Continual Federated Learning](https://openreview.net/pdf?id=nAs4LdaP9Y) | 2024 | ICLR
| [A Data-Free Approach to Mitigate Catastrophic Forgetting in Federated Class Incremental Learning for Vision Tasks](https://arxiv.org/pdf/2311.07784.pdf)|2023 | NeurIPS
| [Federated Continual Learning via Knowledge Fusion: A Survey](https://arxiv.org/pdf/2312.16475.pdf) | 2023 | Arxiv
| [A Data-Free Approach to Mitigate Catastrophic Forgetting in Federated Class Incremental Learning for Vision Tasks](https://arxiv.org/pdf/2311.07784.pdf)| 2023 | NeurIPS
| [TARGET: Federated Class-Continual Learning via Exemplar-Free Distillation](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_TARGET_Federated_Class-Continual_Learning_via_Exemplar-Free_Distillation_ICCV_2023_paper.pdf)| 2023 | ICCV
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
<a href="#top">[Back to top]</a>
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
<a href="#top">[Back to top]</a>

####  Combat Overfitting Through Forgetting
<a href="#top">[Back to top]</a>
> Overfitting in neural networks occurs when the model excessively memorizes the training data, leading to poor generalization. To address overfitting, it is necessary to selectively forget irrelevant or noisy information.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| ["Forgetting" in Machine Learning and Beyond: A Survey](https://arxiv.org/pdf/2405.20620v1)| 2024  | Arxiv
| [The Effectiveness of Random Forgetting for Robust Generalization](https://arxiv.org/pdf/2402.11733.pdf)| 2024|ICLR
| [Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier](https://openreview.net/pdf?id=OpC-9aBBVJe) | 2023|ICLR
| [The Primacy Bias in Deep Reinforcement Learning](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf) | 2022|ICML
| [The Impact of Reinitialization on Generalization in Convolutional Neural Networks](https://arxiv.org/pdf/2109.00267.pdf) | 2021 | Arxiv
| [Learning with Selective Forgetting](https://www.ijcai.org/proceedings/2021/0137.pdf) | 2021|IJCAI
| [SIGUA: Forgetting May Make Learning with Noisy Labels More Robust](https://arxiv.org/pdf/1809.11008.pdf) | 2020|ICML
| [Invariant Representations through Adversarial Forgetting](https://ojs.aaai.org/index.php/AAAI/article/view/5850/5706) |2020 |AAAI
| [Forget a Bit to Learn Better: Soft Forgetting for CTC-based Automatic Speech Recognition](https://www.isca-speech.org/archive_v0/Interspeech_2019/pdfs/2841.pdf) | 2019 |Interspeech



####  Learning New Knowledge Through Forgetting Previous Knowledge
<a href="#top">[Back to top]</a>
> "Learning to forget" suggests that not all previously acquired prior knowledge is helpful for learning new tasks.

| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| ["Forgetting" in Machine Learning and Beyond: A Survey](https://arxiv.org/pdf/2405.20620v1)| 2024  | Arxiv
| [Improving Language Plasticity via Pretraining with Active Forgetting](https://arxiv.org/pdf/2307.01163.pdf) | 2023 | NeurIPS
| [ReFactor GNNs: Revisiting Factorisation-based Models from a Message-Passing Perspective](https://openreview.net/pdf?id=81LQV4k7a7X) | 2022|NeurIPS
| [Fortuitous Forgetting in Connectionist Networks](https://openreview.net/pdf?id=ei3SY1_zYsE) | 2022|ICLR
| [Skin Deep Unlearning: Artefact and Instrument Debiasing in the Context of Melanoma Classification](https://proceedings.mlr.press/v162/bevan22a/bevan22a.pdf) |2022 |ICML
| [Near-Optimal Task Selection for Meta-Learning with Mutual Information and Online Variational Bayesian Unlearning](https://proceedings.mlr.press/v151/chen22h/chen22h.pdf) |2022 |AISTATS
| [AFEC: Active Forgetting of Negative Transfer in Continual Learning](https://proceedings.neurips.cc/paper/2021/file/bc6dc48b743dc5d013b1abaebd2faed2-Paper.pdf) |2021 |NeurIPS
| [Knowledge Evolution in Neural Networks](https://openaccess.thecvf.com/content/CVPR2021/papers/Taha_Knowledge_Evolution_in_Neural_Networks_CVPR_2021_paper.pdf) | 2021 | CVPR
| [Active Forgetting: Adaptation of Memory by Prefrontal Control](https://www.annualreviews.org/doi/10.1146/annurev-psych-072720-094140) | 2021|Annual Review of Psychology
| [Learning to Forget for Meta-Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Baik_Learning_to_Forget_for_Meta-Learning_CVPR_2020_paper.pdf) | 2020|CVPR
| [The Forgotten Part of Memory](https://www.nature.com/articles/d41586-019-02211-5) |2019 |Nature
| [Learning Not to Learn: Training Deep Neural Networks with Biased Data](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Learning_Not_to_Learn_Training_Deep_Neural_Networks_With_Biased_CVPR_2019_paper.pdf) | 2019| CVPR
| [Inhibiting your native language: the role of retrieval-induced forgetting during second-language acquisition](https://pubmed.ncbi.nlm.nih.gov/17362374/) | 2007|Psychological Science


----------

### Machine Unlearning
<a href="#top">[Back to top]</a>

> Machine unlearning, a recent area of research, addresses the need to forget previously learned training data in order to protect user data privacy.


| **Paper Title** | **Year** | **Conference/Journal** |
| --------------- | :----: | :----: |
| [A Unified Gradient-based Framework for Task-agnostic Continual Learning-Unlearning](https://arxiv.org/pdf/2505.16441)| 2025 | Arxiv
| [Exploring Incremental Unlearning: Techniques, Challenges, and Future Directions](https://arxiv.org/pdf/2502.16708) | 2025 | Arxiv
| [Unlearning during Learning: An Efficient Federated Machine Unlearning Method](https://arxiv.org/pdf/2405.15474) | 2024 | IJCAI
| [Label-Agnostic Forgetting: A Supervision-Free Unlearning in Deep Models](https://arxiv.org/pdf/2404.00506.pdf) |2024 | ICLR
| [Machine Unlearning: A Survey](https://dl.acm.org/doi/10.1145/3603620) | 2023 | ACM Computing Surveys
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

**Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=EnnengYang/Awesome-Forgetting-in-Deep-Learning&type=Date)](https://star-history.com/#EnnengYang/Awesome-Forgetting-in-Deep-Learning&Date)


******

**Contact**

We welcome all researchers to contribute to this repository **'forgetting in deep learning'**.

Email: wangzhenyineu@gmail.com | ennengyang@stumail.neu.edu.cn
