# FedEdgeHealth: A Byzantine-Resilient Hierarchical Federated Learning Framework for Privacy-Preserving Medical Diagnosis

This repository contains the official implementation of FedEdgeHealth, a hierarchical federated learning framework designed to integrate multi-metric Byzantine-resilient aggregation, dual-layer privacy protection, and communication-efficient edge computing for medical diagnosis.

**Note on Code Availability:** A portion of the basic source code for FedEdgeHealth has been uploaded to this repository to provide a foundational overview of the framework architecture. The complete source code, including all training scripts, advanced defense modules, and detailed preprocessing pipelines, will be made publicly available upon the official acceptance of our manuscript. For the duration of the peer-review process, the full codebase has been provided to the journal editors and reviewers as part of the submission package to ensure transparency and reproducibility.

## Dataset Information

FedEdgeHealth was evaluated on three publicly available medical datasets. We do not host the raw data in this repository due to size and licensing constraints, but the official sources are listed below. Preprocessing and stratified splitting scripts will be included in the final release.

* **COVID-19 Radiography Database:** [Official Kaggle Repository](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
* **COVID-Qu-Ex:** [Official Mendeley Database](https://data.mendeley.com/datasets/9xkhgts2s6/3)
* **MIMIC-III:** [Official Kaggle Repository](https://www.kaggle.com/datasets/asjad99/mimiciii?resource=download)

## Framework Overview

The FedEdgeHealth framework addresses the interdependent challenges of data privacy, heterogeneity, and adversarial robustness in federated medical image and electronic health record analysis. Key components include:

* **Hierarchical Client-Edge-Cloud Architecture:** A three-tier topology that reduces central server communication overhead by 67% to 80% compared to traditional flat federated learning architectures.
* **Multi-Metric Byzantine Defense:** An aggregation mechanism leveraging cosine similarity, Euclidean distance, and update magnitude to filter malicious updates under clinical non-independent and identically distributed (non-IID) data shifts.
* **Dual-Layer Privacy Mechanism:** Integration of client-level differential privacy (epsilon = 0.93) with lightweight transmission obfuscation (LWC) to ensure formal privacy guarantees and transit-level confidentiality without the computational burden of full homomorphic encryption.

## Reproduction and Setup

Upon public release, this repository will include:

* A `requirements.txt` file for dependency management.
* Step-by-step scripts for data preprocessing, non-IID Dirichlet client partitioning, and hierarchical model training.
* Configuration files matching the hyperparameters reported in the manuscript.

## Citation

If you find this work or the provided code useful for your research, please cite our paper:

```bibtex
@article{fededgehealth2026,
  title={FedEdgeHealth: A Byzantine-Resilient Hierarchical Federated Learning Framework for Privacy-Preserving Medical Diagnosis},
  author={Author Names},
  journal={Engineering Applications of Artificial Intelligence},
  year={2026},
  note={Submitted for publication},
  publisher={Elsevier}
}
