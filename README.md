**Team Name:** 314IV

**Members:** * Ismoil Salohiddinov (ID: 220626)
* Komiljon Qosimov (ID: 220493)
* Abdurashid Djumabaev (ID: 210004)

**Topic:** #6 Federated Continual Learning for MRI Segmentation
**GitHub Repository:** [https://github.com/smailxpy/Federated-MRI-Segmentation](https://github.com/smailxpy/Federated-MRI-Segmentation)

---

# CV25_Proposal_314IV

## Federated Continual Learning for MRI Segmentation

### 1. Title & Team

**Project Title:** Federated Continual Learning for MRI Segmentation
**Team Name:** 314IV
**Members:** Ismoil Salohiddinov (Coordinator), Komiljon Qosimov, Abdurashid Djumabaev
**Emails:** [220626@centralasian.uz](mailto:220626@centralasian.uz), [220493@centralasian.uz](mailto:220493@centralasian.uz), [210004@centralasian.uz](mailto:210004@centralasian.uz)
**GitHub:** [smailxpy/Federated-MRI-Segmentation](https://github.com/smailxpy/Federated-MRI-Segmentation)

### 2. Abstract (150–200 words)

Federated Learning (FL) enables collaborative model training without centralized data sharing, which is critical for privacy-sensitive fields like medical imaging. However, traditional FL methods struggle with continual learning, where new data distributions (e.g., from new hospitals or modalities) cause catastrophic forgetting. This project proposes a Federated Continual Learning (FCL) framework for MRI brain tumor segmentation using drift-aware adapters within a U-Net model. We simulate a federated environment with data splits representing multiple hospitals using the BraTS2021 dataset. Each client adapts locally while the central server aggregates shared representations. We evaluate segmentation quality using Dice and Hausdorff Distance (HD95) and assess model forgetting across continual updates. The proposed system aims to achieve high segmentation accuracy and robustness under domain drift, contributing to privacy-preserving and adaptive medical AI systems.

### 3. Problem & Motivation

MRI segmentation plays a vital role in medical diagnostics, enabling accurate tumor boundary identification. Centralized model training, however, raises privacy and ethical concerns due to patient data sharing restrictions. Federated Learning addresses this challenge but remains vulnerable to *catastrophic forgetting* when new data distributions appear over time. Hospitals often introduce new MRI scanners, imaging sequences, or patient populations, causing performance degradation. This proposal tackles the dual challenge of **privacy-preserving learning** and **continual adaptability**. Our goal is to build a system that continuously improves without retraining from scratch or requiring access to past data, ensuring reliability and trustworthiness in real-world deployments.

### 4. Related Work

| Work                                  | Approach                          | Limitation                          | Contribution                       |
| ------------------------------------- | --------------------------------- | ----------------------------------- | ---------------------------------- |
| Li et al. (2021) FedAvg               | Simple parameter averaging        | Forgetting under drift              | We add adapters to mitigate drift  |
| Chen et al. (2022) FedProx            | Regularized FL training           | Partial improvement on domain shift | Use local stability control        |
| Zhao et al. (2023) FCL-U-Net          | Continual segmentation            | High compute demand                 | Optimize with lightweight adapters |
| Kamnitsas et al. (2017) U-Net (BraTS) | Centralized segmentation          | Privacy risk                        | Federated simulation of hospitals  |
| Hsu et al. (2019) SplitFed            | Model split between client/server | Slow training                       | Hybrid adaptive aggregation        |

**Datasets:** BraTS2021, BraTS2018 (public, CC BY 4.0).
**Frameworks:** PyTorch, Flower (FL framework), MONAI (medical imaging tools).

### 5. Data & Resources

* **Datasets:** BraTS2021 (~1,250 MRI volumes, labeled tumor regions, 240×240×155 resolution).
* **Splits:** 4 virtual hospitals (25%, 25%, 25%, 25%).
* **Hardware:** Local GPU (NVIDIA RTX 3060) + Colab Pro for distributed simulation.
* **Ethics:** Only open-source, de-identified datasets are used.

### 6. Method

**Baseline:** Federated U-Net trained using FedAvg across 4 clients.
**Proposed Extension:** Introduce *drift-aware adapters* (small learnable layers inserted into U-Net encoder) that adapt to each client’s domain while preserving shared global weights. The server periodically aggregates shared parameters but excludes adapter weights.

**Algorithm Overview:**

1. Initialize global model ( M_0 ).
2. Distribute ( M_0 ) to all clients.
3. Clients train locally on their dataset and update local adapter layers.
4. Server aggregates shared parameters with FedAvg.
5. Repeat over new domains (continual tasks) without revisiting old data.

**Ablations:** With/without adapters, different aggregation strategies (FedProx, FedAvgM).

### 7. Experiments & Metrics

| Metric           | Description                | Goal        |
| ---------------- | -------------------------- | ----------- |
| Dice Coefficient | Tumor segmentation overlap | ≥ 0.85      |
| HD95             | Boundary accuracy          | ≤ 4 mm      |
| Forgetting Rate  | Accuracy loss over tasks   | ≤ 10%       |
| Convergence Time | Epochs until stability     | ≤ 20 epochs |

**Evaluation Plan:**

* Compare against centralized U-Net and standard FedAvg.
* Measure performance under sequential client updates.
* Perform cross-client validation to assess generalization.

### 8. Risks & Mitigations

| Risk                  | Description                 | Mitigation                        |
| --------------------- | --------------------------- | --------------------------------- |
| Limited GPU resources | Training slowdown           | Use smaller image crops, batch=4  |
| Data imbalance        | Some clients have less data | Weighted aggregation              |
| Forgetting            | Loss of earlier knowledge   | Adapter-based continual retention |
| Network instability   | Simulated FL on local host  | Offline aggregation script        |

### 9. Timeline & Roles

**Roadmap Snapshot:**

| Week | Milestone                           | Owner      | Due Date |
| ---- | ----------------------------------- | ---------- | -------- |
| W1   | Team setup & topic finalization     | All        | Oct 21   |
| W2   | Literature review & dataset split   | Komiljon   | Oct 28   |
| W3   | Baseline U-Net & FL setup           | Ismoil     | Nov 4    |
| W4   | Adapter module implementation       | Abdurashid | Nov 11   |
| W5   | Full training & ablation studies    | All        | Nov 18   |
| W6   | Results, analysis, and presentation | All        | Nov 25   |

**Roles (RACI):**

| Task               | Responsible | Accountable | Consulted | Informed |
| ------------------ | ----------- | ----------- | --------- | -------- |
| Dataset setup      | Komiljon    | Ismoil      | All       | All      |
| Model architecture | Ismoil      | Abdurashid  | Komiljon  | All      |
| Experiments        | Abdurashid  | Ismoil      | All       | All      |
| Documentation      | Ismoil      | Komiljon    | All       | All      |


### 10. Expected Outcomes

* Final trained FCL U-Net model.
* Evaluation report + comparison table.
* Demo video of client-server training rounds.
* Open-source repository with README, ROADMAP.md, and training scripts.

### 11. Ethics & Compliance

No patient data collection. All datasets are open and anonymized. The project follows responsible AI standards: fairness, privacy, and transparency. Dataset licenses (BraTS — CC BY 4.0) and citations will be included in the final report.

### 12. References

1. Li, T. et al. (2021). *Federated Optimization in Heterogeneous Networks*. PMLR.
2. Chen, M. et al. (2022). *FedProx: Mitigating Data Heterogeneity in Federated Learning*. IEEE TPAMI.
3. Zhao, L. et al. (2023). *FCL-U-Net: Federated Continual Learning for Medical Image Segmentation*. arXiv:2304.
4. Kamnitsas, K. et al. (2017). *Efficient Multi-Scale 3D CNNs for Brain Tumor Segmentation*. MICCAI.
5. Hsu, C. et al. (2019). *SplitFed: Splitting Models for Faster Federated Training*. arXiv:1909.

---
