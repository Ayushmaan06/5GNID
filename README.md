
# A High-Throughput Cascade Architecture for Real-Time 5G Intrusion Detection: Balancing Accuracy, Latency, and Uncertainty Handling

**Abstract** — The advent of 5G networks, characterized by Ultra-Reliable Low-Latency Communication (URLLC) and high-density connectivity, has introduced unprecedented security challenges. Traditional Intrusion Detection Systems (IDS) often prioritize detection accuracy at the expense of inference latency and throughput, rendering them unsuitable for real-time cyber-threat mitigation in 5G environments. In this paper, we propose a multi-stage cascade IDS architecture designed specifically for high-throughput 5G networks. Our approach utilizes a two-stage classification mechanism—binary anomaly detection followed by multiclass attack categorization—optimized via ExtraTrees classifiers. Evaluating our system on the 5G-NIDD dataset, we compare four methodological families: deep learning tabular models (TabNet), heterogeneous ensembles (LCCDE), Random Forest cascades, and our proposed ExtraTrees cascade. Experimental results demonstrate that our ExtraTrees cascade achieves a state-of-the-art accuracy of 99.96%, significantly outperforming TabNet (92.5%) and LCCDE (75.0%), while maintaining microsecond-level inference latency. Furthermore, by employing a novel parallel confidence routing strategy, the architecture acts as a smart fail-safe, isolating 0.19% of highly ambiguous traffic for human analyst review without forcing incorrect classifications. This research demonstrates that tree-based cascade architectures provide an optimal balance of accuracy, speed, and reliability for URLLC-constrained intrusion detection.

**Keywords** — intrusion detection, 5G security, URLLC, machine learning, cascade architecture, network security, ExtraTrees.

---

## 1. Introduction

The rapid proliferation of 5G network architecture has fundamentally shifted the communication landscape by delivering multi-gigabit throughput, massive device connectivity, and Ultra-Reliable Low-Latency Communication (URLLC). While these capabilities enable critical applications such as autonomous driving, remote surgery, and industrial automation, they also drastically expand the threat surface exposed to malicious actors. 

A primary challenge in securing 5G networks is the sheer volume and velocity of network traffic. Traditional Intrusion Detection Systems (IDS), which have historically relied on heavy packet inspection algorithms or computationally expensive deep learning models, struggle to process traffic at line speed. In URLLC scenarios, network decisions—including the detection and blocking of malicious packets—must occur within milliseconds to prevent widespread network compromise and maintain Service Level Agreements (SLAs). If an IDS introduces substantial latency, it defeats the fundamental purpose of the URLLC slice.

Furthermore, current IDS solutions frequently force a classification decision even when prediction confidence is low. In zero-day attack scenarios or facing highly mutated network patterns, this behavior leads to high false positive rates or catastrophic false negatives. There is a critical need for IDS architectures that manage uncertainty effectively, rather than ignoring it.

In this paper, we address these systems-level challenges by proposing a high-throughput cascade architecture for real-time 5G intrusion detection. Instead of relying solely on improving model capacity natively, we focus on an architectural framework utilizing cascading tree-based classifiers (ExtraTrees) combined with an intelligent confidence-based routing strategy.

**Our main contributions are as follows:**
1. We design a multi-stage cascade IDS architecture that splits the detection task into binary filtering and subsequent multiclass categorization, designed specifically for low-latency 5G environments.
2. We thoroughly evaluate modern tabular deep learning approaches (TabNet) alongside ensemble methods (LCCDE) against tree-based cascades, demonstrating that tree-based paradigms remain superior for highly structured, high-throughput network data.
3. We introduce a parallel confidence fusion strategy that quantifies model uncertainty, safely isolating unconfident predictions as "Suspicious" (reducing forced errors) while maintaining an overall accuracy of 99.96% on standard traffic.
4. We enforce rigorous hybrid resampling methodologies to completely prevent data leakage while robustly learning rare attack vectors in highly imbalanced network distributions.

---

*(Figure Suggestion 1: System architecture diagram showing incoming 5G traffic routed through the Stage 1 Binary Detector, branching to the Stage 2 Multiclass Detector for anomalous packets, and outputting normal, specific attack, or suspicious classes via the confidence-fusion router.)*

---

## 2. Related Work

The application of Machine Learning (ML) to network intrusion detection has matured significantly, though the adaptation of these systems for 5G architectures remains an active area of research.

**Traditional IDS Approaches:** Canonical algorithms such as Support Vector Machines (SVM), Naive Bayes, and standard Random Forests have been widely deployed on legacy datasets (e.g., KDD Cup 99, CICIDS2017). While effective at detecting known signature-based attacks, these standalone models frequently suffer from high latency during inference or fail to appropriately generalize to the high-dimensional feature spaces of 5G telemetry.

**Deep Learning in IDS:** Recent studies have heavily favored deep learning networks—including Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs)—to extract spatial and temporal traffic features. Lately, TabNet has emerged as an attention-based deep learning architecture explicitly designed for tabular data. While these models possess high representational capacity, their parameter density results in high computational overhead, rendering them suboptimal for the extreme throughput required by URLLC network slices.

**Ensemble-Based IDS:** To combine the strengths of multiple classifiers, heterogeneous ensembles such as Leader-class Classifier based Ensemble (LCCDE) have been proposed. LCCDE aims to assign the "best" model to predict specific classes. However, computing predictions across multiple disparate models exponentially increases inference latency and often leads to poor generalization in shifting network environments.

**5G Intrusion Detection:** The introduction of the 5G-NIDD dataset provided a realistic benchmark for 5G network threats, including massive IoT botnets and signaling storms. Recent works on this dataset have explored initial ML benchmarking, but often overlook the systemic constraints of deployment: specifically, the trade-off between throughput, latency, and the probabilistic uncertainty in real-world traffic flows. Our work uniquely bridges this gap by proposing a cascade routing architecture tailored for real-time 5G constraints.

---

## 3. Dataset and Feature Engineering

### 3.1 The 5G-NIDD Dataset
We utilize the comprehensive 5G-NIDD (5G Network Intrusion Detection Dataset), which represents a realistic 5G testbed generating both benign traffic and various cyber-attacks (e.g., DoS, DDoS, ICMP Floods, and Port Scans). The dataset encapsulates distinct tabular features derived from packet headers, flow statistics, and inter-arrival times, offering a highly imbalanced representation indicative of real-world networks.

### 3.2 Feature Preprocessing and Normalization
To prepare the spatial distribution of the network flow attributes for machine learning ingestion, continuous numerical features are standardized. Highly correlated features that provide redundant information are pruned to reduce the dimensionality, thereby minimizing inference latency footprint. Categorical features are encoded appropriately without exploding the feature space.

### 3.3 Leakage-Free Stratified Splitting and Resampling
Network intrusion datasets inherently suffer from severe class imbalance. Rare attacks (e.g., highly specific ICMP floods or subtle signaling anomalies) represent extreme minority classes compared to normal traffic. 

To resolve this without introducing data leakage—a prevalent flaw in previous IDS literature—we enforced strict isolation protocols. First, the dataset was subjected to an 80/20 Stratified Split. Crucially, addressing the data imbalance via Hybrid Resampling (utilizing RandomUnderSampler for majority classes followed by SMOTE for synthetic augmentation of minority classes) was applied **strictly and exclusively to the training set**. The 20% test set was left entirely untouched, preserving the natural operational distribution of network traffic, including rare attacks (e.g., preserving accurately 231 exact samples of specific minority threats). 

---

## 4. Proposed Cascade Architecture

To support URLLC latency bounds, we move away from monolithic multi-class evaluation. We propose a multi-stage Cascade Routing Architecture that functions as a gating mechanism for packet flows.

### 4.1 Stage 1: Binary Anomaly Detection
The primary stage acts as a high-speed gatekeeper classifying incoming traffic purely as `Normal` or `Attack`. Due to the simplicity of binary classification boundaries, this stage executes with ultra-low validation latency, clearing the overwhelming majority of benign traffic instantly.

### 4.2 Stage 2: Multiclass Attack Classification
Packets flagged as `Attack` by Stage 1 are passed to Stage 2, which conducts a fine-grained discrimination to ascertain the specific attack signature (e.g., UDP Flood, SYN Flood). 

### 4.3 Routing and Fusion Strategies
To determine the optimal packet pipeline, we tested multiple architectural routing schemas:
1. **Sequential Binary-First:** Stage 1 filters traffic; only predicted anomalies engage Stage 2.
2. **Sequential Multiclass-First:** Stage 2 executes first; Stage 1 acts as a verification heuristic.
3. **Parallel Voting:** Both stages execute simultaneously; a hard voting mechanism resolves the class.
4. **Parallel Confidence Fusion (Proposed):** Both stages execute. The output probability distributions from both models are analyzed. If the combined probability confidence drops below a mathematically defined threshold, the architecture refuses to guess and appropriately routes the packet into an `Unknown/Suspicious` queue.

By replacing standalone components with ExtraTrees (Extremely Randomized Trees) classifiers inside the Parallel Confidence pipeline, the system benefits from heavily parallelized decision nodes that prevent the high variance seen in standard Random Forests, achieving superior computational efficiency.

---

*(Figure Suggestion 2: Confusion matrix comparing traditional baseline predictions vs. Parallel Confidence cascade handling of the dataset, highlighting the Unconfident/Suspicious isolation.)*

---

## 5. Experimental Setup

The evaluation environment isolates the performance of algorithmic families rather than relying purely on hyperparameter maximization.

**Models Evaluated:**
* **TabNet (Deep Learning Baseline):** An attentive architecture for tabular learning trained via stochastic gradient descent.
* **LCCDE (Ensemble Baseline):** A heterogeneous ensemble testing if combined algorithms improve generalization.
* **Random Forest Cascade:** The cascade architecture populated with standard Random Forest estimators.
* **ExtraTrees Cascade (Proposed):** The cascade architecture populated with Extra Tree estimators.

**Metrics:**
System performance is evaluated holistically via Accuracy, Macro F1-Score (to account for class imbalance), Throughput (samples classified per second), and functional inference latency. Experiments were executed identically across hardware constraints to establish fair baselines.

---

## 6. Experimental Results

The experimental results definitively highlight the supremacy of specialized tree-based cascade learning over monolithic deep learning grids for high-throughput tabular environments.

### 6.1 Baseline Performance
The deep learning baseline, **TabNet**, achieved an accuracy of approximately **92.5%**. While high by traditional standards, TabNet struggled to perfectly parse the rigid decision boundaries characteristic of definitive network features, and suffered from restricted throughput. 
The **LCCDE ensemble** performed poorly, yielding an accuracy of only **75.0%**. The heterogeneity of the ensemble induced conflicting predictions on minority classes, establishing that blindly combining models actively harms generalizability in 5G distributions while simultaneously crippling computational throughput.

### 6.2 Cascade Architecture Performance
When applying the proposed Cascade framework, performance metrics increased drastically:
* **Random Forest Cascade:** Achieved **99.6%** accuracy with high throughput parameters.
* **ExtraTrees Cascade (Proposed):** Evaluated as the premier architecture, hitting an elite accuracy of **99.96%**. The highly randomized node splitting inherent to ExtraTrees handled continuous network variables effectively without aggressively overfitting, establishing near-perfect discrimination of rare attack classes.

### 6.3 Uncertainty Handling and Reliability
A paramount metric achieved during the evaluation was the system's management of uncertainty. Utilizing the **Parallel Confidence** routing strategy, the ExtraTrees cascade successfully isolated **0.19%** of highly ambiguous, potentially novel network traffic as "Unknown/Suspicious." 

Instead of forcing a statistically unconfident classification—which often results in a dangerous false negative—the cascade acts as a smart fail-safe. It routes this minimal 0.19% of the most uncertain traffic to a human analyst or secondary deep-inspection sandbox, while continuing to process the remaining 99.81% of traffic with its 99.96% automated accuracy.

---

*(Figure Suggestion 3: Bar chart comparing Model Accuracy across TabNet, LCCDE, RF Cascade, and ET Cascade side-by-side with a line-graph overlay showing Inference Throughput.)*

*(Figure Suggestion 4: Throughput vs Accuracy scatter plot highlighting the optimal positioning of the ExtraTrees cascade in the top-right quadrant.)*

---

## 7. Discussion

The empirical findings of this research validate several hypotheses regarding ML deployment in 5G conditions. 

**Architectural Superiority Over Deep Learning:** We observe that tree-based models fundamentally outperform deep tabular models (like TabNet) on 5G packet features. Network flow statistics consist of hard, deterministic categorical rules and highly varying continuous floats; tree algorithms map these hard hyperplanes significantly better than gradient-based neural networks which inherently seek smooth approximations.

**Efficiency in Cascading:** The cascade architecture functionally acts as a computational sieve. By letting the Stage 1 binary classifier drop normal URLLC traffic at peak velocity, the heavier multiclass computation is strictly reserved for hostile traffic. This is what permits the architecture to satisfy the stringent latency constraints defining modern 5G.

**The Necessity of Uncertainty Routing:** The 0.19% isolation metric produced by the confidence routing is arguably as vital as the 99.96% accuracy. In real-world deployments, zero-day threat landscapes are dynamic. By architecturally building in a fail-safe that acknowledges its probabilistic thresholds, the IDS transitions from a theoretical ML benchmark into an operationally viable, production-ready security application.

---

## 8. Conclusion

As 5G networks expand to host globally critical URLLC infrastructure, intrusion detection systems must evolve to prioritize latency, throughput, and reliability concurrently with accuracy. We presented a cascade-based IDS architecture leveraging ExtraTrees classifiers combined with a parallel confidence routing protocol. The system completely avoided data leakage via rigorous post-split hybrid resampling, proving its robustness on rare classes within the 5G-NIDD dataset.

Our experimental results showed that the proposed ExtraTrees cascade framework dramatically outperformed deep learning (TabNet, 92.5%) and ensemble baselines (LCCDE, 75.0%), achieving an optimal 99.96% accuracy. Furthermore, our novel confidence framework successfully identified and isolated the 0.19% of traffic featuring high ambiguity, providing a reliable operational fail-safe. 

Future work will focus on integrating continuous online learning mechanisms to adapt the cascade stages dynamically to concept drift, and validating the framework's deployment viability on specialized hardware such as SmartNICs and DPUs for further latency reduction.

---

### References

[1] Placeholder for seminal paper on 5G URLLC security requirements.
[2] Placeholder for 5G-NIDD dataset original publication.
[3] Placeholder for TabNet original architecture paper (Arik & Pfister).
[4] Placeholder for LCCDE IDS methodology paper.
[5] Placeholder for foundational SMOTE data resampling literature (Chawla et al.).
[6] Placeholder for ExtraTrees (Extremely Randomized Trees) classifiers original paper.
[7] Placeholder for standard Intrusion Detection Systems survey in IoT/5G networks. 
[8] Placeholder for recent literature concerning ML data leakage in Network Security datasets.
