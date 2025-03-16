# In the Picture: Medical Imaging Datasets, Artifacts, and their Living Review

**Authors:** Amelia Jiménez-Sánchez, Natalia-Rozalia Avlona, Sarah de Boer, Víctor M. Campello, Aasa Feragen, Enzo Ferrante, Melanie Ganz, Judy Wawira Gichoya, Camila González, Steff Groefsema, Alessa Hering, Adam Hulman, Leo Joskowicz, Dovile Juodelyte, Melih Kandemir, Thijs Kooi, Jorge del Pozo Lérida, Livie Yumeng Li, Andre Pacheco, Tim Rädsch, Mauricio Reyes, Théo Sourget, Bram van Ginneken, David Wen, Nina Weng, Jack Junchi Xu, Hubert Dariusz Zając, Maria A. Zuluaga, Veronika Cheplygina

**Published:** 2025-01-18

**URL:** http://arxiv.org/abs/2501.10727v1

## Abstract

Datasets play a critical role in medical imaging research, yet issues such as
label quality, shortcuts, and metadata are often overlooked. This lack of
attention may harm the generalizability of algorithms and, consequently,
negatively impact patient outcomes. While existing medical imaging literature
reviews mostly focus on machine learning (ML) methods, with only a few focusing
on datasets for specific applications, these reviews remain static -- they are
published once and not updated thereafter. This fails to account for emerging
evidence, such as biases, shortcuts, and additional annotations that other
researchers may contribute after the dataset is published. We refer to these
newly discovered findings of datasets as research artifacts. To address this
gap, we propose a living review that continuously tracks public datasets and
their associated research artifacts across multiple medical imaging
applications. Our approach includes a framework for the living review to
monitor data documentation artifacts, and an SQL database to visualize the
citation relationships between research artifact and dataset. Lastly, we
discuss key considerations for creating medical imaging datasets, review best
practices for data annotation, discuss the significance of shortcuts and
demographic diversity, and emphasize the importance of managing datasets
throughout their entire lifecycle. Our demo is publicly available at
http://130.226.140.142.

## Summary

### Comprehensive Summary of the Research Paper:

#### 1. **Key Findings**:
   - The paper introduces a **living review framework** to continuously track and update medical imaging datasets and their associated research artifacts (e.g., biases, shortcuts, and additional annotations).
   - It highlights the importance of **dataset quality** (e.g., label accuracy, demographic diversity, and metadata) in ensuring the generalizability and clinical utility of machine learning (ML) models.
   - The authors propose an **SQL database** to visualize citation relationships between datasets and research artifacts, facilitating the discovery of emerging dataset insights.
   - The paper emphasizes the need for **dynamic dataset management** throughout their lifecycle, addressing issues like shortcuts, biases, and evolving annotations.
   - A **public demo** of the living database is released, showcasing 16 datasets and 24 research artifacts across two medical imaging applications.

---

#### 2. **Research Question/Problem**:
   - The paper addresses the **lack of attention to dataset quality** in medical imaging research, particularly issues like label accuracy, shortcuts, biases, and metadata. These shortcomings can harm the generalizability of ML models and negatively impact patient outcomes.
   - Existing literature reviews in medical imaging are **static** and focus primarily on ML methods, failing to account for emerging dataset insights (e.g., biases or additional annotations) after publication.
   - The authors aim to bridge this gap by proposing a **living review framework** that continuously updates and tracks datasets and their associated research artifacts.

---

#### 3. **Methodology**:
   - The authors conducted a **year-long collaborative webinar** and an **in-person workshop** involving 50 researchers from academia, industry, and clinical practice across 10+ countries.
   - They developed a **living review framework** that includes:
     - A system to monitor and document **research artifacts** (e.g., biases, shortcuts, and annotations) associated with datasets.
     - An **SQL database** to visualize citation relationships between datasets and research artifacts.
   - The framework was applied to **16 datasets** and **24 research artifacts** across two medical imaging applications, with a public demo released for community use.
   - The paper also synthesizes best practices for dataset creation, annotation, demographic diversity, and lifecycle management.

---

#### 4. **Results**:
   - The living review framework successfully tracks and updates datasets with emerging research artifacts, providing a dynamic resource for the medical imaging community.
   - The **SQL database** enables visualization of citation relationships, helping researchers discover new insights about datasets.
   - The **public demo** showcases the framework’s functionality, demonstrating its potential to improve dataset documentation and usage.
   - The paper provides a comprehensive discussion of best practices for dataset creation, annotation, and management, addressing key challenges like shortcuts, biases, and demographic diversity.

---

#### 5. **Implications**:
   - The living review framework addresses a critical gap in medical imaging research by ensuring datasets remain **up-to-date** and **well-documented**, improving the reliability and generalizability of ML models.
   - By emphasizing dataset quality and lifecycle management, the framework has the potential to enhance **patient outcomes** and reduce biases in clinical applications.
   - The proposed approach encourages **collaboration** within the research community, enabling continuous updates and contributions to dataset documentation.
   - The framework sets a precedent for **dynamic reviews** in other fields, moving beyond static literature reviews to incorporate evolving evidence and insights.

This work represents a significant step toward improving the quality and utility of medical imaging datasets, with broad implications for ML research and clinical practice.

