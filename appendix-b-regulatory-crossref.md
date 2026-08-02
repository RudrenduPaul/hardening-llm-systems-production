> **Living document**: This file is updated as libraries and regulations evolve. Last updated: 2026-08-01.

# Appendix B: Regulatory Cross-Reference: EU AI Act, NIST AI 600-1, NIST AI 100-2, and ISO/IEC 42001

This appendix is your side-by-side mapping of four regulatory and standards frameworks, organized for engineers preparing for audits or regulatory inquiries. Use the tables and checklists here alongside Chapter 10 to determine which engineering artifacts satisfy which obligations across each framework. This appendix doesn't provide legal advice; interpretation of regulatory obligations belongs to legal counsel.

---

## B.1 How to use this appendix

The four frameworks covered here address overlapping but distinct concerns:

- **EU AI Act (Regulation 2024/1689)**: Binding EU law covering AI systems placed on the EU market or used by EU subjects. Requires conformity assessments, technical documentation (Annex IV), post-market monitoring, and incident reporting for high-risk systems. Penalties up to 35 million euros or 7% of global annual turnover.

- **NIST AI 600-1 (GenAI Profile)**: US voluntary framework with 400+ mitigation actions organized across 12 risk categories specific to generative AI systems. Widely used as a baseline for enterprise risk management and referenced in US federal AI governance guidance.

- **NIST AI 100-2 E2025 (Adversarial ML taxonomy)**: Technical taxonomy of attack types against AI systems, with mitigation guidance. Updated in 2025 to include GenAI-specific attack patterns (prompt injection, RAG poisoning, chain-of-thought manipulation).

- **ISO/IEC 42001**: International standard for AI management systems. Provides a process framework analogous to ISO 27001 for information security. Organizations can certify against it.

The cross-reference tables below show which framework elements share the same engineering artifact. Build the artifact once; reference it in all applicable frameworks.

---

## B.2 EU AI Act Annex III, High-risk system classification decision tree

Before producing Annex IV documentation, determine whether your system qualifies as high-risk under Annex III ([EU AI Act Annex III](https://artificialintelligenceact.eu/annex/3/)). High-risk obligations (conformity assessment, Annex IV documentation, post-market monitoring) apply only to high-risk systems.

**Step 1: Is your system a GPAI model?**
General-purpose AI models (foundation models) have separate obligations under EU AI Act Chapter V (Articles 51-56). If yes, the GPAI obligations apply regardless of Annex III classification.

**Step 2: Does your system fall into an Annex III category?**

| Annex III Category | Examples | Engineering implication |
|-------------------|----------|------------------------|
| Biometric identification and categorization | Facial recognition, emotion recognition | Full conformity assessment required |
| Critical infrastructure management | Power grid, water, transport AI | Full conformity assessment required |
| Education and vocational training | Student assessment, admission screening | Conformity assessment required |
| Employment and worker management | CV screening, performance monitoring | Conformity assessment required |
| Access to essential services | Credit scoring, insurance underwriting, social benefits eligibility | Conformity assessment required |
| Law enforcement | Predictive policing, evidence assessment | Conformity assessment + specific restrictions |
| Migration and border control | Asylum processing, visa assessment | Conformity assessment required |
| Administration of justice | Legal decision support | Conformity assessment required |

**Step 3: Does an exemption apply?**
AI systems used exclusively for research and development, narrow procedural tasks with human oversight, or spam detection may qualify for reduced obligations. Consult legal counsel.

**Step 4: Is your system excluded from high-risk classification?**
Annex III systems that don't pose significant risk to health, safety, or fundamental rights may be excluded: narrow scope, an AI that is a minor component, or a human making the final decision can each support an exclusion argument. The burden of demonstrating exclusion sits with the operator.

---

## B.3 EU AI Act Annex IV, Technical documentation checklist

The following checklist maps each Annex IV requirement to the engineering artifact that satisfies it. Items marked with the chapter reference indicate where in this book to find the implementation.

| Annex IV Requirement | Engineering Artifact | Chapter | Status |
|----------------------|---------------------|---------|--------|
| 1.1 General description of the AI system | System architecture diagram + model card | Ch 10 | |
| 1.2 Intended purpose and use cases | Use case specification document | Ch 10 | |
| 1.3 Version and change log | Git-tagged release notes, prompt template version log | Ch 10 | |
| 2.1 System components and interactions | Component diagram, API documentation | Ch 10 | |
| 2.2 Training data description | Data sheet (provenance, preprocessing, known biases) | Ch 10 | |
| 2.3 Development methodologies | MLOps documentation, evaluation methodology | Ch 10 | |
| 3.1 Monitoring and control measures | Observability dashboard config, alert thresholds | Ch 10, 11 | |
| 3.2 Human oversight mechanisms | Human-in-the-loop documentation (for agents: approval queue config) | Ch 7, 10 | |
| 3.3 Input data specifications | Input validation schema, PII handling policy | Ch 8 | |
| 4.1 Risk management documentation | Risk assessment, red-team report, bias assessment | Ch 6, 9, 10 | |
| 4.2 Robustness and cybersecurity measures | Injection defense architecture, adversarial test results | Ch 4, 6 | |
| 5.1 Lifecycle change documentation | Change log with dates, versions, approvals | Ch 10 | |
| 6.1 Conformity assessment results | Self-assessment checklist or third-party audit report | Ch 10 | |

Completeness target: 100% of required fields before production deployment of a high-risk system.

---

## B.4 NIST AI 600-1 risk category to EU AI Act article mapping

| NIST AI 600-1 Risk Category | EU AI Act Obligation | EU AI Act Article |
|----------------------------|---------------------|-------------------|
| CBRN Content (harmful content generation) | Prohibited AI practice prohibition | Article 5 |
| Confabulation (hallucination) | Accuracy and robustness requirements | Article 15 |
| Data Privacy (PII, memorization) | Data governance and management | Article 10 |
| Harmful Bias | Non-discrimination, fundamental rights | Articles 9, 10 |
| Data Disclosure (system prompt, training data) | Security requirements, information to users | Articles 13, 15 |
| Human-AI Configuration | Human oversight requirements | Article 14 |
| Information Security (adversarial attacks) | Cybersecurity and robustness | Article 15 |
| Intellectual Property | Transparency about training data | Article 53 (GPAI) |
| Obscene/Harmful Content | Prohibited practices, operator obligations | Articles 5, 26 |
| Operational Risk | Post-market monitoring, incident reporting | Articles 72, 73 |
| Value Chain (supply chain) | Provider obligations, transparency | Articles 25, 53 |
| Environmental Risk | Transparency about energy use | Article 53 (GPAI) |

---

## B.5 NIST AI 100-2 E2025 adversarial taxonomy to OWASP LLM Top 10 mapping

NIST AI 100-2 E2025 provides a taxonomy of adversarial attacks against AI/ML systems, updated in 2025 to include GenAI-specific patterns ([NIST AI 100-2 E2025](https://csrc.nist.gov/pubs/ai/100/2/e2025/final)). The following table maps the AI 100-2 attack classes to their OWASP LLM counterparts.

| NIST AI 100-2 Attack Class | OWASP LLM Top 10 Equivalent | This Book |
|---------------------------|----------------------------|-----------|
| Evasion attacks (adversarial inputs) | LLM01 Prompt Injection, LLM09 Misinformation | Ch 4, 6 |
| Poisoning attacks (training data) | LLM04 Data and Model Poisoning | Ch 5, 6 |
| Privacy attacks (membership inference, extraction) | LLM02 Sensitive Information Disclosure | Ch 9 |
| Abuse attacks (misuse of legitimate capabilities) | LLM06 Excessive Agency, LLM10 Unbounded Consumption | Ch 7, 11 |
| Prompt injection (GenAI-specific) | LLM01 Prompt Injection | Ch 4 |
| Clean-label poisoning (GenAI-specific) | LLM04 Data and Model Poisoning | Ch 5, 6 |
| RAG-specific attacks (GenAI-specific) | LLM08 Vector and Embedding Weaknesses | Ch 5 |

---

## B.6 ISO/IEC 42001 controls to engineering artifact mapping

ISO/IEC 42001 specifies requirements for AI management systems, the organizational processes, policies, and controls that govern how AI systems are developed, deployed, and monitored ([ISO](https://www.iso.org/standard/42001)).

The following table maps ISO 42001 clause requirements to the engineering artifacts in this book that provide evidence of compliance.

| ISO 42001 Clause | Requirement Summary | Engineering Artifact |
|-----------------|--------------------|--------------------|
| 4.1 Understanding the organization | Document AI context and stakeholder needs | System architecture + use case specification |
| 4.2 Understanding stakeholder needs | Identify interested parties and their requirements | Risk register, stakeholder impact assessment |
| 5.2 AI policy | Establish and communicate AI policy | AI governance policy document (not covered in this book, organizational) |
| 6.1 Risk and opportunity assessment | Identify and assess AI-specific risks | Risk assessment document, red-team report |
| 7.5 Documented information | Maintain required documentation | Annex IV package, evaluation results |
| 8.4 AI system impact assessment | Assess impacts before deployment | Bias assessment, PII impact assessment |
| 9.1 Monitoring and measurement | Establish AI performance monitoring | Observability dashboards, hallucination rate tracking |
| 10.1 Continual improvement | Process for improving AI systems | Incident post-mortems, bias trend analysis |

---

## B.7 Full cross-reference index

The following index maps each major engineering artifact in this book to the regulatory frameworks it addresses. Use this index when assembling compliance packages to identify which artifacts to include.

| Engineering Artifact | EU AI Act | NIST AI 600-1 | NIST AI 100-2 | ISO 42001 |
|---------------------|-----------|----------------|----------------|-----------|
| System architecture diagram | Annex IV 1.1 | Operational Risk | -- | 7.5 |
| Model card | Annex IV 1.2, 2.2 | Confabulation | -- | 7.5 |
| Hallucination rate CI gate | Article 15 | Confabulation | -- | 9.1 |
| Red-team report | Articles 9, 15 | Information Security | Evasion, Abuse | 6.1, 8.4 |
| Bias assessment | Articles 9, 10 | Harmful Bias | -- | 8.4 |
| PII detection pipeline | Article 10, 13 | Data Privacy | Privacy attacks | 8.4 |
| Memorization probe results | Article 10 | Data Privacy | Privacy attacks | 8.4 |
| Incident post-mortem | Article 73 | Operational Risk | -- | 10.1 |
| Post-market monitoring report | Article 72 | Operational Risk | -- | 9.1 |
| Audit log (tamper-evident) | Article 73 | Data Disclosure | -- | 7.5, 9.1 |
| Annex IV package | Annex IV | All categories | -- | 7.5 |
| Agent scope test results | Article 14, 15 | Human-AI Config, Value Chain | Evasion | 8.4 |
| MCP tool allowlist | Article 15, 26 | Value Chain | Abuse | 6.1 |
| Output provenance records | Article 13, 73 | Data Disclosure | -- | 7.5 |

---

## References

European Parliament and Council of the European Union. (2024). *Regulation (EU) 2024/1689: Artificial Intelligence Act.* https://eur-lex.europa.eu/eli/reg/2024/1689

European Parliament and Council of the European Union. (2024). *EU AI Act implementation timeline.* https://artificialintelligenceact.eu/implementation-timeline/

European Parliament and Council of the European Union. (2024). *EU AI Act Annex III.* https://artificialintelligenceact.eu/annex/3/

European Parliament and Council of the European Union. (2024). *EU AI Act Annex IV.* https://artificialintelligenceact.eu/annex/4/

National Institute of Standards and Technology. (2024). *Artificial intelligence risk management framework: Generative artificial intelligence profile* (NIST AI 600-1). https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence

National Institute of Standards and Technology. (2025). *Adversarial machine learning: A taxonomy and terminology of attacks and mitigations* (NIST AI 100-2 E2025). https://csrc.nist.gov/pubs/ai/100/2/e2025/final

International Organization for Standardization. (2023). *ISO/IEC 42001: Artificial intelligence, Management system.* https://www.iso.org/standard/42001
