"""Legal document analysis tasks: patent prior art, contract risk, regulatory filing review.

Each factory generates a realistic multi-document scenario requiring cross-reference
analysis. Seeds control randomization of planted issues, false positives, claim
structures, contract types, filing data, etc.
"""

from __future__ import annotations

import random as _random

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools
from ..content_pools import (
    random_name,
    random_names,
    pick1,
    COMPANY_NAMES,
)


# =============================================================================
# HELPERS
# =============================================================================


def _fmt_money(amount: float) -> str:
    """Format a float as a dollar string with commas."""
    return f"${amount:,.2f}"


def _fmt_pct(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string."""
    return f"{value:.{decimals}f}%"


def _ordinal(n: int) -> str:
    """Return ordinal string for an integer (1st, 2nd, 3rd, ...)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# =============================================================================
# DOMAIN: PATENT PRIOR ART
# =============================================================================

TECH_DOMAINS = [
    {
        "field": "Machine Learning Image Classification",
        "abstract_prefix": "A system and method for classifying images using",
        "feature_pool": [
            "convolutional neural network with attention mechanism",
            "multi-scale feature extraction pyramid",
            "real-time inference pipeline with batch normalization",
            "transfer learning from pre-trained backbone model",
            "data augmentation using geometric transformations",
            "ensemble prediction with model averaging",
            "feature map pooling using spatial pyramid",
            "gradient-based class activation mapping for interpretability",
            "hardware-accelerated inference on edge devices",
            "adaptive learning rate scheduling with warm restarts",
            "automated hyperparameter tuning via Bayesian optimization",
            "knowledge distillation to compact student model",
        ],
        "key_terms": {
            "attention mechanism": "A trainable weighting scheme that selectively focuses on relevant spatial or channel features within a feature map.",
            "feature extraction pyramid": "A hierarchical structure that processes input at multiple resolutions to capture both fine-grained and coarse semantic information.",
            "batch normalization": "A technique that normalizes layer inputs by re-centering and re-scaling, applied per mini-batch during training.",
            "transfer learning": "The practice of initializing model weights from a model pre-trained on a different but related task or dataset.",
            "edge device": "A computing device located at the periphery of a network, such as a mobile phone, IoT sensor, or embedded processor.",
        },
    },
    {
        "field": "Autonomous Vehicle Navigation",
        "abstract_prefix": "A method and apparatus for navigating an autonomous vehicle using",
        "feature_pool": [
            "LiDAR point cloud fusion with camera imagery",
            "predictive path planning using Monte Carlo tree search",
            "real-time obstacle detection and avoidance with safety margins",
            "HD map integration for lane-level localization",
            "vehicle-to-vehicle communication for cooperative perception",
            "reinforcement learning policy for intersection handling",
            "sensor fusion with Kalman filtering for state estimation",
            "failsafe mode with graceful degradation on sensor failure",
            "pedestrian intent prediction using trajectory analysis",
            "weather-adaptive perception parameter adjustment",
            "dynamic speed profiling based on road geometry",
            "redundant computation across independent processing units",
        ],
        "key_terms": {
            "point cloud fusion": "The process of combining 3D spatial data from LiDAR sensors with 2D image data from cameras to create a unified scene representation.",
            "predictive path planning": "An algorithm that computes a future trajectory for a vehicle by simulating potential outcomes over a planning horizon.",
            "safety margin": "A buffer zone around detected obstacles within which the vehicle must not plan any trajectory.",
            "cooperative perception": "A technique where multiple vehicles share sensor data to extend their individual perception range beyond line-of-sight.",
            "graceful degradation": "A design principle where system performance decreases proportionally to component failures rather than failing catastrophically.",
        },
    },
    {
        "field": "Blockchain-Based Supply Chain Tracking",
        "abstract_prefix": "A decentralized system for tracking supply chain provenance using",
        "feature_pool": [
            "immutable ledger recording with cryptographic hash chains",
            "smart contract-based automated compliance verification",
            "IoT sensor integration for real-time condition monitoring",
            "zero-knowledge proof for confidential transaction details",
            "multi-party consensus mechanism with delegated validators",
            "tokenized asset representation with fractional ownership",
            "cross-chain interoperability bridge for multi-network tracking",
            "off-chain storage with on-chain content addressing",
            "role-based access control with hierarchical permissions",
            "automated recall trigger based on sensor threshold breach",
            "regulatory audit trail with timestamped event log",
            "geofencing integration for location-based compliance checks",
        ],
        "key_terms": {
            "cryptographic hash chain": "A sequence of data blocks where each block contains a hash of the previous block, ensuring tamper-evident integrity.",
            "smart contract": "A self-executing program stored on a blockchain that automatically enforces agreed-upon conditions when triggered.",
            "zero-knowledge proof": "A cryptographic method allowing one party to prove possession of information without revealing the information itself.",
            "content addressing": "A storage paradigm where data is referenced by its cryptographic hash rather than by a location-based identifier.",
            "delegated validator": "A network participant elected by stakeholders to verify transactions and maintain consensus on behalf of the network.",
        },
    },
    {
        "field": "Natural Language Processing for Legal Documents",
        "abstract_prefix": "A computer-implemented method for analyzing legal documents using",
        "feature_pool": [
            "transformer-based named entity recognition for legal entities",
            "clause classification using fine-tuned language model",
            "cross-reference resolution across multi-document filings",
            "temporal reasoning for deadline and obligation extraction",
            "semantic similarity matching between contract provisions",
            "hierarchical document parsing with section-level embeddings",
            "contradiction detection between document clauses",
            "automated risk scoring based on extracted obligation terms",
            "jurisdiction-aware legal terminology normalization",
            "citation graph construction from case law references",
            "confidentiality classification using contextual embeddings",
            "multi-language support with aligned embedding spaces",
        ],
        "key_terms": {
            "named entity recognition": "The task of identifying and classifying named entities (persons, organizations, dates, monetary values) in unstructured text.",
            "clause classification": "The process of categorizing contractual provisions into predefined types such as indemnification, limitation of liability, or termination.",
            "cross-reference resolution": "The linking of internal references (e.g., 'as defined in Section 3.2') to their target definitions or provisions.",
            "temporal reasoning": "Inference about time-related constraints, including deadlines, effective dates, and duration calculations.",
            "contextual embedding": "A vector representation of a word or phrase that varies based on its surrounding context within a document.",
        },
    },
    {
        "field": "Distributed Cloud Resource Orchestration",
        "abstract_prefix": "A system for dynamically orchestrating cloud computing resources using",
        "feature_pool": [
            "predictive autoscaling based on time-series workload forecasting",
            "container orchestration with affinity and anti-affinity constraints",
            "cost-aware resource placement across multi-cloud providers",
            "service mesh with automatic circuit breaker and retry logic",
            "stateful workload migration with minimal downtime",
            "resource quotas with hierarchical namespace enforcement",
            "automated anomaly detection in resource consumption patterns",
            "green computing optimization to minimize carbon footprint",
            "latency-aware traffic routing across geographic regions",
            "serverless function chaining with event-driven triggers",
            "infrastructure-as-code reconciliation with drift detection",
            "chaos engineering integration for resilience verification",
        ],
        "key_terms": {
            "predictive autoscaling": "The dynamic adjustment of allocated compute resources based on forecasted future demand rather than current utilization only.",
            "affinity constraint": "A scheduling rule that co-locates related workloads on the same or nearby compute nodes to reduce inter-process latency.",
            "circuit breaker": "A fault-tolerance pattern that prevents cascading failures by halting requests to a failing service after a threshold of errors.",
            "drift detection": "The automated identification of differences between the declared desired state of infrastructure and its actual running state.",
            "event-driven trigger": "An invocation mechanism where a function or process executes in response to a specific event rather than on a fixed schedule.",
        },
    },
    {
        "field": "Wearable Health Monitoring Systems",
        "abstract_prefix": "A wearable device and method for continuous health monitoring using",
        "feature_pool": [
            "photoplethysmography sensor with motion artifact cancellation",
            "electrodermal activity measurement for stress detection",
            "multi-axis accelerometer fusion for activity classification",
            "continuous blood pressure estimation using pulse transit time",
            "adaptive sampling rate based on detected activity level",
            "edge-computed arrhythmia detection with low-power neural network",
            "encrypted health data transmission to cloud analytics platform",
            "personalized baseline calibration from longitudinal user data",
            "battery life optimization through dynamic sensor duty cycling",
            "fall detection with automatic emergency notification",
            "skin temperature monitoring with circadian rhythm tracking",
            "medication adherence reminders based on physiological indicators",
        ],
        "key_terms": {
            "photoplethysmography": "An optical measurement technique that detects blood volume changes in tissue by illuminating the skin and measuring light absorption variations.",
            "pulse transit time": "The time interval for a pressure wave to travel between two arterial sites, used as a surrogate measure for blood pressure.",
            "motion artifact": "Noise introduced into a physiological signal by physical movement of the sensor relative to the body.",
            "dynamic duty cycling": "A power management strategy that varies sensor sampling frequency based on context to conserve battery life.",
            "longitudinal data": "Data collected from the same subject over an extended period, enabling detection of individual trends and deviations.",
        },
    },
    {
        "field": "Quantum-Resistant Cryptographic Protocols",
        "abstract_prefix": "A method for securing digital communications against quantum computing attacks using",
        "feature_pool": [
            "lattice-based key encapsulation with ring structure",
            "hash-based digital signature with stateful tree traversal",
            "code-based encryption using binary Goppa codes",
            "hybrid classical-quantum key exchange with fallback mechanism",
            "post-quantum TLS handshake with backward compatibility",
            "multi-party key agreement using isogeny-based primitives",
            "hardware security module integration for key generation",
            "side-channel resistant implementation with constant-time operations",
            "key rotation protocol with forward secrecy guarantees",
            "certificate transparency log for quantum-safe certificates",
            "threshold signature scheme for distributed key management",
            "formal verification of protocol security properties",
        ],
        "key_terms": {
            "lattice-based cryptography": "A family of cryptographic constructions whose security relies on the hardness of problems in mathematical lattices, believed resistant to quantum attacks.",
            "key encapsulation": "A mechanism to securely transport a symmetric key using asymmetric encryption, allowing the recipient to decrypt subsequent communications.",
            "forward secrecy": "A property ensuring that compromise of long-term keys does not compromise past session keys.",
            "side-channel attack": "An attack that exploits physical implementation characteristics (timing, power consumption, electromagnetic emissions) rather than algorithmic weaknesses.",
            "isogeny": "A structure-preserving map between elliptic curves, used in certain post-quantum cryptographic schemes.",
        },
    },
    {
        "field": "Precision Agriculture Drone Systems",
        "abstract_prefix": "An unmanned aerial system for precision agriculture comprising",
        "feature_pool": [
            "multispectral imaging for crop health index computation",
            "automated flight path optimization over irregular field boundaries",
            "variable-rate application of inputs based on prescription maps",
            "real-time weed detection using onboard image classification",
            "terrain-following altitude control using downward-facing rangefinder",
            "multi-drone coordination with collision avoidance protocol",
            "soil moisture estimation from thermal infrared imagery",
            "data pipeline from drone to farm management information system",
            "regulatory-compliant geofencing with no-fly zone enforcement",
            "battery swap scheduling for continuous coverage operations",
            "wind compensation for consistent spray distribution patterns",
            "crop row detection for inter-row precision navigation",
        ],
        "key_terms": {
            "multispectral imaging": "The capture of image data at specific wavelength bands across the electromagnetic spectrum beyond visible light, including near-infrared.",
            "prescription map": "A georeferenced data layer specifying spatially variable application rates for agricultural inputs such as fertilizer or pesticides.",
            "variable-rate application": "The technology of adjusting the quantity of an applied material in real time based on location-specific requirements.",
            "terrain following": "A flight mode where the aircraft maintains a constant height above ground level by adjusting altitude based on terrain elevation changes.",
            "crop health index": "A computed metric derived from spectral reflectance values (e.g., NDVI) that indicates plant vigor and stress levels.",
        },
    },
]

PRIOR_ART_DATE_POOL = [
    "2018-03-15", "2018-07-22", "2018-11-04", "2019-02-18", "2019-05-30",
    "2019-09-12", "2019-12-01", "2020-04-20", "2020-08-14", "2020-11-28",
    "2021-01-10", "2021-06-05", "2021-09-22", "2022-01-15", "2022-05-08",
]

APPLICANT_POOL = [
    "TechForward Inc.", "InnoSystems LLC", "QuantumLeap Labs",
    "NexGen Innovations Corp.", "Vertex AI Solutions",
    "Frontier Computing Group", "Apex Digital Technologies",
    "ClearPath Systems Inc.", "Synapse Dynamics LLC",
    "Helios Research Corporation",
]

EXAMINER_TITLES = [
    "Patent Examiner, Art Unit 2121",
    "Patent Examiner, Art Unit 2435",
    "Patent Examiner, Art Unit 3621",
    "Patent Examiner, Art Unit 2612",
    "Patent Examiner, Art Unit 1765",
]


def make_patent_prior_art(rand_seed: int = 42) -> RubricDatapoint:
    """Analyze a patent application against prior art references. Identify
    anticipated claims, obvious combinations, and genuinely novel claims.

    Seed varies: tech domain, claim structure, which claims are anticipated
    vs obvious vs novel, prior art reference content, near-miss references.
    """
    rng = _random.Random(rand_seed)

    examiner_name = random_name(rand_seed)
    examiner_title = rng.choice(EXAMINER_TITLES)
    applicant = rng.choice(APPLICANT_POOL)
    app_number = f"US {rng.randint(2023, 2025)}/{rng.randint(100000, 999999):06d}"
    filing_date = f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"

    domain = rng.choice(TECH_DOMAINS)
    features = list(domain["feature_pool"])
    rng.shuffle(features)

    # --- Build claims ---
    n_claims = rng.randint(8, 12)

    # Partition claims: anticipated (2-4), obvious (1-2), novel (rest, 2-3+)
    n_anticipated = rng.randint(2, min(4, n_claims - 3))
    n_obvious = rng.randint(1, 2)
    n_novel = n_claims - n_anticipated - n_obvious
    if n_novel < 2:
        n_novel = 2
        n_anticipated = n_claims - n_novel - n_obvious

    # Assign features to claims
    # Each claim gets 1-2 features from the pool
    claim_features: list[list[str]] = []
    feat_idx = 0
    for _ in range(n_claims):
        n_feat = rng.randint(1, 2)
        cf = features[feat_idx:feat_idx + n_feat]
        feat_idx += n_feat
        if feat_idx >= len(features):
            feat_idx = 0
            rng.shuffle(features)
        claim_features.append(cf)

    # Determine claim types: "independent" or "dependent"
    claim_types: list[str] = []
    claim_depends_on: list[int | None] = []
    # First claim is always independent
    claim_types.append("independent")
    claim_depends_on.append(None)
    last_independent = 0
    for i in range(1, n_claims):
        if rng.random() < 0.35 and i > 2:
            claim_types.append("independent")
            claim_depends_on.append(None)
            last_independent = i
        else:
            claim_types.append("dependent")
            claim_depends_on.append(last_independent + 1)  # 1-indexed

    # Assign claim statuses
    claim_indices = list(range(n_claims))
    rng.shuffle(claim_indices)
    anticipated_indices = sorted(claim_indices[:n_anticipated])
    obvious_indices = sorted(claim_indices[n_anticipated:n_anticipated + n_obvious])
    novel_indices = sorted(claim_indices[n_anticipated + n_obvious:])

    # Build claim status map
    claim_status: list[str] = [""] * n_claims
    for i in anticipated_indices:
        claim_status[i] = "anticipated"
    for i in obvious_indices:
        claim_status[i] = "obvious"
    for i in novel_indices:
        claim_status[i] = "novel"

    # --- Build prior art references ---
    n_refs = rng.randint(10, 15)
    available_dates = list(PRIOR_ART_DATE_POOL)
    rng.shuffle(available_dates)

    prior_art_refs: list[dict] = []
    # Assign anticipating references to anticipated claims
    anticipation_map: dict[int, int] = {}  # claim_idx -> ref_idx
    for claim_idx in anticipated_indices:
        ref_idx = len(prior_art_refs)
        anticipation_map[claim_idx] = ref_idx
        ref_features = list(claim_features[claim_idx])
        # The reference teaches all features of the claim
        ref_date = available_dates[ref_idx % len(available_dates)]
        inventor = random_name(rand_seed + ref_idx + 100)
        prior_art_refs.append({
            "ref_id": f"REF-{ref_idx + 1:03d}",
            "title": f"{inventor} et al., '{domain['field']} Implementation'",
            "date": ref_date,
            "inventor": inventor,
            "abstract": f"{domain['abstract_prefix']} {' and '.join(ref_features)}. "
                        f"The system provides improved performance through the combination of "
                        f"these techniques in a unified framework.",
            "key_features": ref_features,
            "role": "anticipates",
            "target_claims": [claim_idx],
        })

    # Assign combination references for obvious claims
    obviousness_map: dict[int, list[int]] = {}  # claim_idx -> list of ref_idxs
    for claim_idx in obvious_indices:
        ref_idxs = []
        cf = claim_features[claim_idx]
        for feat_part_idx, feat in enumerate(cf):
            ref_idx = len(prior_art_refs)
            ref_date = available_dates[ref_idx % len(available_dates)]
            inventor = random_name(rand_seed + ref_idx + 200)
            prior_art_refs.append({
                "ref_id": f"REF-{ref_idx + 1:03d}",
                "title": f"{inventor} et al., 'Advances in {domain['field']}'",
                "date": ref_date,
                "inventor": inventor,
                "abstract": f"{domain['abstract_prefix']} {feat}. "
                            f"The disclosed technique achieves notable results in controlled experiments.",
                "key_features": [feat],
                "role": "obvious_component",
                "target_claims": [claim_idx],
            })
            ref_idxs.append(ref_idx)
        # If claim only had 1 feature, add a second reference teaching a related but
        # not identical concept to motivate the combination
        if len(ref_idxs) < 2:
            ref_idx = len(prior_art_refs)
            related_feat = rng.choice([f for f in features if f not in cf])
            ref_date = available_dates[ref_idx % len(available_dates)]
            inventor = random_name(rand_seed + ref_idx + 300)
            prior_art_refs.append({
                "ref_id": f"REF-{ref_idx + 1:03d}",
                "title": f"{inventor} et al., 'Improvements in {domain['field']}'",
                "date": ref_date,
                "inventor": inventor,
                "abstract": f"{domain['abstract_prefix']} {related_feat}, "
                            f"with discussion of combining it with {cf[0]} for enhanced results.",
                "key_features": [related_feat, cf[0]],
                "role": "obvious_component",
                "target_claims": [claim_idx],
            })
            ref_idxs.append(ref_idx)
        obviousness_map[claim_idx] = ref_idxs

    # Add near-miss references (share terminology but don't actually anticipate)
    n_near_miss = rng.randint(2, 4)
    near_miss_refs: list[int] = []
    for nm in range(n_near_miss):
        ref_idx = len(prior_art_refs)
        near_miss_refs.append(ref_idx)
        # Pick features from novel claims but twist the description
        if novel_indices:
            target_novel = rng.choice(novel_indices)
            nm_features = claim_features[target_novel]
        else:
            nm_features = [rng.choice(features)]
        ref_date = available_dates[ref_idx % len(available_dates)]
        inventor = random_name(rand_seed + ref_idx + 400)
        # Near-miss: uses same terms but in a different context/application
        twist_phrases = [
            "in the context of offline batch processing rather than real-time systems",
            "applied to a fundamentally different domain without the disclosed integration",
            "as a theoretical framework without practical implementation details",
            "limited to single-instance deployment without the scalability aspects",
            "using a substantially different underlying mathematical formulation",
        ]
        twist = rng.choice(twist_phrases)
        prior_art_refs.append({
            "ref_id": f"REF-{ref_idx + 1:03d}",
            "title": f"{inventor} et al., 'Studies in {domain['field']}'",
            "date": ref_date,
            "inventor": inventor,
            "abstract": f"A study of {' and '.join(nm_features)}, "
                        f"{twist}. The work does not address the specific "
                        f"combination or application disclosed in the present application.",
            "key_features": nm_features,
            "role": "near_miss",
            "target_claims": [],
        })

    # Fill remaining references to reach n_refs
    while len(prior_art_refs) < n_refs:
        ref_idx = len(prior_art_refs)
        ref_date = available_dates[ref_idx % len(available_dates)]
        inventor = random_name(rand_seed + ref_idx + 500)
        filler_feat = rng.choice(features)
        prior_art_refs.append({
            "ref_id": f"REF-{ref_idx + 1:03d}",
            "title": f"{inventor} et al., 'General {domain['field']} Methods'",
            "date": ref_date,
            "inventor": inventor,
            "abstract": f"A general discussion of {filler_feat} in the field of {domain['field'].lower()}.",
            "key_features": [filler_feat],
            "role": "background",
            "target_claims": [],
        })

    # --- Build patent_application.txt ---
    app_lines = [
        f"PATENT APPLICATION",
        f"",
        f"Application Number: {app_number}",
        f"Filing Date: {filing_date}",
        f"Applicant: {applicant}",
        f"Title: System and Method for {domain['field']}",
        f"",
        "=" * 65,
        "ABSTRACT",
        "=" * 65,
        "",
        f"{domain['abstract_prefix']} a novel combination of techniques including "
        f"{', '.join(features[:4])}, and related methods. The invention provides "
        f"significant improvements over existing approaches in the field.",
        "",
        "=" * 65,
        "CLAIMS",
        "=" * 65,
        "",
    ]
    for i in range(n_claims):
        claim_num = i + 1
        if claim_types[i] == "independent":
            app_lines.append(f"  {claim_num}. A method for {domain['field'].lower()} comprising:")
            for feat in claim_features[i]:
                app_lines.append(f"       (a) {feat};")
            app_lines.append(f"       wherein said method achieves improved performance.")
        else:
            dep = claim_depends_on[i]
            app_lines.append(f"  {claim_num}. The method of claim {dep}, further comprising:")
            for feat in claim_features[i]:
                app_lines.append(f"       {feat}.")
        app_lines.append("")
    app_content = "\n".join(app_lines) + "\n"

    # --- Build prior_art_references.csv ---
    csv_lines = ["ref_id,title,date,abstract,key_features"]
    for ref in prior_art_refs:
        feats_str = "; ".join(ref["key_features"])
        # Escape commas in fields
        abstract_esc = ref["abstract"].replace('"', '""')
        title_esc = ref["title"].replace('"', '""')
        csv_lines.append(
            f'{ref["ref_id"]},"{title_esc}",{ref["date"]},"{abstract_esc}","{feats_str}"'
        )
    csv_content = "\n".join(csv_lines) + "\n"

    # --- Build claim_construction.txt ---
    cc_lines = [
        "CLAIM CONSTRUCTION — KEY TERM DEFINITIONS",
        "",
        f"Application Number: {app_number}",
        "",
        "The following terms used in the claims shall be construed as follows:",
        "",
        "=" * 65,
    ]
    for term, definition in domain["key_terms"].items():
        cc_lines.append(f"")
        cc_lines.append(f'TERM: "{term}"')
        cc_lines.append(f"  Definition: {definition}")
    cc_lines.append("")
    cc_content = "\n".join(cc_lines) + "\n"

    # --- Build examination_guidelines.txt ---
    eg_lines = [
        "PATENT EXAMINATION GUIDELINES",
        "",
        "=" * 65,
        "35 U.S.C. 102 — NOVELTY (Anticipation)",
        "=" * 65,
        "",
        "A claim is anticipated (lacks novelty) under Section 102 if a single",
        "prior art reference discloses EVERY element of the claim, either",
        "explicitly or inherently. The reference must be prior art (published",
        "before the effective filing date of the application).",
        "",
        "Key principles:",
        "  - All claim elements must be found in ONE reference.",
        "  - The reference must enable one of ordinary skill to practice the",
        "    claimed invention.",
        "  - Differences in terminology alone do not avoid anticipation if the",
        "    reference teaches the same concept.",
        "  - A reference that teaches a broader genus anticipates a species",
        "    only if the species is specifically disclosed.",
        "",
        "=" * 65,
        "35 U.S.C. 103 — OBVIOUSNESS",
        "=" * 65,
        "",
        "A claim is obvious under Section 103 if the differences between the",
        "claimed invention and the prior art are such that the invention as a",
        "whole would have been obvious to a person having ordinary skill in",
        "the art at the time the invention was made.",
        "",
        "Key principles:",
        "  - Unlike Section 102, Section 103 allows COMBINING multiple references.",
        "  - The examiner must identify: (1) the scope of the prior art,",
        "    (2) the differences between the art and the claims, and (3) a",
        "    motivation or reason to combine the references.",
        "  - Teaching, suggestion, or motivation (TSM) to combine can come from",
        "    the references themselves, the knowledge of one skilled in the art,",
        "    or the nature of the problem being solved.",
        "  - Mere proximity of features in the same technical field is NOT",
        "    sufficient motivation to combine.",
        "",
        "=" * 65,
        "CLAIM CONSTRUCTION",
        "=" * 65,
        "",
        "Claims must be given their broadest reasonable interpretation consistent",
        "with the specification. Dependent claims incorporate all limitations of",
        "the claims from which they depend.",
        "",
        "=" * 65,
        "ANALYSIS FORMAT",
        "=" * 65,
        "",
        "For each claim, the examiner should:",
        "  1. Identify the claim elements",
        "  2. Search for each element in the prior art",
        "  3. Determine if a single reference anticipates (102) or if a",
        "     combination renders obvious (103)",
        "  4. If novel and non-obvious, explain why no reference or combination",
        "     of references teaches the claimed invention",
        "",
    ]
    eg_content = "\n".join(eg_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Patent Prior Art Analysis

You are {examiner_name}, {examiner_title}. You have been assigned to examine
patent application {app_number} filed by {applicant} in the field of
{domain['field']}.

## Source Files
- /testbed/data/patent_application.txt — The patent application with {n_claims} claims
- /testbed/data/prior_art_references.csv — {len(prior_art_refs)} prior art references with titles, dates, abstracts, and key features
- /testbed/data/claim_construction.txt — Definitions of key terms used in the claims
- /testbed/data/examination_guidelines.txt — Rules for novelty (Section 102) and obviousness (Section 103) analysis

## Requirements
1. Parse and understand all {n_claims} claims, noting which are independent and which are dependent
2. For each claim, identify the key elements/features
3. Map claim elements against prior art reference features
4. For each claim, determine whether it is:
   - Anticipated under 35 U.S.C. 102 (a single reference teaches all elements)
   - Obvious under 35 U.S.C. 103 (a combination of references renders it obvious)
   - Novel and non-obvious (no reference or combination invalidates it)
5. Cite specific reference IDs for any rejection
6. Be careful of near-miss references that share terminology but do not actually teach the claimed elements

Write your examination report to /testbed/examination_report.txt"""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/examination_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_claim_count",
            question=f"Does the report correctly identify that the application contains {n_claims} claims?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_independent_dependent",
            question=(
                f"Does the report distinguish between independent and dependent claims? "
                f"There are {sum(1 for t in claim_types if t == 'independent')} independent "
                f"and {sum(1 for t in claim_types if t == 'dependent')} dependent claims."
            ),
            points=1,
        ),
    ]

    # Per-anticipated-claim checks
    for claim_idx in anticipated_indices:
        claim_num = claim_idx + 1
        ref_idx = anticipation_map[claim_idx]
        ref = prior_art_refs[ref_idx]
        rubric_items.append(
            BinaryRubricCategory(
                name=f"claim_{claim_num}_anticipated",
                question=(
                    f"Does the report correctly identify Claim {claim_num} as anticipated "
                    f"(rejected under 35 U.S.C. 102) by a prior art reference?"
                ),
                points=2,
            )
        )
        rubric_items.append(
            BinaryRubricCategory(
                name=f"claim_{claim_num}_cites_correct_ref",
                question=(
                    f"Does the report cite {ref['ref_id']} (or a reference teaching "
                    f"'{', '.join(ref['key_features'])}') as the basis for rejecting "
                    f"Claim {claim_num}?"
                ),
                points=2,
            )
        )

    # Per-obvious-claim checks
    for claim_idx in obvious_indices:
        claim_num = claim_idx + 1
        ref_idxs = obviousness_map[claim_idx]
        ref_ids = [prior_art_refs[ri]["ref_id"] for ri in ref_idxs]
        rubric_items.append(
            BinaryRubricCategory(
                name=f"claim_{claim_num}_obvious",
                question=(
                    f"Does the report correctly identify Claim {claim_num} as obvious "
                    f"(rejected under 35 U.S.C. 103 as a combination of references)?"
                ),
                points=2,
            )
        )
        rubric_items.append(
            BinaryRubricCategory(
                name=f"claim_{claim_num}_cites_combination",
                question=(
                    f"Does the report cite at least two references (e.g., "
                    f"{', '.join(ref_ids[:2])}) as the basis for the obviousness "
                    f"rejection of Claim {claim_num}?"
                ),
                points=2,
            )
        )

    # Per-novel-claim checks
    for claim_idx in novel_indices:
        claim_num = claim_idx + 1
        rubric_items.append(
            BinaryRubricCategory(
                name=f"claim_{claim_num}_novel",
                question=(
                    f"Does the report correctly identify Claim {claim_num} as novel and "
                    f"non-obvious (not rejected under either 102 or 103)?"
                ),
                points=2,
            )
        )

    # Near-miss false positive checks
    for nm_ref_idx in near_miss_refs[:3]:
        ref = prior_art_refs[nm_ref_idx]
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_anticipation_{ref['ref_id'].lower().replace('-', '_')}",
                question=(
                    f"Does the report correctly avoid using {ref['ref_id']} "
                    f"('{ref['title'][:60]}...') as the sole basis for anticipating "
                    f"any claim? This reference shares terminology but does not actually "
                    f"teach the claimed elements."
                ),
                points=2,
            )
        )

    # Summary checks
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_anticipated_count",
            question=(
                f"Does the report identify exactly {n_anticipated} claim(s) as anticipated "
                f"under Section 102?"
            ),
            points=2,
        )
    )
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_obvious_count",
            question=(
                f"Does the report identify exactly {n_obvious} claim(s) as obvious "
                f"under Section 103?"
            ),
            points=2,
        )
    )
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_novel_count",
            question=(
                f"Does the report identify exactly {n_novel} claim(s) as novel and "
                f"non-obvious?"
            ),
            points=2,
        )
    )

    rubric_items.append(
        RubricCategory(
            name="analysis_thoroughness",
            description="How thorough and systematic is the patent examination?",
            failure="Superficial analysis; most claims not individually analyzed or citations missing.",
            minor_failure="Some claims analyzed but analysis lacks element-by-element mapping to prior art.",
            minor_success="Most claims analyzed with element mapping; minor gaps in citation specificity.",
            success="Every claim analyzed with element-by-element mapping to specific prior art references, clear 102/103 reasoning, and identification of near-miss references.",
            points=3,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed examination report to /testbed/examination_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/patent_application.txt": app_content,
            "/testbed/data/prior_art_references.csv": csv_content,
            "/testbed/data/claim_construction.txt": cc_content,
            "/testbed/data/examination_guidelines.txt": eg_content,
        },
        problem_type="patent_prior_art",
    )


# =============================================================================
# DOMAIN: CONTRACT RISK ANALYSIS
# =============================================================================

CONTRACT_TYPES = [
    {
        "key": "msa",
        "label": "Master Services Agreement",
        "parties_template": "{provider} (\"Provider\") and {customer} (\"Customer\")",
        "preamble": "This Master Services Agreement sets forth the terms under which Provider shall deliver professional services to Customer.",
        "section_pool": [
            "Scope of Services", "Term and Renewal", "Fees and Payment",
            "Intellectual Property", "Confidentiality", "Data Protection",
            "Representations and Warranties", "Limitation of Liability",
            "Indemnification", "Termination", "Force Majeure",
            "Governing Law", "Dispute Resolution", "General Provisions",
        ],
    },
    {
        "key": "saas",
        "label": "SaaS Subscription Agreement",
        "parties_template": "{provider} (\"Vendor\") and {customer} (\"Subscriber\")",
        "preamble": "This SaaS Subscription Agreement governs Subscriber's access to and use of Vendor's cloud-based software platform.",
        "section_pool": [
            "Grant of License", "Service Level Agreement", "Term and Renewal",
            "Fees and Payment", "Data Ownership and Processing", "Security",
            "Intellectual Property", "Warranties", "Limitation of Liability",
            "Indemnification", "Termination", "Acceptable Use", "Audit Rights",
            "Governing Law", "General Provisions",
        ],
    },
    {
        "key": "supply",
        "label": "Supply Agreement",
        "parties_template": "{provider} (\"Supplier\") and {customer} (\"Buyer\")",
        "preamble": "This Supply Agreement establishes the terms for Supplier's provision of goods and related services to Buyer.",
        "section_pool": [
            "Product Specifications", "Ordering and Delivery", "Pricing and Payment",
            "Quality Standards", "Inspection and Acceptance", "Warranties",
            "Intellectual Property", "Limitation of Liability", "Indemnification",
            "Term and Termination", "Force Majeure", "Insurance",
            "Governing Law", "Confidentiality", "General Provisions",
        ],
    },
    {
        "key": "consulting",
        "label": "Consulting Services Agreement",
        "parties_template": "{provider} (\"Consultant\") and {customer} (\"Client\")",
        "preamble": "This Consulting Services Agreement governs the engagement of Consultant to provide advisory and professional services to Client.",
        "section_pool": [
            "Scope of Engagement", "Deliverables", "Fees and Expenses",
            "Term and Termination", "Intellectual Property", "Confidentiality",
            "Non-Solicitation", "Non-Compete", "Representations and Warranties",
            "Limitation of Liability", "Indemnification", "Insurance",
            "Governing Law", "Dispute Resolution", "General Provisions",
        ],
    },
    {
        "key": "license",
        "label": "Software License Agreement",
        "parties_template": "{provider} (\"Licensor\") and {customer} (\"Licensee\")",
        "preamble": "This Software License Agreement grants Licensee certain rights to use Licensor's proprietary software under the terms set forth herein.",
        "section_pool": [
            "Grant of License", "License Restrictions", "Fees and Payment",
            "Maintenance and Support", "Intellectual Property Rights",
            "Confidentiality", "Warranties", "Limitation of Liability",
            "Indemnification", "Term and Termination", "Export Compliance",
            "Audit Rights", "Governing Law", "General Provisions",
        ],
    },
    {
        "key": "distribution",
        "label": "Distribution Agreement",
        "parties_template": "{provider} (\"Manufacturer\") and {customer} (\"Distributor\")",
        "preamble": "This Distribution Agreement appoints Distributor as an authorized channel for the marketing and sale of Manufacturer's products in the designated territory.",
        "section_pool": [
            "Appointment and Territory", "Products and Pricing", "Ordering and Delivery",
            "Minimum Purchase Requirements", "Marketing and Promotion",
            "Intellectual Property", "Confidentiality", "Warranties",
            "Limitation of Liability", "Indemnification", "Term and Termination",
            "Non-Compete", "Insurance", "Governing Law", "General Provisions",
        ],
    },
    {
        "key": "data_processing",
        "label": "Data Processing Agreement",
        "parties_template": "{provider} (\"Processor\") and {customer} (\"Controller\")",
        "preamble": "This Data Processing Agreement establishes the obligations of Processor with respect to personal data processed on behalf of Controller.",
        "section_pool": [
            "Definitions", "Scope of Processing", "Processor Obligations",
            "Sub-processors", "Data Subject Rights", "Security Measures",
            "Data Breach Notification", "Data Transfer Mechanisms",
            "Audits and Inspections", "Term and Termination",
            "Data Return and Deletion", "Liability", "Governing Law",
            "General Provisions",
        ],
    },
    {
        "key": "joint_venture",
        "label": "Joint Venture Agreement",
        "parties_template": "{provider} (\"Party A\") and {customer} (\"Party B\")",
        "preamble": "This Joint Venture Agreement establishes a collaborative business arrangement between the Parties for the purpose of pursuing mutual commercial objectives.",
        "section_pool": [
            "Purpose and Scope", "Contributions", "Management and Governance",
            "Profit and Loss Sharing", "Intellectual Property",
            "Confidentiality", "Non-Compete", "Representations and Warranties",
            "Limitation of Liability", "Indemnification", "Term and Termination",
            "Dispute Resolution", "Exit Provisions", "Governing Law",
            "General Provisions",
        ],
    },
]

# Risk issue templates: each is a function of contract context
RISK_ISSUES = [
    {
        "key": "uncapped_liability",
        "label": "Uncapped liability exposure",
        "severity": "High",
        "section": "Limitation of Liability",
        "clause_template": (
            "To the maximum extent permitted by law, {provider_role}'s total aggregate "
            "liability under this Agreement shall not be limited and shall include all "
            "direct, indirect, consequential, incidental, and special damages."
        ),
        "risk_description": "Liability is entirely uncapped, exposing the company to unlimited financial risk.",
    },
    {
        "key": "missing_data_protection",
        "label": "Missing data protection clause",
        "severity": "High",
        "section": "Data Protection",
        "clause_template": None,  # This risk is the ABSENCE of a clause
        "risk_description": "No data protection or privacy clause exists despite the agreement involving processing of personal data.",
    },
    {
        "key": "auto_renewal_trap",
        "label": "Auto-renewal with excessive notice period",
        "severity": "Medium",
        "section": "Term and Renewal",
        "clause_template": (
            "This Agreement shall automatically renew for successive {term_years}-year "
            "terms unless either party provides written notice of non-renewal at least "
            "{notice_days} days prior to the end of the then-current term."
        ),
        "risk_description": "Auto-renewal notice period is excessive, creating a practical lock-in effect.",
    },
    {
        "key": "one_sided_termination",
        "label": "One-sided termination rights",
        "severity": "High",
        "section": "Termination",
        "clause_template": (
            "{provider_role} may terminate this Agreement at any time for convenience "
            "upon thirty (30) days written notice. {customer_role} may terminate only for "
            "material breach by {provider_role}, and only after providing written notice "
            "and a cure period of ninety (90) days."
        ),
        "risk_description": "Termination rights are heavily asymmetric; counterparty can exit at will while we are locked in.",
    },
    {
        "key": "broad_ip_assignment",
        "label": "Overly broad IP assignment",
        "severity": "High",
        "section": "Intellectual Property",
        "clause_template": (
            "{customer_role} hereby assigns to {provider_role} all right, title, and "
            "interest in and to any and all intellectual property created, developed, "
            "conceived, or reduced to practice during the term of this Agreement, "
            "including but not limited to inventions, works of authorship, trade secrets, "
            "and improvements thereto, whether or not related to the services performed "
            "under this Agreement."
        ),
        "risk_description": "IP assignment extends to all IP created during the term, not just deliverables, capturing unrelated innovations.",
    },
    {
        "key": "inadequate_warranty",
        "label": "Inadequate warranty disclaimer",
        "severity": "Medium",
        "section": "Warranties",
        "clause_template": (
            "THE SERVICES AND DELIVERABLES ARE PROVIDED \"AS IS\" WITHOUT WARRANTY OF "
            "ANY KIND, WHETHER EXPRESS, IMPLIED, OR STATUTORY, INCLUDING WITHOUT "
            "LIMITATION ANY WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR "
            "PURPOSE, OR NON-INFRINGEMENT."
        ),
        "risk_description": "Complete warranty disclaimer leaves no recourse if deliverables are defective or unfit for purpose.",
    },
    {
        "key": "missing_force_majeure",
        "label": "Missing force majeure clause",
        "severity": "Medium",
        "section": "Force Majeure",
        "clause_template": None,  # Absence of clause
        "risk_description": "No force majeure provision; parties remain obligated to perform even during events beyond their control.",
    },
    {
        "key": "excessive_noncompete",
        "label": "Non-compete period exceeding legal limits",
        "severity": "High",
        "section": "Non-Compete",
        "clause_template": (
            "{customer_role} shall not, for a period of {noncompete_years} years "
            "following termination of this Agreement, directly or indirectly engage "
            "in any business that competes with {provider_role}'s business anywhere "
            "in the world."
        ),
        "risk_description": "Non-compete duration and geographic scope exceed enforceable limits in most jurisdictions.",
    },
    {
        "key": "below_market_sla",
        "label": "Below-market SLA targets",
        "severity": "Medium",
        "section": "Service Level Agreement",
        "clause_template": (
            "{provider_role} shall use commercially reasonable efforts to maintain "
            "service availability of {sla_pct}% measured on a monthly basis. Service "
            "credits shall be limited to {credit_pct}% of monthly fees for any month "
            "in which the target is not met."
        ),
        "risk_description": "SLA target is significantly below industry standard and service credit remedy is minimal.",
    },
    {
        "key": "missing_audit_rights",
        "label": "Missing audit rights",
        "severity": "Medium",
        "section": "Audit Rights",
        "clause_template": None,  # Absence of clause
        "risk_description": "No audit rights provision; unable to verify compliance with security, data handling, or financial obligations.",
    },
]

# Clauses that look aggressive but are within market benchmarks (false positives)
FALSE_POSITIVE_CLAUSES = [
    {
        "key": "standard_liability_cap",
        "section": "Limitation of Liability",
        "clause_template": (
            "Except for obligations under the Indemnification section, each party's "
            "total aggregate liability shall not exceed the greater of (a) the total "
            "fees paid or payable in the twelve (12) months preceding the claim, or "
            "(b) {liability_cap}. Neither party shall be liable for indirect, "
            "consequential, incidental, or special damages except in cases of gross "
            "negligence or willful misconduct."
        ),
        "explanation": "Liability cap of 12 months' fees with carve-outs for gross negligence is standard market practice.",
    },
    {
        "key": "standard_indemnification",
        "section": "Indemnification",
        "clause_template": (
            "Each party shall indemnify, defend, and hold harmless the other party from "
            "and against any third-party claims arising from: (a) the indemnifying party's "
            "material breach of this Agreement, (b) the indemnifying party's negligence "
            "or willful misconduct, or (c) the indemnifying party's violation of applicable "
            "law. The indemnified party shall provide prompt written notice and reasonable "
            "cooperation."
        ),
        "explanation": "Mutual indemnification with standard triggers and cooperation requirements is within market norms.",
    },
    {
        "key": "standard_termination_notice",
        "section": "Termination",
        "clause_template": (
            "Either party may terminate this Agreement for convenience upon ninety (90) "
            "days prior written notice. Upon termination for convenience by {customer_role}, "
            "{customer_role} shall pay all fees for services rendered through the effective "
            "date of termination."
        ),
        "explanation": "Mutual 90-day termination for convenience with payment for services rendered is standard practice.",
    },
    {
        "key": "standard_ip_ownership",
        "section": "Intellectual Property",
        "clause_template": (
            "{customer_role} shall own all deliverables created specifically for "
            "{customer_role} under this Agreement. {provider_role} retains ownership of "
            "all pre-existing intellectual property, tools, methodologies, and general "
            "know-how. {provider_role} grants {customer_role} a perpetual, non-exclusive "
            "license to use any {provider_role} pre-existing IP embedded in the deliverables."
        ),
        "explanation": "Customer owns bespoke deliverables; provider retains pre-existing IP with license grant. This is industry standard.",
    },
    {
        "key": "standard_confidentiality",
        "section": "Confidentiality",
        "clause_template": (
            "Each party agrees to maintain the confidentiality of the other party's "
            "Confidential Information for a period of three (3) years following disclosure, "
            "using the same degree of care it uses to protect its own confidential "
            "information, but in no event less than reasonable care. Standard exceptions "
            "apply: publicly known information, independently developed information, and "
            "information received from a third party without restriction."
        ),
        "explanation": "Three-year confidentiality obligation with standard exceptions is well within market norms.",
    },
]


def make_contract_risk_analysis(rand_seed: int = 42) -> RubricDatapoint:
    """Analyze a commercial contract for risk clauses. Cross-reference against
    risk matrix, market benchmarks, and regulatory requirements.

    Seed varies: contract type, which risk issues are planted (4-7), specific
    clause parameters, false positive clauses, party names.
    """
    rng = _random.Random(rand_seed)

    analyst_name = random_name(rand_seed)
    provider_company = pick1(COMPANY_NAMES, rand_seed)
    customer_company = pick1(COMPANY_NAMES, rand_seed + 1)
    # Make sure they're different
    while customer_company == provider_company:
        customer_company = rng.choice(COMPANY_NAMES)

    contract_type = rng.choice(CONTRACT_TYPES)
    # Extract role names (e.g. "Provider", "Customer") from the template
    role_parts = contract_type["parties_template"].split('"')
    provider_role = role_parts[1] if len(role_parts) > 1 else "Provider"
    customer_role = role_parts[3] if len(role_parts) > 3 else "Customer"

    contract_date = f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    contract_number = f"AGR-{rng.randint(10000, 99999)}"

    # Contract parameters for clause templates
    term_years = rng.choice([1, 2, 3])
    notice_days = rng.choice([120, 180, 270])  # Excessive notice periods
    noncompete_years = rng.choice([3, 4, 5])  # Excessive
    sla_pct = rng.choice([95.0, 96.0, 97.0])  # Below market (market is 99.5+)
    credit_pct = rng.choice([2, 3, 5])  # Low credits
    liability_cap = _fmt_money(rng.choice([500000, 750000, 1000000]))
    annual_fees = rng.choice([250000, 500000, 750000, 1000000])

    # Select 4-7 risk issues to plant
    n_risks = rng.randint(4, 7)
    available_risks = list(RISK_ISSUES)
    rng.shuffle(available_risks)
    planted_risks = available_risks[:n_risks]

    # Select 2-3 false positive clauses
    n_fp = rng.randint(2, 3)
    available_fps = list(FALSE_POSITIVE_CLAUSES)
    rng.shuffle(available_fps)
    planted_fps = available_fps[:n_fp]

    # --- Build contract.txt ---
    contract_lines = [
        f"{contract_type['label'].upper()}",
        "",
        f"Agreement Number: {contract_number}",
        f"Effective Date: {contract_date}",
        "",
        f"BETWEEN: {contract_type['parties_template'].format(provider=provider_company, customer=customer_company)}",
        "",
        f"{contract_type['preamble']}",
        "",
        "=" * 65,
    ]

    # Track which sections have been used by risk clauses
    sections_used_by_risk: dict[str, str] = {}
    sections_used_by_fp: dict[str, str] = {}

    for risk in planted_risks:
        if risk["clause_template"] is not None:
            clause_text = risk["clause_template"].format(
                provider_role=provider_role,
                customer_role=customer_role,
                term_years=term_years,
                notice_days=notice_days,
                noncompete_years=noncompete_years,
                sla_pct=sla_pct,
                credit_pct=credit_pct,
            )
            sections_used_by_risk[risk["section"]] = clause_text

    for fp in planted_fps:
        clause_text = fp["clause_template"].format(
            provider_role=provider_role,
            customer_role=customer_role,
            liability_cap=liability_cap,
        )
        # Don't overwrite risk clause in same section
        if fp["section"] not in sections_used_by_risk:
            sections_used_by_fp[fp["section"]] = clause_text

    # Build sections
    section_num = 1
    for section_name in contract_type["section_pool"]:
        contract_lines.append("")
        contract_lines.append(f"SECTION {section_num}. {section_name.upper()}")
        contract_lines.append("")

        if section_name in sections_used_by_risk:
            contract_lines.append(f"  {section_num}.1 {sections_used_by_risk[section_name]}")
        elif section_name in sections_used_by_fp:
            contract_lines.append(f"  {section_num}.1 {sections_used_by_fp[section_name]}")
        else:
            # Generate generic clause
            generic_clauses = {
                "Scope of Services": f"  {section_num}.1 {provider_role} shall provide the services described in Exhibit A, attached hereto and incorporated by reference.",
                "Fees and Payment": f"  {section_num}.1 {customer_role} shall pay {provider_role} the fees set forth in Exhibit B. Payment is due within thirty (30) days of invoice date. Late payments shall accrue interest at 1.5% per month.",
                "Confidentiality": f"  {section_num}.1 Each party shall maintain the confidentiality of the other party's proprietary information disclosed under this Agreement.",
                "Governing Law": f"  {section_num}.1 This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to conflict of laws principles.",
                "Dispute Resolution": f"  {section_num}.1 Any dispute arising under this Agreement shall first be submitted to mediation. If mediation is unsuccessful within sixty (60) days, either party may pursue binding arbitration under the rules of the American Arbitration Association.",
                "General Provisions": f"  {section_num}.1 This Agreement constitutes the entire agreement between the parties and supersedes all prior negotiations, representations, or agreements.",
            }
            if section_name in generic_clauses:
                contract_lines.append(generic_clauses[section_name])
            else:
                contract_lines.append(f"  {section_num}.1 The parties shall comply with all applicable requirements regarding {section_name.lower()} as set forth in this Agreement and any applicable exhibits.")

        section_num += 1

    # Note: for "missing" risk issues, we intentionally omit the section
    # Check which risk issues are about missing clauses
    missing_sections = [r["section"] for r in planted_risks if r["clause_template"] is None]

    contract_lines.append("")
    contract_lines.append("=" * 65)
    contract_lines.append("")
    contract_lines.append("IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.")
    contract_lines.append("")
    contract_lines.append(f"For {provider_company}:")
    contract_lines.append(f"  Name: {random_name(rand_seed + 10)}")
    contract_lines.append(f"  Title: Chief Executive Officer")
    contract_lines.append(f"  Date: {contract_date}")
    contract_lines.append("")
    contract_lines.append(f"For {customer_company}:")
    contract_lines.append(f"  Name: {random_name(rand_seed + 11)}")
    contract_lines.append(f"  Title: Chief Procurement Officer")
    contract_lines.append(f"  Date: {contract_date}")
    contract_lines.append("")
    contract_content = "\n".join(contract_lines) + "\n"

    # --- Build risk_matrix.txt ---
    risk_matrix_lines = [
        f"RISK TOLERANCE MATRIX — {customer_company}",
        "",
        "This document defines the company's risk tolerance for commercial agreements.",
        "",
        "=" * 65,
        "SEVERITY CLASSIFICATIONS",
        "=" * 65,
        "",
        "  Critical: Potential exposure exceeding $5M or regulatory sanctions.",
        "  High: Potential exposure of $1M-$5M or significant operational disruption.",
        "  Medium: Potential exposure of $250K-$1M or moderate operational impact.",
        "  Low: Potential exposure below $250K with limited operational impact.",
        "",
        "=" * 65,
        "RISK TOLERANCE BY CATEGORY",
        "=" * 65,
        "",
        "  Liability Exposure: Maximum acceptable liability cap is 2x annual fees.",
        f"     Current annual fee level: {_fmt_money(annual_fees)}",
        f"     Maximum acceptable cap: {_fmt_money(annual_fees * 2)}",
        "     Uncapped liability: NOT ACCEPTABLE under any circumstances.",
        "",
        "  Data Protection: Mandatory clause for any agreement involving personal",
        "     data or customer information. Must include breach notification,",
        "     processing limitations, and data deletion upon termination.",
        "",
        "  Auto-Renewal: Maximum acceptable notice period is 60 days.",
        "     Terms exceeding 60 days create unacceptable lock-in risk.",
        "",
        "  Termination: Mutual termination for convenience is required.",
        "     One-sided termination rights are classified as High risk.",
        "",
        "  IP Rights: Company retains ownership of all deliverables created for",
        "     the company. Pre-existing IP of the vendor may be licensed but not",
        "     assigned. Assignment of company's own IP to vendor: NOT ACCEPTABLE.",
        "",
        "  Warranty: Bare 'as-is' disclaimer without any warranty is High risk.",
        "     Minimum: warranty of professional workmanship for 12 months.",
        "",
        "  Force Majeure: Required for agreements with >6 month term.",
        "",
        "  Non-Compete: Maximum enforceable period is 1 year with limited geography.",
        "     Periods exceeding 2 years are likely unenforceable and High risk.",
        "",
        "  SLA: Minimum acceptable uptime is 99.5% for any production service.",
        "     Service credits must be at least 10% of monthly fees per SLA breach.",
        "",
        "  Audit Rights: Required for any agreement involving data processing",
        "     or fees based on usage/consumption.",
        "",
    ]
    risk_matrix_content = "\n".join(risk_matrix_lines) + "\n"

    # --- Build market_benchmarks.txt ---
    benchmarks_lines = [
        "INDUSTRY BENCHMARK DATA — COMMERCIAL AGREEMENTS",
        "",
        "Source: Legal Department Market Analysis (2024 Q2)",
        "",
        "=" * 65,
        "LIMITATION OF LIABILITY",
        "=" * 65,
        "",
        "  Typical cap: 12 months' fees (standard) to 24 months' fees (favorable).",
        "  Carve-outs: Gross negligence, willful misconduct, IP infringement,",
        "    and confidentiality breaches typically excluded from cap.",
        "  Consequential damages: Usually excluded except for carve-outs.",
        "",
        "=" * 65,
        "SERVICE LEVEL AGREEMENTS",
        "=" * 65,
        "",
        "  Industry standard uptime: 99.9% (three nines) for production SaaS.",
        "  Acceptable minimum: 99.5% for non-mission-critical services.",
        "  Service credits: 10-25% of monthly fees per SLA tier missed.",
        "  Below 99.0%: Considered substandard and grounds for termination.",
        "",
        "=" * 65,
        "TERMINATION PROVISIONS",
        "=" * 65,
        "",
        "  Mutual termination for convenience: Standard (30-90 day notice).",
        "  Termination for cause: 30-day cure period is standard.",
        "  Auto-renewal notice: 30-60 days is standard; >90 days is unusual.",
        "",
        "=" * 65,
        "INDEMNIFICATION",
        "=" * 65,
        "",
        "  Mutual indemnification is market standard.",
        "  Triggers: material breach, negligence, IP infringement, law violations.",
        "  Cooperation requirement: Standard (prompt notice + reasonable cooperation).",
        "",
        "=" * 65,
        "INTELLECTUAL PROPERTY",
        "=" * 65,
        "",
        "  Customer owns bespoke deliverables: Standard.",
        "  Vendor retains pre-existing IP with license to customer: Standard.",
        "  Broad assignment of all IP created during term: Non-standard / aggressive.",
        "",
        "=" * 65,
        "NON-COMPETE",
        "=" * 65,
        "",
        "  Enforceable duration: 6-12 months in most jurisdictions.",
        "  Geographic scope: Must be reasonably limited.",
        "  Worldwide non-compete >2 years: Generally unenforceable.",
        "",
        "=" * 65,
        "DATA PROTECTION",
        "=" * 65,
        "",
        "  Mandatory for any agreement involving personal data (GDPR/CCPA).",
        "  Standard elements: processing limitations, data breach notification",
        "    (72 hours), sub-processor restrictions, data deletion on termination.",
        "",
        "=" * 65,
        "CONFIDENTIALITY",
        "=" * 65,
        "",
        "  Duration: 2-5 years is standard; 3 years most common.",
        "  Standard exceptions: public knowledge, independent development,",
        "    third-party disclosure without restriction, legal compulsion.",
        "",
    ]
    benchmarks_content = "\n".join(benchmarks_lines) + "\n"

    # --- Build regulatory_requirements.txt ---
    reg_lines = [
        "MANDATORY REGULATORY REQUIREMENTS FOR COMMERCIAL AGREEMENTS",
        "",
        "This document lists clauses required by applicable regulations.",
        "",
        "=" * 65,
        "DATA PROTECTION REGULATIONS (GDPR / CCPA)",
        "=" * 65,
        "",
        "  If agreement involves processing personal data of EU/California residents:",
        "  - Data Processing Agreement (DPA) or equivalent must be included",
        "  - Data breach notification within 72 hours of discovery",
        "  - Data subject access request procedures",
        "  - Sub-processor approval mechanisms",
        "  - Data deletion/return upon termination",
        "  - Cross-border transfer safeguards (Standard Contractual Clauses)",
        "",
        "=" * 65,
        "FINANCIAL SERVICES REGULATIONS",
        "=" * 65,
        "",
        "  For agreements with financial service providers:",
        "  - Audit and examination rights for regulatory authorities",
        "  - Business continuity and disaster recovery requirements",
        "  - Data localization requirements where applicable",
        "  - Subcontracting restrictions and notification requirements",
        "",
        "=" * 65,
        "ANTI-CORRUPTION / EXPORT CONTROL",
        "=" * 65,
        "",
        "  All agreements must include:",
        "  - Anti-corruption representation and warranty",
        "  - Export control compliance commitment",
        "  - Sanctions screening obligations",
        "",
        "=" * 65,
        "INSURANCE REQUIREMENTS",
        "=" * 65,
        "",
        "  Vendors handling company data or performing on-site services must maintain:",
        "  - Commercial general liability: $1M per occurrence / $2M aggregate",
        "  - Professional liability (E&O): $2M per occurrence",
        "  - Cyber liability: $5M per occurrence",
        "",
    ]
    reg_content = "\n".join(reg_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Contract Risk Analysis

You are {analyst_name}, a senior legal analyst at {customer_company}. You have been
asked to review a proposed {contract_type['label']} with {provider_company} and
identify any risk issues, comparing terms against your company's risk tolerance,
industry benchmarks, and regulatory requirements.

## Source Files
- /testbed/data/contract.txt — The proposed {contract_type['label']} ({len(contract_type['section_pool'])} sections)
- /testbed/data/risk_matrix.txt — Company risk tolerance guidelines with severity classifications
- /testbed/data/market_benchmarks.txt — Industry standard terms for comparison
- /testbed/data/regulatory_requirements.txt — Mandatory regulatory clauses

## Requirements
1. Read the entire contract and identify each clause that poses a risk
2. For each risk identified, cite the specific section and clause
3. Classify each risk by severity (Critical, High, Medium, Low)
4. Compare each flagged clause against market benchmarks
5. Identify any mandatory regulatory clauses that are missing
6. Identify clauses that may appear aggressive but are actually within market norms
7. Provide specific recommendations for risk mitigation or clause modification

Write your risk analysis report to /testbed/risk_analysis.txt"""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/risk_analysis.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_contract_type",
            question=f"Does the report correctly identify the agreement as a {contract_type['label']}?",
            points=1,
        ),
    ]

    # Per-risk detection rubric items
    for risk in planted_risks:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"identifies_{risk['key']}",
                question=(
                    f"Does the report identify the following risk issue: {risk['label']}? "
                    f"({risk['risk_description']})"
                ),
                points=2,
            )
        )
        rubric_items.append(
            BinaryRubricCategory(
                name=f"correct_severity_{risk['key']}",
                question=(
                    f"Does the report classify the '{risk['label']}' issue as "
                    f"{risk['severity']} severity (or equivalent)?"
                ),
                points=1,
            )
        )

    # Missing clause detection
    missing_risks = [r for r in planted_risks if r["clause_template"] is None]
    if missing_risks:
        rubric_items.append(
            BinaryRubricCategory(
                name="identifies_missing_clauses",
                question=(
                    f"Does the report identify the following missing clauses: "
                    f"{', '.join(r['section'] for r in missing_risks)}?"
                ),
                points=2,
            )
        )

    # False positive checks
    for fp in planted_fps:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_flag_{fp['key']}",
                question=(
                    f"Does the report correctly avoid flagging the '{fp['section']}' "
                    f"clause as a risk issue? (This clause is within market benchmarks: "
                    f"{fp['explanation']})"
                ),
                points=2,
            )
        )

    # Summary checks
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_risk_count",
            question=f"Does the report identify approximately {n_risks} risk issues (within +/- 1)?",
            points=2,
        )
    )
    rubric_items.append(
        BinaryRubricCategory(
            name="cites_risk_matrix",
            question="Does the report reference or apply the company's risk tolerance guidelines when classifying at least one issue?",
            points=1,
        )
    )
    rubric_items.append(
        BinaryRubricCategory(
            name="cites_benchmarks",
            question="Does the report compare at least one flagged clause against industry benchmark data?",
            points=1,
        )
    )
    rubric_items.append(
        BinaryRubricCategory(
            name="cites_regulatory",
            question="Does the report reference regulatory requirements when identifying missing mandatory clauses?",
            points=1,
        )
    )
    rubric_items.append(
        BinaryRubricCategory(
            name="provides_recommendations",
            question="Does the report provide at least one specific recommendation for each identified risk (not just generic advice)?",
            points=2,
        )
    )

    rubric_items.append(
        RubricCategory(
            name="analysis_quality",
            description="How thorough and well-structured is the contract risk analysis?",
            failure="Superficial analysis; most risks missed or no cross-referencing.",
            minor_failure="Some risks found but lacking severity classification or benchmark comparison.",
            minor_success="Most risks identified with severity and benchmark comparison; minor gaps.",
            success="All risks identified with severity classification, benchmark comparison, regulatory cross-reference, and actionable recommendations.",
            points=3,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed risk analysis to /testbed/risk_analysis.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/contract.txt": contract_content,
            "/testbed/data/risk_matrix.txt": risk_matrix_content,
            "/testbed/data/market_benchmarks.txt": benchmarks_content,
            "/testbed/data/regulatory_requirements.txt": reg_content,
        },
        problem_type="contract_risk_analysis",
    )


# =============================================================================
# DOMAIN: REGULATORY FILING REVIEW
# =============================================================================

COMPANY_PROFILES = [
    {
        "name": "consumer_tech",
        "industry": "Consumer Technology",
        "sic_code": "3571",
        "products": "consumer electronics, mobile devices, and software services",
        "segments": ["Hardware", "Software & Services", "Advertising"],
        "revenue_range": (2_000_000_000, 8_000_000_000),
    },
    {
        "name": "pharma",
        "industry": "Pharmaceuticals",
        "sic_code": "2834",
        "products": "prescription drugs, biologics, and medical devices",
        "segments": ["Branded Pharmaceuticals", "Generics", "Medical Devices"],
        "revenue_range": (1_500_000_000, 6_000_000_000),
    },
    {
        "name": "retail",
        "industry": "Retail",
        "sic_code": "5331",
        "products": "general merchandise through physical stores and e-commerce",
        "segments": ["Stores", "E-Commerce", "Wholesale"],
        "revenue_range": (3_000_000_000, 12_000_000_000),
    },
    {
        "name": "energy",
        "industry": "Energy",
        "sic_code": "1311",
        "products": "crude oil, natural gas, and refined petroleum products",
        "segments": ["Upstream", "Downstream", "Midstream"],
        "revenue_range": (5_000_000_000, 20_000_000_000),
    },
    {
        "name": "financial",
        "industry": "Financial Services",
        "sic_code": "6022",
        "products": "banking, lending, and investment management services",
        "segments": ["Consumer Banking", "Commercial Banking", "Wealth Management"],
        "revenue_range": (3_000_000_000, 10_000_000_000),
    },
    {
        "name": "manufacturing",
        "industry": "Industrial Manufacturing",
        "sic_code": "3559",
        "products": "industrial equipment, automation systems, and replacement parts",
        "segments": ["Industrial Equipment", "Automation Solutions", "Aftermarket Services"],
        "revenue_range": (1_000_000_000, 5_000_000_000),
    },
    {
        "name": "healthcare_services",
        "industry": "Healthcare Services",
        "sic_code": "8062",
        "products": "hospital and outpatient care services",
        "segments": ["Hospital Operations", "Outpatient Services", "Insurance Services"],
        "revenue_range": (4_000_000_000, 15_000_000_000),
    },
    {
        "name": "telecom",
        "industry": "Telecommunications",
        "sic_code": "4813",
        "products": "wireless and wireline communication services",
        "segments": ["Wireless", "Wireline", "Enterprise Solutions"],
        "revenue_range": (6_000_000_000, 25_000_000_000),
    },
]

# Pool of plantable disclosure issues
FILING_ISSUES = [
    {
        "key": "material_change_undisclosed",
        "label": "Material change not disclosed",
        "description_template": (
            "The {segment} segment experienced a {change_pct}% decline in revenue in Q4 "
            "compared to Q3, but the MD&A section does not discuss this decline or its causes."
        ),
    },
    {
        "key": "narrative_data_inconsistency",
        "label": "Narrative inconsistent with financial data",
        "description_template": (
            "The MD&A states that '{positive_narrative}' but the financial data shows "
            "{actual_data} — a direct contradiction."
        ),
    },
    {
        "key": "missing_risk_factor",
        "label": "Missing risk factor update",
        "description_template": (
            "The prior year filing included a risk factor regarding '{old_risk}'. This risk "
            "has materialized (evidenced by {evidence}), but the current filing does not "
            "update or reference this risk factor."
        ),
    },
    {
        "key": "related_party_buried",
        "label": "Related party transaction buried or omitted",
        "description_template": (
            "A transaction of {amount} with {related_party} (a company controlled by "
            "{relationship}) appears in the financial data but is not disclosed in the "
            "related party transactions note."
        ),
    },
    {
        "key": "off_balance_sheet",
        "label": "Off-balance-sheet item not disclosed",
        "description_template": (
            "The company has {obs_item} totaling {obs_amount} that does not appear on "
            "the balance sheet and is not mentioned in the notes to financial statements."
        ),
    },
    {
        "key": "accounting_policy_change",
        "label": "Accounting policy change without disclosure",
        "description_template": (
            "The company changed its {policy_area} from {old_method} to {new_method} "
            "between the prior year and current year filings, but this change is not "
            "disclosed in the accounting policies note."
        ),
    },
    {
        "key": "segment_reporting_gap",
        "label": "Segment reporting gap",
        "description_template": (
            "The {missing_segment} segment's {missing_metric} is not reported separately "
            "despite representing {seg_pct}% of consolidated {missing_metric}, which "
            "exceeds the 10% threshold for separate segment disclosure."
        ),
    },
    {
        "key": "going_concern_indicator",
        "label": "Going concern indicators not flagged",
        "description_template": (
            "The company's {gc_indicator} raises substantial doubt about its ability to "
            "continue as a going concern, but no going concern disclosure is included."
        ),
    },
]


def make_regulatory_filing_review(rand_seed: int = 42) -> RubricDatapoint:
    """Review a regulatory filing (SEC 10-K style) for disclosure issues.
    Cross-reference financial data, narrative sections, disclosure checklist,
    and prior year filing.

    Seed varies: company profile, financial data, which issues are planted,
    narrative content, prior year comparison data.
    """
    rng = _random.Random(rand_seed)

    reviewer_name = random_name(rand_seed)
    company_name = pick1(COMPANY_NAMES, rand_seed)
    cfo_name = random_name(rand_seed + 1)
    auditor_name = random_name(rand_seed + 2)

    profile = rng.choice(COMPANY_PROFILES)
    fiscal_year = rng.choice([2023, 2024])

    # --- Generate financial data ---
    base_revenue = rng.uniform(*profile["revenue_range"])
    n_segments = len(profile["segments"])

    # Segment revenue allocation
    seg_pcts = []
    remaining = 100.0
    for i in range(n_segments - 1):
        pct = rng.uniform(15, remaining - 15 * (n_segments - i - 1))
        seg_pcts.append(round(pct, 1))
        remaining -= pct
    seg_pcts.append(round(remaining, 1))

    # Quarterly revenue data (4 quarters)
    quarterly_data: list[dict] = []
    for q in range(4):
        q_rev = base_revenue / 4 * rng.uniform(0.85, 1.15)
        q_cogs = q_rev * rng.uniform(0.55, 0.70)
        q_sga = q_rev * rng.uniform(0.12, 0.22)
        q_rd = q_rev * rng.uniform(0.03, 0.10)
        q_interest = rng.uniform(5_000_000, 30_000_000)
        q_tax_rate = rng.uniform(0.18, 0.25)
        q_other_income = rng.uniform(-5_000_000, 10_000_000)
        q_operating = q_rev - q_cogs - q_sga - q_rd
        q_pretax = q_operating + q_other_income - q_interest
        q_net = q_pretax * (1 - q_tax_rate)
        quarterly_data.append({
            "quarter": f"Q{q + 1}",
            "revenue": round(q_rev, 2),
            "cogs": round(q_cogs, 2),
            "sga": round(q_sga, 2),
            "rd": round(q_rd, 2),
            "operating_income": round(q_operating, 2),
            "interest_expense": round(q_interest, 2),
            "other_income": round(q_other_income, 2),
            "pretax_income": round(q_pretax, 2),
            "income_tax": round(q_pretax * q_tax_rate, 2),
            "net_income": round(q_net, 2),
        })

    total_revenue = sum(q["revenue"] for q in quarterly_data)
    total_net_income = sum(q["net_income"] for q in quarterly_data)

    # Segment revenue data
    segment_data: list[dict] = []
    for i, seg_name in enumerate(profile["segments"]):
        seg_rev = total_revenue * seg_pcts[i] / 100
        seg_op_margin = rng.uniform(0.05, 0.25)
        segment_data.append({
            "segment": seg_name,
            "revenue": round(seg_rev, 2),
            "pct_of_total": seg_pcts[i],
            "operating_margin": round(seg_op_margin * 100, 1),
            "operating_income": round(seg_rev * seg_op_margin, 2),
        })

    # Balance sheet items
    total_assets = total_revenue * rng.uniform(0.8, 1.5)
    total_debt = total_assets * rng.uniform(0.25, 0.55)
    total_equity = total_assets - total_debt - total_assets * rng.uniform(0.1, 0.25)
    cash = total_assets * rng.uniform(0.05, 0.15)
    current_ratio = rng.uniform(0.8, 2.5)

    # Prior year data for comparison
    prior_revenue = total_revenue * rng.uniform(0.90, 1.15)
    prior_net_income = total_net_income * rng.uniform(0.85, 1.20)

    # --- Select and parameterize planted issues ---
    n_issues = rng.randint(5, 8)
    available_issues = list(FILING_ISSUES)
    rng.shuffle(available_issues)
    planted_issues_raw = available_issues[:n_issues]

    # Parameterize each issue
    planted_issues: list[dict] = []
    for issue_raw in planted_issues_raw:
        issue = dict(issue_raw)
        key = issue["key"]

        if key == "material_change_undisclosed":
            # Plant a Q4 decline in a random segment
            decline_seg_idx = rng.randint(0, n_segments - 1)
            decline_seg = profile["segments"][decline_seg_idx]
            change_pct = rng.randint(15, 35)
            # Actually modify Q4 revenue for this segment
            issue["params"] = {"segment": decline_seg, "change_pct": change_pct}
            issue["instantiated"] = issue["description_template"].format(
                segment=decline_seg, change_pct=change_pct,
            )

        elif key == "narrative_data_inconsistency":
            # MD&A says revenue grew, but Q4 actually declined
            q4_rev = quarterly_data[3]["revenue"]
            q3_rev = quarterly_data[2]["revenue"]
            # Force Q4 to be lower than Q3 for this issue
            if q4_rev >= q3_rev:
                quarterly_data[3]["revenue"] = round(q3_rev * rng.uniform(0.82, 0.92), 2)
                quarterly_data[3]["net_income"] = round(
                    quarterly_data[3]["net_income"] * rng.uniform(0.70, 0.85), 2
                )
            positive_narrative = "revenue demonstrated consistent growth throughout the fiscal year"
            actual_q4 = quarterly_data[3]["revenue"]
            actual_q3 = quarterly_data[2]["revenue"]
            decline_amount = actual_q3 - actual_q4
            issue["params"] = {
                "positive_narrative": positive_narrative,
                "actual_data": f"Q4 revenue of {_fmt_money(actual_q4)} declined by {_fmt_money(decline_amount)} from Q3 revenue of {_fmt_money(actual_q3)}",
            }
            issue["instantiated"] = issue["description_template"].format(**issue["params"])

        elif key == "missing_risk_factor":
            old_risks = [
                ("supply chain disruption in key component markets", "a 12% increase in procurement costs and 3 delayed product launches"),
                ("regulatory changes in our primary operating jurisdictions", "two new compliance penalties totaling " + _fmt_money(rng.uniform(2_000_000, 10_000_000))),
                ("cybersecurity threats to customer data", "a data breach affecting " + str(rng.randint(50000, 500000)) + " customer records disclosed in Q2"),
                ("key customer concentration risk", "the loss of a customer representing " + _fmt_pct(rng.uniform(8, 18)) + " of annual revenue"),
                ("foreign currency exchange rate volatility", "a " + _fmt_money(rng.uniform(10_000_000, 50_000_000)) + " foreign exchange loss recognized in Q3"),
            ]
            old_risk, evidence = rng.choice(old_risks)
            issue["params"] = {"old_risk": old_risk, "evidence": evidence}
            issue["instantiated"] = issue["description_template"].format(**issue["params"])

        elif key == "related_party_buried":
            related_names = random_names(rand_seed + 50, 3)
            related_party_name = rng.choice(related_names)
            relationships = [
                f"a board member ({related_party_name})",
                f"the CEO's spouse ({related_party_name})",
                f"a major shareholder ({related_party_name})",
            ]
            relationship = rng.choice(relationships)
            rp_amount = _fmt_money(rng.uniform(1_000_000, 15_000_000))
            rp_company = f"{related_party_name.split()[1]} Holdings LLC"
            issue["params"] = {
                "amount": rp_amount,
                "related_party": rp_company,
                "relationship": relationship,
            }
            issue["instantiated"] = issue["description_template"].format(**issue["params"])

        elif key == "off_balance_sheet":
            obs_items = [
                ("operating lease commitments", _fmt_money(rng.uniform(50_000_000, 200_000_000))),
                ("unconditional purchase obligations", _fmt_money(rng.uniform(30_000_000, 100_000_000))),
                ("variable interest entity (VIE) assets", _fmt_money(rng.uniform(40_000_000, 150_000_000))),
                ("guarantee obligations to unconsolidated subsidiaries", _fmt_money(rng.uniform(20_000_000, 80_000_000))),
            ]
            obs_item, obs_amount = rng.choice(obs_items)
            issue["params"] = {"obs_item": obs_item, "obs_amount": obs_amount}
            issue["instantiated"] = issue["description_template"].format(**issue["params"])

        elif key == "accounting_policy_change":
            policy_changes = [
                ("revenue recognition method", "completed contract method", "percentage of completion method"),
                ("inventory valuation", "FIFO (first-in, first-out)", "weighted average cost"),
                ("depreciation method for fixed assets", "straight-line depreciation", "accelerated depreciation (double declining balance)"),
                ("bad debt estimation approach", "aging schedule method", "percentage of sales method"),
            ]
            policy_area, old_method, new_method = rng.choice(policy_changes)
            issue["params"] = {
                "policy_area": policy_area,
                "old_method": old_method,
                "new_method": new_method,
            }
            issue["instantiated"] = issue["description_template"].format(**issue["params"])

        elif key == "segment_reporting_gap":
            missing_seg_idx = rng.randint(0, n_segments - 1)
            missing_seg = profile["segments"][missing_seg_idx]
            seg_pct_val = seg_pcts[missing_seg_idx]
            if seg_pct_val < 10:
                seg_pct_val = rng.uniform(11, 18)
                seg_pcts[missing_seg_idx] = round(seg_pct_val, 1)
            missing_metrics = ["revenue", "operating income", "total assets"]
            missing_metric = rng.choice(missing_metrics)
            issue["params"] = {
                "missing_segment": missing_seg,
                "missing_metric": missing_metric,
                "seg_pct": round(seg_pct_val, 1),
            }
            issue["instantiated"] = issue["description_template"].format(**issue["params"])

        elif key == "going_concern_indicator":
            gc_indicators = [
                f"current ratio of {current_ratio:.2f} (below 1.0) combined with negative operating cash flow of {_fmt_money(rng.uniform(-50_000_000, -10_000_000))} and debt covenants requiring a minimum current ratio of 1.2",
                f"accumulated deficit of {_fmt_money(total_equity * rng.uniform(-0.5, -0.2))} with total debt of {_fmt_money(total_debt)} maturing within 18 months and insufficient cash reserves of {_fmt_money(cash)}",
                f"three consecutive quarters of operating losses totaling {_fmt_money(rng.uniform(-100_000_000, -30_000_000))} with a debt-to-equity ratio of {total_debt / max(total_equity, 1):.1f}x",
            ]
            gc_indicator = rng.choice(gc_indicators)
            issue["params"] = {"gc_indicator": gc_indicator}
            issue["instantiated"] = issue["description_template"].format(**issue["params"])

        planted_issues.append(issue)

    # --- Build filing_draft.txt ---
    filing_lines = [
        f"ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d)",
        f"OF THE SECURITIES EXCHANGE ACT OF 1934",
        "",
        f"For the fiscal year ended December 31, {fiscal_year}",
        f"Commission File Number: 001-{rng.randint(10000, 99999)}",
        "",
        f"Company: {company_name}",
        f"SIC Code: {profile['sic_code']}",
        f"Industry: {profile['industry']}",
        "",
        "=" * 65,
        "PART I",
        "=" * 65,
        "",
        "Item 1. Business",
        "",
        f"{company_name} is a leading provider of {profile['products']}. The company",
        f"operates through {n_segments} reportable segments: {', '.join(profile['segments'])}.",
        "",
        "Item 1A. Risk Factors",
        "",
        "The following risk factors could materially affect our business:",
        "",
    ]

    # Add some risk factors but omit one that should be updated
    standard_risks = [
        "Competition in our markets is intense and may increase.",
        "Our success depends on our ability to attract and retain key personnel.",
        "Disruptions in our information technology systems could harm our business.",
        "Changes in tax laws or regulations could adversely affect our results.",
        "Our international operations subject us to various risks.",
    ]
    for i, risk in enumerate(standard_risks, 1):
        filing_lines.append(f"  Risk Factor {i}: {risk}")
        filing_lines.append("")

    filing_lines.extend([
        "=" * 65,
        "PART II",
        "=" * 65,
        "",
        "Item 7. Management's Discussion and Analysis (MD&A)",
        "",
        f"Overview: Fiscal year {fiscal_year} was a year of strategic execution for",
        f"{company_name}. We continued to invest in our core business segments while",
        f"pursuing operational efficiencies.",
        "",
        "Revenue Performance:",
    ])

    # The narrative may contain the inconsistency if that issue is planted
    has_narrative_inconsistency = any(i["key"] == "narrative_data_inconsistency" for i in planted_issues)
    if has_narrative_inconsistency:
        filing_lines.append(
            f"  Total revenue for fiscal year {fiscal_year} was {_fmt_money(total_revenue)}. "
            f"Revenue demonstrated consistent growth throughout the fiscal year, reflecting "
            f"strong demand across all segments and successful execution of our pricing strategy."
        )
    else:
        filing_lines.append(
            f"  Total revenue for fiscal year {fiscal_year} was {_fmt_money(total_revenue)}."
        )
    filing_lines.append("")

    # Segment discussion
    filing_lines.append("Segment Performance:")
    for seg in segment_data:
        filing_lines.append(
            f"  {seg['segment']}: Revenue of {_fmt_money(seg['revenue'])} "
            f"({seg['pct_of_total']}% of total), operating margin of {seg['operating_margin']}%."
        )
    filing_lines.append("")

    filing_lines.extend([
        f"Net income for the year was {_fmt_money(total_net_income)}.",
        "",
        "Item 8. Financial Statements and Supplementary Data",
        "",
        "  See financial_data.csv for detailed quarterly financial data.",
        "",
    ])

    # Notes section — deliberately sparse to allow missing disclosures
    filing_lines.extend([
        "NOTES TO FINANCIAL STATEMENTS",
        "",
        "Note 1. Summary of Significant Accounting Policies",
        f"  The financial statements are prepared in conformity with U.S. GAAP.",
        f"  Revenue is recognized when control of promised goods or services transfers to customers.",
        "",
        "Note 2. Revenue Disaggregation",
        f"  Revenue is disaggregated by segment as shown in the segment performance section.",
        "",
        "Note 3. Debt",
        f"  Total long-term debt as of December 31, {fiscal_year}: {_fmt_money(total_debt)}",
        f"  Current portion of long-term debt: {_fmt_money(total_debt * rng.uniform(0.05, 0.15))}",
        "",
    ])

    filing_lines.append("")
    filing_content = "\n".join(filing_lines) + "\n"

    # --- Build financial_data.csv ---
    csv_lines = [
        "quarter,revenue,cost_of_goods_sold,sg_and_a,r_and_d,operating_income,interest_expense,other_income,pretax_income,income_tax,net_income"
    ]
    for q in quarterly_data:
        csv_lines.append(
            f"{q['quarter']},{q['revenue']:.2f},{q['cogs']:.2f},{q['sga']:.2f},"
            f"{q['rd']:.2f},{q['operating_income']:.2f},{q['interest_expense']:.2f},"
            f"{q['other_income']:.2f},{q['pretax_income']:.2f},{q['income_tax']:.2f},"
            f"{q['net_income']:.2f}"
        )
    # Add segment data rows
    csv_lines.append("")
    csv_lines.append("segment,segment_revenue,pct_of_total,operating_margin_pct,segment_operating_income")
    for seg in segment_data:
        csv_lines.append(
            f"{seg['segment']},{seg['revenue']:.2f},{seg['pct_of_total']},{seg['operating_margin']},{seg['operating_income']:.2f}"
        )

    # Add balance sheet summary
    csv_lines.append("")
    csv_lines.append("balance_sheet_item,amount")
    csv_lines.append(f"Total Assets,{total_assets:.2f}")
    csv_lines.append(f"Total Debt,{total_debt:.2f}")
    csv_lines.append(f"Total Equity,{total_equity:.2f}")
    csv_lines.append(f"Cash and Equivalents,{cash:.2f}")
    csv_lines.append(f"Current Ratio,{current_ratio:.2f}")

    # Plant related party transaction in the financial data if that issue exists
    rp_issue = next((i for i in planted_issues if i["key"] == "related_party_buried"), None)
    if rp_issue:
        csv_lines.append("")
        csv_lines.append("other_transactions,counterparty,amount,description")
        csv_lines.append(
            f"Consulting Services,{rp_issue['params']['related_party']},"
            f"{rp_issue['params']['amount'].replace('$', '').replace(',', '')},"
            f"Management consulting and advisory services"
        )

    csv_content = "\n".join(csv_lines) + "\n"

    # --- Build disclosure_checklist.txt ---
    checklist_lines = [
        "SEC / GAAP DISCLOSURE CHECKLIST",
        "",
        "Required disclosures for annual report (10-K) filing:",
        "",
        "=" * 65,
        "FINANCIAL STATEMENTS",
        "=" * 65,
        "",
        "  [ ] Income statement by quarter",
        "  [ ] Balance sheet (end of period)",
        "  [ ] Cash flow statement",
        "  [ ] Shareholders' equity statement",
        "  [ ] Revenue disaggregated by segment (ASC 606)",
        "  [ ] Segment reporting with quantitative thresholds (ASC 280)",
        "      - Segments representing 10%+ of revenue, profit, or assets must be reported separately",
        "",
        "=" * 65,
        "MD&A REQUIREMENTS",
        "=" * 65,
        "",
        "  [ ] Discussion of results of operations (period-over-period comparison)",
        "  [ ] Material trends, events, or uncertainties affecting revenue or income",
        "  [ ] Known trends that may affect future operations",
        "  [ ] Critical accounting estimates and judgments",
        "  [ ] Liquidity and capital resources",
        "  [ ] Off-balance-sheet arrangements",
        "",
        "=" * 65,
        "NOTES TO FINANCIAL STATEMENTS",
        "=" * 65,
        "",
        "  [ ] Summary of significant accounting policies (ASC 235)",
        "  [ ] Changes in accounting policies (ASC 250) — must disclose nature, reason, and effect",
        "  [ ] Related party transactions (ASC 850) — nature, amounts, and terms",
        "  [ ] Commitments and contingencies (ASC 450)",
        "  [ ] Debt and credit facilities (ASC 470)",
        "  [ ] Fair value measurements (ASC 820)",
        "  [ ] Going concern assessment (ASC 205-40) — required when indicators present",
        "",
        "=" * 65,
        "RISK FACTORS",
        "=" * 65,
        "",
        "  [ ] Material risks specific to the company and its operations",
        "  [ ] Updates for risks that have materialized since prior filing",
        "  [ ] New risks identified during the period",
        "  [ ] Risk factors should not be generic boilerplate",
        "",
        "=" * 65,
        "CONSISTENCY CHECKS",
        "=" * 65,
        "",
        "  [ ] Narrative in MD&A must be consistent with financial data",
        "  [ ] Year-over-year changes must be explained",
        "  [ ] Segment totals must reconcile to consolidated figures",
        "  [ ] Prior year comparisons must be provided",
        "",
    ]
    checklist_content = "\n".join(checklist_lines) + "\n"

    # --- Build prior_year_filing.txt ---
    prior_lines = [
        f"ANNUAL REPORT — FISCAL YEAR ENDED DECEMBER 31, {fiscal_year - 1}",
        f"(Prior Year Filing — Summary for Comparison)",
        "",
        f"Company: {company_name}",
        "",
        "=" * 65,
        "KEY FINANCIAL DATA",
        "=" * 65,
        "",
        f"Total Revenue: {_fmt_money(prior_revenue)}",
        f"Net Income: {_fmt_money(prior_net_income)}",
        f"Total Assets: {_fmt_money(total_assets * rng.uniform(0.90, 1.05))}",
        f"Total Debt: {_fmt_money(total_debt * rng.uniform(0.85, 1.10))}",
        "",
        "=" * 65,
        "RISK FACTORS (Prior Year)",
        "=" * 65,
        "",
    ]

    # Include the risk that should have been updated
    risk_factor_issue = next((i for i in planted_issues if i["key"] == "missing_risk_factor"), None)
    prior_risk_factors = list(standard_risks)
    if risk_factor_issue:
        prior_risk_factors.append(
            f"We face significant risk from {risk_factor_issue['params']['old_risk']}."
        )
    for i, risk in enumerate(prior_risk_factors, 1):
        prior_lines.append(f"  Risk Factor {i}: {risk}")
        prior_lines.append("")

    prior_lines.extend([
        "=" * 65,
        "ACCOUNTING POLICIES (Prior Year)",
        "=" * 65,
        "",
    ])

    # Include the old accounting policy if that issue is planted
    policy_issue = next((i for i in planted_issues if i["key"] == "accounting_policy_change"), None)
    if policy_issue:
        prior_lines.append(
            f"  {policy_issue['params']['policy_area'].capitalize()}: "
            f"{policy_issue['params']['old_method']}"
        )
    else:
        prior_lines.append("  Revenue recognition: Recognized when control transfers to customer.")
    prior_lines.append("  Inventory valuation: FIFO method.")
    prior_lines.append("  Depreciation: Straight-line over estimated useful lives.")
    prior_lines.append("")

    prior_lines.extend([
        "=" * 65,
        "SEGMENT REPORTING (Prior Year)",
        "=" * 65,
        "",
    ])
    for seg in profile["segments"]:
        prior_seg_rev = prior_revenue * rng.uniform(0.2, 0.5)
        prior_lines.append(f"  {seg}: {_fmt_money(prior_seg_rev)}")
    prior_lines.append("")

    prior_lines.extend([
        "=" * 65,
        "RELATED PARTY TRANSACTIONS (Prior Year)",
        "=" * 65,
        "",
        "  No material related party transactions to report.",
        "",
    ])

    prior_content = "\n".join(prior_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Regulatory Filing Review (SEC 10-K)

You are {reviewer_name}, a senior financial analyst at an independent audit review firm.
You have been engaged to review the annual report (10-K filing) of {company_name}
for fiscal year {fiscal_year}.

## Source Files
- /testbed/data/filing_draft.txt — The annual report draft with financial statements, risk factors, and MD&A
- /testbed/data/financial_data.csv — Detailed financial line items by quarter, segment data, and balance sheet
- /testbed/data/disclosure_checklist.txt — Required disclosure items per SEC/GAAP regulations
- /testbed/data/prior_year_filing.txt — Last year's filing for comparison

## Requirements
1. Cross-reference the MD&A narrative against the actual financial data for inconsistencies
2. Compare the current filing against the prior year filing to identify undisclosed changes
3. Check all items on the disclosure checklist against the filing
4. Identify any material items in the financial data that lack proper disclosure
5. Review risk factors for completeness and updates
6. Look for related party transactions, off-balance-sheet items, and accounting policy changes
7. Assess whether going concern indicators are present and properly disclosed

Write your filing review report to /testbed/review_report.txt"""

    # --- Build rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/review_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_company_and_year",
            question=f"Does the report correctly identify {company_name} and fiscal year {fiscal_year}?",
            points=1,
        ),
    ]

    # Per-issue detection rubric items
    for issue in planted_issues:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"identifies_{issue['key']}",
                question=(
                    f"Does the report identify the following disclosure issue: "
                    f"{issue['label']}? (Specifically: {issue['instantiated']})"
                ),
                points=2,
            )
        )

    # Specific cross-reference checks
    if has_narrative_inconsistency:
        rubric_items.append(
            BinaryRubricCategory(
                name="cites_specific_q4_numbers",
                question=(
                    f"When identifying the narrative/data inconsistency, does the report cite "
                    f"specific Q4 and Q3 revenue figures from the financial data?"
                ),
                points=2,
            )
        )

    if risk_factor_issue:
        rubric_items.append(
            BinaryRubricCategory(
                name="cross_references_prior_year_risk",
                question=(
                    f"Does the report cross-reference the current risk factors against the "
                    f"prior year filing to identify the missing risk factor update?"
                ),
                points=2,
            )
        )

    if rp_issue:
        rubric_items.append(
            BinaryRubricCategory(
                name="finds_rp_in_financial_data",
                question=(
                    f"Does the report identify the related party transaction by finding it "
                    f"in the financial data (CSV) and noting its absence from the filing narrative?"
                ),
                points=2,
            )
        )

    if policy_issue:
        rubric_items.append(
            BinaryRubricCategory(
                name="compares_accounting_policies",
                question=(
                    f"Does the report identify the accounting policy change by comparing "
                    f"the prior year's {policy_issue['params']['policy_area']} method to the current year?"
                ),
                points=2,
            )
        )

    # Checklist cross-reference
    rubric_items.append(
        BinaryRubricCategory(
            name="references_disclosure_checklist",
            question="Does the report reference the SEC/GAAP disclosure checklist when identifying missing items?",
            points=1,
        )
    )

    # Financial data cross-checks (always present)
    rev_change_pct = round((total_revenue - prior_revenue) / prior_revenue * 100, 1)
    rev_direction = "increase" if rev_change_pct >= 0 else "decrease"
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_total_revenue",
            question=f"Does the report cite the total revenue as approximately {_fmt_money(total_revenue)} (within 5%)?",
            points=1,
        )
    )
    rubric_items.append(
        BinaryRubricCategory(
            name="year_over_year_comparison",
            question=(
                f"Does the report note the year-over-year revenue change "
                f"({rev_direction} of approximately {abs(rev_change_pct):.1f}%) "
                f"by comparing to prior year revenue of {_fmt_money(prior_revenue)}?"
            ),
            points=2,
        )
    )
    rubric_items.append(
        BinaryRubricCategory(
            name="identifies_segment_count",
            question=f"Does the report identify that the company operates {n_segments} reportable segments ({', '.join(profile['segments'])})?",
            points=1,
        )
    )
    rubric_items.append(
        BinaryRubricCategory(
            name="reviews_all_four_sources",
            question="Does the report demonstrate that it reviewed all four source documents (filing draft, financial data CSV, disclosure checklist, and prior year filing)?",
            points=1,
        )
    )

    # Issue count
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_issue_count",
            question=f"Does the report identify approximately {n_issues} disclosure issues (within +/- 1)?",
            points=2,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="provides_remediation",
            question="Does the report provide specific remediation recommendations for at least 3 of the identified issues?",
            points=2,
        )
    )

    rubric_items.append(
        RubricCategory(
            name="analysis_thoroughness",
            description="How thorough and systematic is the filing review?",
            failure="Superficial review; most issues missed or analysis lacks cross-referencing.",
            minor_failure="Some issues found but analysis does not cross-reference between documents systematically.",
            minor_success="Most issues found with cross-referencing between filing, financial data, and prior year.",
            success="All issues identified with detailed cross-referencing, specific citations to financial data, checklist items, and prior year comparison.",
            points=3,
        )
    )

    rubric = tuple(rubric_items)

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=rubric,
        submission_instructions="Write your completed filing review report to /testbed/review_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/filing_draft.txt": filing_content,
            "/testbed/data/financial_data.csv": csv_content,
            "/testbed/data/disclosure_checklist.txt": checklist_content,
            "/testbed/data/prior_year_filing.txt": prior_content,
        },
        problem_type="regulatory_filing_review",
    )
