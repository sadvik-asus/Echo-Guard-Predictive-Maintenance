"""
EchoGuard Predictive Maintenance Advisor & Root Cause SOP System
Knowledge-assisted diagnostic engine providing Standard Operating Procedures (SOPs),
Root Cause Failure Analysis (RCFA), and ISO-compliant maintenance recommendations.
"""

from typing import Dict, Any, List, Optional
import re


# Industrial Maintenance Knowledge Base (ISO 13373, ISO 15243, ISO 10816)
MAINTENANCE_KNOWLEDGE_BASE = [
    {
        "id": "KB-FAULT-BPFO",
        "title": "Bearing Outer Race Defect (BPFO / Flaking & Spalling)",
        "keywords": ["outer race", "bpfo", "spalling", "flaking", "shockwave", "kurtosis", "impact"],
        "category": "Bearing Fatigue",
        "iso_standard": "ISO 15243 §5.1.2 (Subsurface Initiated Fatigue)",
        "symptoms": [
            "Periodic transient shockwaves at BPFO frequency (~236 Hz at 2000 RPM)",
            "Elevated Kurtosis (> 3.5) indicating impulsive surface impacts",
            "Crest Factor spikes exceeding 4.0",
        ],
        "root_causes": [
            "Normal L10 fatigue life exceeded under continuous industrial loading",
            "Contamination of grease with solid abrasive debris or moisture",
            "Excessive dynamic radial load or shaft misalignment",
        ],
        "severity": "CRITICAL",
        "action_timeline": "Within 24 to 48 operating hours",
        "sop_steps": [
            "1. Reduce machine operational load immediately to avoid catastrophic spindle seizure.",
            "2. Inspect bearing housing for abnormal acoustic noise and elevated temperature (> 70°C).",
            "3. Take grease/oil sample and perform ferrographic particle analysis for metallic flakes.",
            "4. Verify coupling alignment using laser measurement tools.",
            "5. Schedule planned maintenance window for bearing unit replacement.",
        ],
        "safety_notes": "Lock out and tag out (LOTO) electrical power before removing bearing guard."
    },
    {
        "id": "KB-FAULT-BPFI",
        "title": "Bearing Inner Race Defect (BPFI / Surface Fatigue)",
        "keywords": ["inner race", "bpfi", "shaft fit", "sideband", "inner ring"],
        "category": "Bearing Fatigue",
        "iso_standard": "ISO 15243 §5.1.3 (Surface Initiated Fatigue)",
        "symptoms": [
            "Impulsive impacts modulated by shaft rotational speed 1X",
            "Distinct sideband peaks around BPFI in high-resolution FFT",
            "Elevated peak-to-peak vibration amplitude",
        ],
        "root_causes": [
            "Excessive shaft interference fit causing hoop stress",
            "Shaft deflection or bent spindle under heavy torque",
            "Inadequate lubrication film thickness (insufficient viscosity)",
        ],
        "severity": "CRITICAL",
        "action_timeline": "Within 24 hours",
        "sop_steps": [
            "1. Plan controlled production slowdown or switch to redundant drive unit.",
            "2. Measure shaft runout and radial clearance.",
            "3. Inspect shaft journal surface for fretting corrosion or scoring.",
            "4. Replace bearing assembly and ensure correct thermal mount induction method.",
        ],
        "safety_notes": "Never apply direct hammer blows to the hardened inner race during replacement."
    },
    {
        "id": "KB-FAULT-LUB",
        "title": "Lubrication Breakdown & Friction Starvation",
        "keywords": ["lubrication", "grease", "dry", "friction", "starvation", "high frequency", "oil"],
        "category": "Tribological Failure",
        "iso_standard": "ISO 15243 §5.2.2 (Abrasive Wear & Inadequate Lubrication)",
        "symptoms": [
            "Broadband high-frequency acoustic emission (> 2 kHz)",
            "Rising operating temperature without prominent discrete defect peaks",
            "Gradual increase in overall RMS vibration",
        ],
        "root_causes": [
            "Grease dry-out, thermal degradation, or extended relubrication interval",
            "Incorrect grease viscosity grade (NLGI rating mismatch)",
            "Damaged contact seals allowing contaminant ingress and lubricant washout",
        ],
        "severity": "WARNING",
        "action_timeline": "Immediate (within current shift)",
        "sop_steps": [
            "1. Check bearing housing temperature using calibrated infrared thermometer.",
            "2. Inspect seal condition for grease leakage or hardening.",
            "3. Purge old contaminated grease and replenish with OEM-specified high-pressure grease.",
            "4. Re-measure vibration after 30 minutes of runtime to confirm baseline recovery.",
        ],
        "safety_notes": "Avoid over-greasing; excess grease causes churning, friction, and thermal runaway."
    },
    {
        "id": "KB-FAULT-UNBALANCE",
        "title": "Rotor Unbalance & Mechanical Misalignment",
        "keywords": ["unbalance", "misalignment", "harmonic", "1x", "2x", "runout", "rotor"],
        "category": "Mechanical Dynamic Fault",
        "iso_standard": "ISO 10816-3 / ISO 20816 (Machinery Vibration Evaluation)",
        "symptoms": [
            "Dominant vibration energy concentrated at 1X rotational speed (~33.3 Hz at 2000 RPM)",
            "Elevated 2X harmonic indicating angular or parallel coupling offset",
            "High overall sinusoidal vibration amplitude with normal Kurtosis (~3.0)",
        ],
        "root_causes": [
            "Rotor mass unbalance due to material deposit or wear",
            "Coupling misalignment between driving motor and driven equipment",
            "Loose foundation hold-down bolts or soft-foot condition",
        ],
        "severity": "WARNING",
        "action_timeline": "Within 7 operating days",
        "sop_steps": [
            "1. Inspect rotor assembly for particulate buildup or missing balance weights.",
            "2. Perform laser shaft alignment check on motor-pump/spindle coupling.",
            "3. Torque all foundation mounting bolts to specified engineering limits.",
            "4. Conduct single-plane or two-plane dynamic field balancing.",
        ],
        "safety_notes": "Ensure all rotating guards are secured prior to test runs."
    },
    {
        "id": "KB-HEALTHY-BASELINE",
        "title": "Nominal Machine Condition & Healthy Baseline",
        "keywords": ["healthy", "normal", "nominal", "good", "baseline", "routine"],
        "category": "Operational Baseline",
        "iso_standard": "ISO 10816-3 Zone A / B",
        "symptoms": [
            "Vibration energy evenly distributed across baseline rotational harmonics",
            "Kurtosis within nominal range (2.7 - 3.2)",
            "Crest Factor stable below 3.5; ISO 10816 Zone A/B status",
        ],
        "root_causes": [
            "Machinery running within designed engineering tolerances",
        ],
        "severity": "NORMAL",
        "action_timeline": "Next scheduled routine interval (30 days)",
        "sop_steps": [
            "1. Continue standard production and telemetry ingestion.",
            "2. Log vibration RMS values into condition monitoring trend database.",
            "3. Maintain preventive relubrication schedule according to maintenance calendar.",
        ],
        "safety_notes": "Maintain standard plant PPE and continuous monitoring sensors."
    }
]


def query_knowledge_base(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves matching maintenance standard procedures from the knowledge base
    using token-weighted keyword matching.

    Args:
        query_text: Query string or fault symptom description.
        top_k: Number of relevant entries to return.

    Returns:
        List of matching knowledge base items with match score.
    """
    query_tokens = set(re.findall(r'\b[a-z0-9]+\b', query_text.lower()))
    if not query_tokens:
        return MAINTENANCE_KNOWLEDGE_BASE[:top_k]

    scored = []
    for item in MAINTENANCE_KNOWLEDGE_BASE:
        score = 0
        text_corpus = f"{item['title']} {item['category']} {' '.join(item['keywords'])} {' '.join(item['symptoms'])}".lower()

        for token in query_tokens:
            if token in item['keywords']:
                score += 3
            elif token in text_corpus:
                score += 1

        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return [MAINTENANCE_KNOWLEDGE_BASE[0]]

    return [item for _, item in scored[:top_k]]


def generate_maintenance_advisory(
    prediction: str,
    confidence: float,
    features: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Synthesizes sensor metrics, CNN classification, and industrial maintenance manuals
    into a comprehensive, actionable Maintenance Advisory report.

    Args:
        prediction: "HEALTHY" or "FAULTY"
        confidence: Model prediction score (0.0 to 1.0)
        features: Optional dictionary of extracted vibration metrics (from src.features).

    Returns:
        Structured advisory dictionary containing root cause, SOP, and immediate actions.
    """
    time_feats = features.get("time_domain", {}) if features else {}
    freq_feats = features.get("frequency_domain", {}) if features else {}
    iso_feats = features.get("iso_severity", {}) if features else {}

    kurtosis = float(time_feats.get("kurtosis", 3.0))
    crest = float(time_feats.get("crest_factor", 3.0))
    rms = float(time_feats.get("rms", 0.1))
    dom_freq = float(freq_feats.get("dominant_frequency_hz", 33.3))

    is_healthy = (prediction.upper() == "HEALTHY") and (confidence >= 0.5)

    if is_healthy:
        kb_entry = MAINTENANCE_KNOWLEDGE_BASE[4]  # Healthy Baseline
        diagnosis_title = "Healthy Bearing — Normal Operational Status"
        urgency = "LOW"
    else:
        # Determine specific fault signature based on acoustic / vibration features
        if kurtosis > 4.0 or crest > 4.5:
            # High impulsiveness -> Outer Race Defect (BPFO)
            kb_entry = MAINTENANCE_KNOWLEDGE_BASE[0]
            diagnosis_title = "Outer Race Fatigue Spalling (BPFO Signature Detected)"
            urgency = "CRITICAL"
        elif dom_freq > 1000 or rms > 0.4:
            # High frequency broadband friction -> Lubrication breakdown
            kb_entry = MAINTENANCE_KNOWLEDGE_BASE[2]
            diagnosis_title = "Severe Lubrication Degradation & Friction Shock"
            urgency = "HIGH"
        elif 30.0 <= dom_freq <= 70.0 and kurtosis <= 3.6:
            # Dominant shaft harmonic -> Unbalance / Misalignment
            kb_entry = MAINTENANCE_KNOWLEDGE_BASE[3]
            diagnosis_title = "Rotational Unbalance / Shaft Misalignment"
            urgency = "MEDIUM"
        else:
            # General bearing defect
            kb_entry = MAINTENANCE_KNOWLEDGE_BASE[0]
            diagnosis_title = "Bearing Rolling Contact Surface Fatigue"
            urgency = "CRITICAL"

    return {
        "advisory_id": f"ADV-{kb_entry['id']}",
        "primary_diagnosis": diagnosis_title,
        "category": kb_entry["category"],
        "severity_level": kb_entry["severity"],
        "urgency": urgency,
        "action_timeline": kb_entry["action_timeline"],
        "iso_standard_reference": kb_entry["iso_standard"],
        "diagnostic_evidence": {
            "prediction": prediction,
            "confidence_score": round(confidence, 4),
            "kurtosis": round(kurtosis, 2),
            "crest_factor": round(crest, 2),
            "rms_acceleration_g": round(rms, 4),
            "dominant_frequency_hz": round(dom_freq, 1),
            "iso_zone": iso_feats.get("zone", "Zone B" if is_healthy else "Zone D"),
        },
        "root_cause_hypotheses": kb_entry["root_causes"],
        "sop_action_plan": kb_entry["sop_steps"],
        "safety_notes": kb_entry["safety_notes"],
    }
