import re


# ============================================================
# MEDICAL SCOPE KEYWORDS
# ============================================================
# IMPORTANT:
# This list ONLY checks whether a query is potentially medical.
# It does NOT diagnose the patient and does NOT determine triage.
# Final medical reasoning is handled by MedGemma.
# ============================================================

MEDICAL_KEYWORDS = {

    # -------------------------
    # General symptoms
    # -------------------------
    "pain",
    "ache",
    "aching",
    "fever",
    "temperature",
    "chills",
    "fatigue",
    "tired",
    "weakness",
    "weak",
    "dizziness",
    "dizzy",
    "faint",
    "fainting",
    "unconscious",
    "nausea",
    "nauseous",
    "vomiting",
    "vomit",
    "diarrhea",
    "diarrhoea",
    "constipation",
    "bloating",
    "swelling",
    "inflammation",
    "infection",
    "symptom",
    "symptoms",
    "discomfort",
    "burning",
    "itching",
    "itchy",
    "cramps",
    "cramping",

    # -------------------------
    # Head / face
    # -------------------------
    "head",
    "headache",
    "migraine",
    "skull",
    "face",
    "facial",
    "forehead",
    "temple",
    "jaw",
    "jaw pain",
    "ear",
    "earache",
    "ear pain",
    "hearing",
    "ringing",
    "tinnitus",
    "eye",
    "eyes",
    "eye pain",
    "vision",
    "blurry",
    "blurred",
    "blind",
    "blindness",
    "red eye",

    # -------------------------
    # Mouth / throat
    # -------------------------
    "mouth",
    "tooth",
    "toothache",
    "teeth",
    "gum",
    "gums",
    "tongue",
    "throat",
    "sore throat",
    "swallow",
    "swallowing",
    "difficulty swallowing",
    "hoarse",
    "hoarseness",

    # -------------------------
    # Neck
    # -------------------------
    "neck",
    "neck pain",
    "stiff neck",
    "neck stiffness",

    # -------------------------
    # Chest / heart
    # -------------------------
    "chest",
    "chest pain",
    "chest tightness",
    "chest pressure",
    "heart",
    "heart pain",
    "palpitation",
    "palpitations",
    "heartbeat",
    "irregular heartbeat",
    "blood pressure",
    "hypertension",
    "hypotension",
    "high blood pressure",
    "low blood pressure",

    # -------------------------
    # Respiratory
    # -------------------------
    "breath",
    "breathe",
    "breathing",
    "breathlessness",
    "shortness",
    "shortness of breath",
    "difficulty breathing",
    "trouble breathing",
    "wheezing",
    "wheeze",
    "cough",
    "coughing",
    "dry cough",
    "phlegm",
    "sputum",
    "congestion",
    "nasal congestion",
    "runny nose",
    "blocked nose",
    "sneezing",
    "asthma",
    "pneumonia",
    "bronchitis",
    "tuberculosis",
    "tb",
    "oxygen",
    "spo2",
    "oxygen saturation",

    # -------------------------
    # Abdomen / digestive
    # -------------------------
    "stomach",
    "stomach pain",
    "abdominal",
    "abdomen",
    "abdominal pain",
    "belly",
    "belly pain",
    "gas",
    "gastric",
    "gastritis",
    "acidity",
    "acid reflux",
    "reflux",
    "heartburn",
    "indigestion",
    "ulcer",
    "ibs",
    "irritable bowel",
    "gallbladder",
    "gallstone",
    "gallstones",
    "liver",
    "pancreas",
    "pancreatitis",
    "appendix",
    "appendicitis",
    "rectal",
    "rectum",
    "anus",
    "stool",
    "black stool",
    "blood in stool",
    "constipation",
    "diarrhea",

    # -------------------------
    # Back / spine
    # -------------------------
    "back",
    "back pain",
    "lower back",
    "lower back pain",
    "upper back",
    "spine",
    "spinal",
    "spine pain",
    "slipped disc",
    "disc",
    "sciatica",
    "stiff back",

    # -------------------------
    # Arms / shoulders
    # -------------------------
    "arm",
    "arm pain",
    "shoulder",
    "shoulder pain",
    "elbow",
    "elbow pain",
    "wrist",
    "wrist pain",
    "hand",
    "hand pain",
    "finger",
    "finger pain",

    # -------------------------
    # Legs / knees / feet
    # -------------------------
    "leg",
    "leg pain",
    "legs",
    "pain in my leg",
    "pain in my legs",
    "leg ache",
    "leg swelling",
    "leg weakness",
    "thigh",
    "thigh pain",
    "calf",
    "calf pain",
    "calf swelling",
    "knee",
    "knee pain",
    "knee swelling",
    "ankle",
    "ankle pain",
    "ankle swelling",
    "foot",
    "foot pain",
    "feet",
    "foot swelling",
    "heel",
    "heel pain",
    "toe",
    "toe pain",
    "cramp",
    "leg cramp",
    "muscle cramp",
    "varicose veins",

    # -------------------------
    # Muscles / joints / bones
    # -------------------------
    "muscle",
    "muscle pain",
    "muscle ache",
    "joint",
    "joint pain",
    "joint swelling",
    "bone",
    "bone pain",
    "arthritis",
    "rheumatoid",
    "sprain",
    "strain",
    "injury",
    "fracture",
    "swelling",
    "stiffness",

    # -------------------------
    # Neurological
    # -------------------------
    "numbness",
    "numb",
    "tingling",
    "pins and needles",
    "neuropathy",
    "nerve",
    "nerve pain",
    "seizure",
    "seizures",
    "convulsion",
    "tremor",
    "trembling",
    "paralysis",
    "paralyzed",
    "weakness",
    "stroke",
    "speech",
    "slurred speech",
    "memory",
    "confusion",
    "disorientation",
    "balance",
    "coordination",

    # -------------------------
    # Skin
    # -------------------------
    "skin",
    "rash",
    "rashes",
    "itch",
    "itching",
    "itchy",
    "redness",
    "red skin",
    "dry skin",
    "eczema",
    "psoriasis",
    "acne",
    "pimple",
    "boil",
    "blister",
    "lesion",
    "wound",
    "cut",
    "burn",
    "burns",
    "infection",
    "hives",
    "urticaria",

    # -------------------------
    # Urinary / kidney
    # -------------------------
    "urine",
    "urination",
    "urinary",
    "kidney",
    "kidney pain",
    "uti",
    "urinary infection",
    "burning urination",
    "painful urination",
    "frequent urination",
    "blood in urine",
    "hematuria",
    "bladder",
    "bladder pain",
    "kidney stone",
    "kidney stones",

    # -------------------------
    # Reproductive / gynecological
    # -------------------------
    "period",
    "periods",
    "menstrual",
    "menstruation",
    "menstrual pain",
    "cramps",
    "pelvic",
    "pelvic pain",
    "ovary",
    "ovarian",
    "pcos",
    "pcod",
    "endometriosis",
    "pregnancy",
    "pregnant",
    "pregnancy pain",
    "vaginal",
    "vaginal bleeding",
    "vaginal discharge",
    "discharge",
    "fertility",
    "infertility",

    # -------------------------
    # Male reproductive
    # -------------------------
    "testicle",
    "testicular",
    "testicular pain",
    "scrotum",
    "penis",
    "prostate",
    "erectile",
    "erection",

    # -------------------------
    # Endocrine / metabolic
    # -------------------------
    "diabetes",
    "diabetic",
    "blood sugar",
    "sugar level",
    "glucose",
    "insulin",
    "thyroid",
    "hypothyroidism",
    "hyperthyroidism",
    "hormone",
    "hormonal",
    "pcos",
    "thirst",
    "excessive thirst",
    "weight loss",
    "weight gain",

    # -------------------------
    # Blood / hematology
    # -------------------------
    "anemia",
    "anaemia",
    "hemoglobin",
    "platelet",
    "platelets",
    "blood",
    "bleeding",
    "blood loss",
    "bruising",
    "bruise",

    # -------------------------
    # Allergies
    # -------------------------
    "allergy",
    "allergic",
    "allergies",
    "anaphylaxis",
    "hives",
    "swelling",
    "allergic reaction",

    # -------------------------
    # Infections
    # -------------------------
    "viral",
    "virus",
    "bacterial",
    "bacteria",
    "infection",
    "infected",
    "dengue",
    "malaria",
    "typhoid",
    "flu",
    "influenza",
    "covid",
    "coronavirus",
    "pneumonia",
    "pneumonia symptoms",
    "sepsis",

    # -------------------------
    # Medical measurements
    # -------------------------
    "spo2",
    "oxygen",
    "temperature",
    "fever",
    "blood pressure",
    "bp",
    "heart rate",
    "pulse",
    "pulse rate",
    "glucose",
    "blood sugar",
    "sugar level",
    "hemoglobin",
    "platelet",
    "bmi",

    # -------------------------
    # Medicines / healthcare
    # -------------------------
    "medicine",
    "medication",
    "tablet",
    "capsule",
    "syrup",
    "injection",
    "antibiotic",
    "antibiotics",
    "prescription",
    "dose",
    "dosage",
    "side effect",
    "side effects",
    "drug",
    "treatment",
    "therapy",
    "diagnosis",
    "doctor",
    "hospital",
    "clinic",
    "emergency",
    "ambulance",
    "specialist",
}


# ============================================================
# EMERGENCY PATTERNS
# ============================================================
# These are checked BEFORE the <3-word rule.
# They only allow the query to proceed.
# They do NOT themselves determine the final medical result.
# ============================================================

EMERGENCY_PATTERNS = [
    r"\bcan't breathe\b",
    r"\bcannot breathe\b",
    r"\bcan not breathe\b",
    r"\bcan't breath\b",
    r"\bcannot breath\b",

    r"\bsevere chest pain\b",
    r"\bchest pain\b",
    r"\bchest pressure\b",
    r"\bheart attack\b",

    r"\bnot breathing\b",
    r"\bsevere bleeding\b",
    r"\bheavy bleeding\b",

    r"\bunconscious\b",
    r"\bpassed out\b",
    r"\bpassing out\b",

    r"\bcannot speak\b",
    r"\bcan't speak\b",
    r"\bslurred speech\b",

    r"\bface.*weak\b",
    r"\bone side.*weak\b",
    r"\bone side.*numb\b",

    r"\bdifficulty breathing\b",
    r"\btrouble breathing\b",
    r"\bsevere shortness of breath\b",

    r"\blips.*swelling\b",
    r"\btongue.*swelling\b",

    r"\bsevere allergic reaction\b",
]


# ============================================================
# RESPONSE: NON-MEDICAL
# ============================================================

NON_MEDICAL_RESPONSE = {
    "status": "unable_to_determine",
    "triage": "Unable to determine",
    "possible_condition": "Unable to determine",
    "specialist": "General Physician",
    "reason": (
        "This query is outside the medical scope of NabhaCare."
    ),
    "safety_advice": (
        "Please ask a medical or health-related query."
    ),
}


# ============================================================
# RESPONSE: TOO SHORT / INSUFFICIENT INFORMATION
# ============================================================

SHORT_QUERY_RESPONSE = {
    "status": "unable_to_determine",
    "triage": "Unable to determine",
    "possible_condition": "Unable to determine",
    "specialist": "General Physician",
    "reason": (
        "The query does not contain enough information "
        "to assess the medical concern."
    ),
    "safety_advice": (
        "Please provide more details about your symptoms, "
        "duration, severity, and any relevant measurements."
    ),
}


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text: str) -> str:

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def word_count(text: str) -> int:

    return len(
        re.findall(
            r"\b[\w%°.-]+\b",
            text,
        )
    )


def contains_medical_term(text: str) -> bool:

    normalized = normalize_text(text)

    # First check complete phrases.
    for keyword in MEDICAL_KEYWORDS:

        if " " in keyword:

            if keyword in normalized:
                return True

    # Then individual words.
    words = set(
        re.findall(
            r"\b[\w%°.-]+\b",
            normalized,
        )
    )

    for keyword in MEDICAL_KEYWORDS:

        if " " not in keyword:

            if keyword in words:
                return True

    return False


def is_emergency_short_query(text: str) -> bool:

    normalized = normalize_text(text)

    for pattern in EMERGENCY_PATTERNS:

        if re.search(
            pattern,
            normalized,
        ):
            return True

    return False


# ============================================================
# MAIN VALIDATION
# ============================================================

def validate_medical_query(text: str) -> dict:

    """
    IMPORTANT:

    This function is ONLY a gate.

    It does NOT diagnose.
    It does NOT decide Critical / Moderate / Mild.
    It does NOT determine a disease.

    It only decides whether the query should proceed
    to the medical AI pipeline.
    """

    if not text or not text.strip():

        return {
            "allowed": False,
            "reason": "empty",
            "response": SHORT_QUERY_RESPONSE.copy(),
        }

    normalized = normalize_text(text)

    # --------------------------------------------------------
    # 1. Emergency phrases bypass the word-count restriction.
    # --------------------------------------------------------

    if is_emergency_short_query(normalized):

        return {
            "allowed": True,
            "reason": "emergency_bypass",
            "response": None,
        }

    # --------------------------------------------------------
    # 2. Scope check.
    #
    # If there is clearly no medical terminology, reject it.
    # This does NOT mean a medical diagnosis.
    # --------------------------------------------------------

    if not contains_medical_term(normalized):

        return {
            "allowed": False,
            "reason": "non_medical",
            "response": NON_MEDICAL_RESPONSE.copy(),
        }

    # --------------------------------------------------------
    # 3. Short-query rule.
    # --------------------------------------------------------

    if word_count(normalized) < 3:

        return {
            "allowed": False,
            "reason": "too_short",
            "response": SHORT_QUERY_RESPONSE.copy(),
        }

    # --------------------------------------------------------
    # 4. Valid medical query.
    #
    # IMPORTANT:
    # We DO NOT return a medical diagnosis here.
    # The query goes to MedGemma.
    # --------------------------------------------------------

    return {
        "allowed": True,
        "reason": "valid_medical_query",
        "response": None,
    }