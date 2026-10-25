"""Timing specification for the eICU Collaborative Research Database (version 2.0).

All times are offsets in minutes from the start of the ICU unit stay. Column
names are handled in lower case.
"""

from __future__ import annotations

from .base import DICTIONARY, FIRST_DAY, STATIC, TIMED, DatasetSchema, TableSpec

EICU = DatasetSchema(
    name="eicu",
    time_kind="minutes",
    patient_id="uniquepid",
    encounter_id="patienthealthsystemstayid",
    stay_id="patientunitstayid",
    site_id="hospitalid",
    first_day_minutes=1440.0,
    notes=(
        "Offsets are minutes from unit admission (negative before it). hospitaldischargeyear "
        "is the only calendar variable and supports temporal ordering at year resolution."
    ),
)

_ID = ("patientunitstayid",)
_E = "patientunitstayid"

EICU.add(TableSpec(
    name="patient", timing=STATIC,
    description="One row per unit stay with demographics, admission and discharge details.",
    id_columns=("patientunitstayid", "patienthealthsystemstayid", "uniquepid", "hospitalid", "wardid"),
    entity_column=_E, event_time="hospitaladmitoffset",
    hindsight_columns=(
        "hospitaldischargeoffset", "hospitaldischargelocation", "hospitaldischargetime24",
        "unitdischargeoffset", "unitdischargelocation", "unitdischargetime24", "dischargeweight",
    ),
    outcome_columns=("hospitaldischargestatus", "unitdischargestatus"),
    calendar_columns=("hospitaldischargeyear",),
    notes="Discharge fields are known only at discharge; discharge status encodes death.",
))
EICU.add(TableSpec(
    name="hospital", timing=STATIC,
    description="Hospital characteristics.",
    id_columns=("hospitalid",), entity_column="hospitalid",
))
EICU.add(TableSpec(
    name="vitalPeriodic", timing=TIMED,
    description="Automatically captured vital signs, usually at 5-minute resolution.",
    id_columns=("vitalperiodicid",) + _ID, entity_column=_E,
    event_time="observationoffset",
    wide_value_columns=(
        "temperature", "sao2", "heartrate", "respiration", "cvp", "etco2",
        "systemicsystolic", "systemicdiastolic", "systemicmean",
        "pasystolic", "padiastolic", "pamean", "st1", "st2", "st3", "icp",
    ),
))
EICU.add(TableSpec(
    name="vitalAperiodic", timing=TIMED,
    description="Intermittent vital signs such as non-invasive blood pressure.",
    id_columns=("vitalaperiodicid",) + _ID, entity_column=_E,
    event_time="observationoffset",
    wide_value_columns=(
        "noninvasivesystolic", "noninvasivediastolic", "noninvasivemean", "paop",
        "cardiacoutput", "cardiacinput", "svr", "svri", "pvr", "pvri",
    ),
))
EICU.add(TableSpec(
    name="lab", timing=TIMED,
    description="Laboratory results.",
    id_columns=("labid",) + _ID, entity_column=_E,
    event_time="labresultoffset", available_time="labresultrevisedoffset",
    variable_column="labname", value_column="labresult",
    notes="labresultrevisedoffset marks when a revised value was entered.",
))
EICU.add(TableSpec(
    name="customLab", timing=TIMED,
    description="Site-specific laboratory results.",
    id_columns=("customlabid",) + _ID, entity_column=_E,
    event_time="labotheroffset", variable_column="labothername", value_column="labotherresult",
))
EICU.add(TableSpec(
    name="microLab", timing=TIMED,
    description="Microbiology cultures and results.",
    id_columns=("microlabid",) + _ID, entity_column=_E,
    event_time="culturetakenoffset", variable_column="culturesite", value_column="organism",
    hindsight_columns=("organism", "antibiotic", "sensitivitylevel"),
    notes="Only the culture time is recorded; organisms are reported later at an unrecorded time.",
))
EICU.add(TableSpec(
    name="medication", timing=TIMED,
    description="Medication orders.",
    id_columns=("medicationid",) + _ID, entity_column=_E,
    event_time="drugstartoffset", available_time="drugorderoffset", end_time="drugstopoffset",
    variable_column="drugname", value_column="dosage",
    hindsight_columns=("drugordercancelled",),
))
EICU.add(TableSpec(
    name="infusionDrug", timing=TIMED,
    description="Continuous infusions.",
    id_columns=("infusiondrugid",) + _ID, entity_column=_E,
    event_time="infusionoffset", variable_column="drugname", value_column="drugrate",
))
EICU.add(TableSpec(
    name="intakeOutput", timing=TIMED,
    description="Fluid intake and output.",
    id_columns=("intakeoutputid",) + _ID, entity_column=_E,
    event_time="intakeoutputoffset", available_time="intakeoutputentryoffset",
    variable_column="celllabel", value_column="cellvaluenumeric",
))
EICU.add(TableSpec(
    name="nurseCharting", timing=TIMED,
    description="Nurse-charted observations including scores and vitals.",
    id_columns=("nursingchartid",) + _ID, entity_column=_E,
    event_time="nursingchartoffset", available_time="nursingchartentryoffset",
    variable_column="nursingchartcelltypevalname", value_column="nursingchartvalue",
))
EICU.add(TableSpec(
    name="nurseAssessment", timing=TIMED,
    description="Nursing assessments.",
    id_columns=("nurseassessid",) + _ID, entity_column=_E,
    event_time="nurseassessoffset", available_time="nurseassessentryoffset",
    variable_column="celllabel", value_column="cellattributevalue",
))
EICU.add(TableSpec(
    name="nurseCare", timing=TIMED,
    description="Nursing care activities.",
    id_columns=("nursecareid",) + _ID, entity_column=_E,
    event_time="nursecareoffset", available_time="nursecareentryoffset",
    variable_column="celllabel", value_column="cellattributevalue",
))
EICU.add(TableSpec(
    name="respiratoryCharting", timing=TIMED,
    description="Respiratory therapy charting.",
    id_columns=("respchartid",) + _ID, entity_column=_E,
    event_time="respchartoffset", available_time="respchartentryoffset",
    variable_column="respchartvaluelabel", value_column="respchartvalue",
))
EICU.add(TableSpec(
    name="respiratoryCare", timing=TIMED,
    description="Airway and ventilation episodes.",
    id_columns=("respcareid",) + _ID, entity_column=_E,
    event_time="respcarestatusoffset", end_time="ventendoffset",
    variable_column="airwaytype",
    notes="ventstartoffset and ventendoffset describe ventilation episodes that may extend past the prediction time.",
))
EICU.add(TableSpec(
    name="physicalExam", timing=TIMED,
    description="Structured physical examination.",
    id_columns=("physicalexamid",) + _ID, entity_column=_E,
    event_time="physicalexamoffset", variable_column="physicalexampath", value_column="physicalexamvalue",
))
EICU.add(TableSpec(
    name="pastHistory", timing=TIMED,
    description="Past medical history.",
    id_columns=("pasthistoryid",) + _ID, entity_column=_E,
    event_time="pasthistoryoffset", available_time="pasthistoryenteredoffset",
    variable_column="pasthistorypath", value_column="pasthistoryvalue",
))
EICU.add(TableSpec(
    name="admissionDx", timing=TIMED,
    description="Admission diagnosis captured for APACHE.",
    id_columns=("admissiondxid",) + _ID, entity_column=_E,
    event_time="admitdxenteredoffset", variable_column="admitdxpath", value_column="admitdxname",
))
EICU.add(TableSpec(
    name="diagnosis", timing=TIMED,
    description="Problem-list diagnoses entered during the stay.",
    id_columns=("diagnosisid",) + _ID, entity_column=_E,
    event_time="diagnosisoffset", variable_column="diagnosisstring", value_column="icd9code",
    code_columns=("icd9code",),
    hindsight_columns=("activeupondischarge",),
    proxy_text_columns=("diagnosisstring",),
    notes="activeupondischarge is only known at discharge; diagnoses can be entered late in the stay.",
))
EICU.add(TableSpec(
    name="treatment", timing=TIMED,
    description="Treatments documented during the stay.",
    id_columns=("treatmentid",) + _ID, entity_column=_E,
    event_time="treatmentoffset", variable_column="treatmentstring",
    hindsight_columns=("activeupondischarge",),
    proxy_text_columns=("treatmentstring",),
))
EICU.add(TableSpec(
    name="admissionDrug", timing=TIMED,
    description="Medications taken before admission.",
    id_columns=("admissiondrugid",) + _ID, entity_column=_E,
    event_time="drugoffset", available_time="drugenteredoffset",
    variable_column="drugname", value_column="drugdosage",
))
EICU.add(TableSpec(
    name="allergy", timing=TIMED,
    description="Allergies.",
    id_columns=("allergyid",) + _ID, entity_column=_E,
    event_time="allergyoffset", available_time="allergyenteredoffset",
    variable_column="allergyname",
))
EICU.add(TableSpec(
    name="note", timing=TIMED,
    description="Structured note fields.",
    id_columns=("noteid",) + _ID, entity_column=_E,
    event_time="noteoffset", available_time="noteenteredoffset",
    variable_column="notepath", value_column="notevalue",
    proxy_text_columns=("notepath", "notevalue", "notetext"),
))
EICU.add(TableSpec(
    name="carePlanGeneral", timing=TIMED,
    description="Care plan items including care limitations and code status.",
    id_columns=("cplgeneralid",) + _ID, entity_column=_E,
    event_time="cplitemoffset", variable_column="cplgroup", value_column="cplitemvalue",
    hindsight_columns=("activeupondischarge",),
    proxy_text_columns=("cplgroup", "cplitemvalue"),
    notes="The 'Care Limitation' group records do-not-resuscitate and comfort-care decisions.",
))
EICU.add(TableSpec(
    name="carePlanEOL", timing=TIMED,
    description="End-of-life discussions.",
    id_columns=("cpleolid",) + _ID, entity_column=_E,
    event_time="cpleolsaveoffset", event_time_fallback="cpleoldiscussionoffset",
    hindsight_columns=("activeupondischarge",),
    whole_table_is_proxy=True,
    notes="Any record in this table documents an end-of-life discussion.",
))
EICU.add(TableSpec(
    name="carePlanGoal", timing=TIMED,
    description="Care plan goals.",
    id_columns=("cplgoalid",) + _ID, entity_column=_E,
    event_time="cplgoaloffset", variable_column="cplgoalcategory", value_column="cplgoalvalue",
    hindsight_columns=("activeupondischarge", "cplgoalstatus"),
    proxy_text_columns=("cplgoalvalue",),
))
EICU.add(TableSpec(
    name="carePlanCareProvider", timing=TIMED,
    description="Care providers involved in the stay.",
    id_columns=("cplcareprovderid",) + _ID, entity_column=_E,
    event_time="careprovidersaveoffset", variable_column="specialty",
    hindsight_columns=("activeupondischarge",),
    proxy_text_columns=("specialty", "providertype"),
))
EICU.add(TableSpec(
    name="carePlanInfectiousDisease", timing=TIMED,
    description="Infectious disease care plan.",
    id_columns=("cplinfectid",) + _ID, entity_column=_E,
    event_time="cplinfectdiseaseoffset", variable_column="infectdiseasesite",
    hindsight_columns=("activeupondischarge", "responsetotherapy"),
))
EICU.add(TableSpec(
    name="apacheApsVar", timing=FIRST_DAY,
    description="Worst APACHE physiology values over the first 24 hours of the unit stay.",
    id_columns=("apacheapsvarid",) + _ID, entity_column=_E,
    first_day_columns=(
        "intubated", "vent", "dialysis", "eyes", "motor", "verbal", "meds", "urine", "wbc",
        "temperature", "respiratoryrate", "sodium", "heartrate", "meanbp", "ph", "hematocrit",
        "creatinine", "albumin", "pao2", "pco2", "bun", "glucose", "bilirubin", "fio2",
    ),
    notes="Values summarise the first day and are complete only 24 hours after unit admission.",
))
EICU.add(TableSpec(
    name="apachePredVar", timing=FIRST_DAY,
    description="Variables used by the APACHE IV prediction equations.",
    id_columns=("apachepredvarid",) + _ID, entity_column=_E,
    first_day_columns=(
        "sicuday", "saps3day1", "saps3today", "saps3yesterday", "meds", "verbal", "motor", "eyes",
        "thrombolytics", "ventday1", "oobventday1", "oobintubday1", "pao2", "fio2", "creatinine",
        "day1meds", "day1verbal", "day1motor", "day1eyes", "day1pao2", "day1fio2",
    ),
    outcome_columns=("diedinhospital",),
    hindsight_columns=("dischargelocation",),
    notes="diedinhospital is the hospital mortality outcome stored alongside the predictors.",
))
EICU.add(TableSpec(
    name="apachePatientResult", timing=FIRST_DAY,
    description="APACHE scores, predictions and observed outcomes.",
    id_columns=("apachepatientresultsid",) + _ID, entity_column=_E,
    first_day_columns=(
        "acutephysiologyscore", "apachescore", "predictedicumortality", "predictediculos",
        "predictedhospitalmortality", "predictedhospitallos", "predventdays",
    ),
    outcome_columns=("actualicumortality", "actualhospitalmortality"),
    hindsight_columns=(
        "actualiculos", "actualhospitallos", "unabridgedunitlos", "unabridgedhosplos",
        "actualventdays", "unabridgedactualventdays",
    ),
    notes="actual* and unabridged* columns are observed outcomes, not predictors.",
))
