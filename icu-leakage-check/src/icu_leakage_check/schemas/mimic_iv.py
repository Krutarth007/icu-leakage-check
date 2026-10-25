"""Timing specification for MIMIC-IV (hosp and icu modules, versions 2.x and 3.x)."""

from __future__ import annotations

from .base import (
    DATE_ONLY,
    DICTIONARY,
    LINKED,
    STATIC,
    TIMED,
    UNTIMED,
    DatasetSchema,
    TableSpec,
)

MIMIC_IV = DatasetSchema(
    name="mimic-iv",
    time_kind="datetime",
    patient_id="subject_id",
    encounter_id="hadm_id",
    stay_id="stay_id",
    site_id=None,
    notes=(
        "Dates are shifted per patient into the future, so absolute dates cannot be compared "
        "across patients. anchor_year_group in the patients table gives the true three-year "
        "period and is the only valid basis for temporal ordering across patients."
    ),
)

_H = "hosp"
_I = "icu"

# ----------------------------------------------------------------------- hosp
MIMIC_IV.add(TableSpec(
    name="admissions", module=_H, timing=STATIC,
    description="One row per hospital admission.",
    id_columns=("subject_id", "hadm_id"), entity_column="hadm_id",
    event_time="admittime",
    datetime_columns=("admittime", "dischtime", "deathtime", "edregtime", "edouttime"),
    hindsight_columns=("dischtime", "discharge_location"),
    outcome_columns=("deathtime", "hospital_expire_flag"),
    notes="discharge_location includes DIED and HOSPICE, so it encodes the outcome.",
))
MIMIC_IV.add(TableSpec(
    name="patients", module=_H, timing=STATIC,
    description="One row per patient.",
    id_columns=("subject_id",), entity_column="subject_id",
    hindsight_columns=(),
    outcome_columns=("dod",),
    calendar_columns=("anchor_year", "anchor_year_group"),
    notes="dod includes out-of-hospital deaths up to one year after the last discharge.",
))
MIMIC_IV.add(TableSpec(
    name="transfers", module=_H, timing=TIMED,
    description="Ward and unit movements during the admission.",
    id_columns=("subject_id", "hadm_id", "transfer_id"), entity_column="hadm_id",
    event_time="intime", end_time="outtime",
    variable_column="careunit", value_column="eventtype",
    datetime_columns=("intime", "outtime"),
    notes="Rows with eventtype 'discharge' are only known at discharge.",
))
MIMIC_IV.add(TableSpec(
    name="labevents", module=_H, timing=TIMED,
    description="Laboratory measurements.",
    id_columns=("labevent_id", "subject_id", "hadm_id", "specimen_id"), entity_column="hadm_id",
    event_time="charttime", available_time="storetime",
    variable_column="itemid", value_column="valuenum",
    datetime_columns=("charttime", "storetime"),
    notes="charttime is close to specimen collection; storetime is when the result was available.",
))
MIMIC_IV.add(TableSpec(
    name="microbiologyevents", module=_H, timing=TIMED,
    description="Cultures, organisms and susceptibilities.",
    id_columns=("microevent_id", "subject_id", "hadm_id", "micro_specimen_id"), entity_column="hadm_id",
    event_time="charttime", event_time_fallback="chartdate",
    available_time="storetime", available_time_fallback="storedate",
    variable_column="test_name", value_column="org_name",
    datetime_columns=("chartdate", "charttime", "storedate", "storetime"),
    notes="Organism and susceptibility results are stored hours to days after the specimen time.",
))
MIMIC_IV.add(TableSpec(
    name="diagnoses_icd", module=_H, timing=UNTIMED,
    description="Billing diagnoses for the admission.",
    id_columns=("subject_id", "hadm_id"), entity_column="hadm_id",
    variable_column="icd_code", code_columns=("icd_code",),
    hindsight_columns=("icd_code", "seq_num"),
    notes="Assigned after discharge for billing; not available during the same admission.",
))
MIMIC_IV.add(TableSpec(
    name="procedures_icd", module=_H, timing=DATE_ONLY,
    description="Billed procedures with the date performed.",
    id_columns=("subject_id", "hadm_id"), entity_column="hadm_id",
    event_time="chartdate", variable_column="icd_code", code_columns=("icd_code",),
    datetime_columns=("chartdate",),
    hindsight_columns=("seq_num",),
    notes="Coded after discharge; chartdate has day resolution only.",
))
MIMIC_IV.add(TableSpec(
    name="drgcodes", module=_H, timing=UNTIMED,
    description="Diagnosis-related group codes used for billing.",
    id_columns=("subject_id", "hadm_id"), entity_column="hadm_id",
    variable_column="drg_code",
    hindsight_columns=("drg_type", "drg_code", "description", "drg_severity", "drg_mortality"),
    notes="Assigned after discharge; drg_mortality is a risk-of-mortality grade.",
))
MIMIC_IV.add(TableSpec(
    name="hcpcsevents", module=_H, timing=DATE_ONLY,
    description="Billed HCPCS events.",
    id_columns=("subject_id", "hadm_id"), entity_column="hadm_id",
    event_time="chartdate", variable_column="hcpcs_cd", code_columns=("hcpcs_cd",),
    datetime_columns=("chartdate",),
    notes="Coded for billing; day resolution only.",
))
MIMIC_IV.add(TableSpec(
    name="prescriptions", module=_H, timing=TIMED,
    description="Medication prescriptions.",
    id_columns=("subject_id", "hadm_id", "pharmacy_id", "poe_id"), entity_column="hadm_id",
    event_time="starttime", end_time="stoptime",
    variable_column="drug", value_column="dose_val_rx",
    datetime_columns=("starttime", "stoptime"),
))
MIMIC_IV.add(TableSpec(
    name="pharmacy", module=_H, timing=TIMED,
    description="Pharmacy orders.",
    id_columns=("subject_id", "hadm_id", "pharmacy_id", "poe_id"), entity_column="hadm_id",
    event_time="starttime", available_time="entertime", end_time="stoptime",
    variable_column="medication", value_column="status",
    datetime_columns=("starttime", "stoptime", "entertime", "verifiedtime", "expirationdate"),
    hindsight_columns=("status",),
    notes="status reflects the final state of the order.",
))
MIMIC_IV.add(TableSpec(
    name="emar", module=_H, timing=TIMED,
    description="Electronic medication administration records.",
    id_columns=("subject_id", "hadm_id", "emar_id", "poe_id", "pharmacy_id"), entity_column="hadm_id",
    event_time="charttime", available_time="storetime",
    variable_column="medication", value_column="event_txt",
    datetime_columns=("charttime", "scheduletime", "storetime"),
))
MIMIC_IV.add(TableSpec(
    name="emar_detail", module=_H, timing=LINKED,
    description="Administration details; time comes from emar via emar_id.",
    id_columns=("subject_id", "emar_id", "pharmacy_id"), entity_column="subject_id",
    variable_column="product_description", value_column="dose_given",
))
MIMIC_IV.add(TableSpec(
    name="poe", module=_H, timing=TIMED,
    description="Provider order entry.",
    id_columns=("poe_id", "subject_id", "hadm_id"), entity_column="hadm_id",
    event_time="ordertime", variable_column="order_type", value_column="order_subtype",
    datetime_columns=("ordertime",),
    hindsight_columns=("order_status", "discontinued_by_poe_id"),
    notes="order_status and discontinued_by_poe_id reflect what happened after the order.",
))
MIMIC_IV.add(TableSpec(
    name="poe_detail", module=_H, timing=LINKED,
    description="Order details; time comes from poe via poe_id.",
    id_columns=("poe_id", "subject_id"), entity_column="subject_id",
    variable_column="field_name", value_column="field_value",
    proxy_text_columns=("field_value",),
))
MIMIC_IV.add(TableSpec(
    name="omr", module=_H, timing=DATE_ONLY,
    description="Online medical record measurements such as weight, height and blood pressure.",
    id_columns=("subject_id",), entity_column="subject_id",
    event_time="chartdate", variable_column="result_name", value_column="result_value",
    datetime_columns=("chartdate",),
    notes="Mostly outpatient; day resolution only, so same-day values may postdate the prediction.",
))
MIMIC_IV.add(TableSpec(
    name="services", module=_H, timing=TIMED,
    description="Clinical service changes.",
    id_columns=("subject_id", "hadm_id"), entity_column="hadm_id",
    event_time="transfertime", variable_column="curr_service",
    datetime_columns=("transfertime",),
))
for _name, _desc in (
    ("provider", "Provider identifiers."),
    ("d_hcpcs", "HCPCS dictionary."),
    ("d_icd_diagnoses", "ICD diagnosis dictionary."),
    ("d_icd_procedures", "ICD procedure dictionary."),
    ("d_labitems", "Laboratory item dictionary."),
):
    MIMIC_IV.add(TableSpec(name=_name, module=_H, timing=DICTIONARY, description=_desc))

# ------------------------------------------------------------------------ icu
MIMIC_IV.add(TableSpec(
    name="icustays", module=_I, timing=STATIC,
    description="One row per ICU stay.",
    id_columns=("subject_id", "hadm_id", "stay_id"), entity_column="stay_id",
    event_time="intime",
    datetime_columns=("intime", "outtime"),
    hindsight_columns=("outtime", "los", "last_careunit"),
    notes="los and outtime are only known when the patient leaves the unit.",
))
MIMIC_IV.add(TableSpec(
    name="chartevents", module=_I, timing=TIMED,
    description="Charted observations in the ICU.",
    id_columns=("subject_id", "hadm_id", "stay_id"), entity_column="stay_id",
    event_time="charttime", available_time="storetime",
    variable_column="itemid", value_column="valuenum",
    datetime_columns=("charttime", "storetime"),
    proxy_text_columns=("value",),
    notes="Contains code status and goals-of-care items; see find_proxy_items().",
))
MIMIC_IV.add(TableSpec(
    name="datetimeevents", module=_I, timing=TIMED,
    description="Charted date-valued observations.",
    id_columns=("subject_id", "hadm_id", "stay_id"), entity_column="stay_id",
    event_time="charttime", available_time="storetime",
    variable_column="itemid", value_column="value",
    datetime_columns=("charttime", "storetime", "value"),
))
MIMIC_IV.add(TableSpec(
    name="inputevents", module=_I, timing=TIMED,
    description="Fluids and medications given in the ICU.",
    id_columns=("subject_id", "hadm_id", "stay_id", "orderid", "linkorderid"), entity_column="stay_id",
    event_time="starttime", available_time="storetime", end_time="endtime",
    variable_column="itemid", value_column="amount",
    datetime_columns=("starttime", "endtime", "storetime"),
    hindsight_columns=("statusdescription",),
    notes="amount and totalamount cover the whole interval up to endtime.",
))
MIMIC_IV.add(TableSpec(
    name="ingredientevents", module=_I, timing=TIMED,
    description="Ingredients of ICU inputs.",
    id_columns=("subject_id", "hadm_id", "stay_id", "orderid", "linkorderid"), entity_column="stay_id",
    event_time="starttime", available_time="storetime", end_time="endtime",
    variable_column="itemid", value_column="amount",
    datetime_columns=("starttime", "endtime", "storetime"),
    hindsight_columns=("statusdescription",),
))
MIMIC_IV.add(TableSpec(
    name="outputevents", module=_I, timing=TIMED,
    description="Patient outputs such as urine.",
    id_columns=("subject_id", "hadm_id", "stay_id"), entity_column="stay_id",
    event_time="charttime", available_time="storetime",
    variable_column="itemid", value_column="value",
    datetime_columns=("charttime", "storetime"),
))
MIMIC_IV.add(TableSpec(
    name="procedureevents", module=_I, timing=TIMED,
    description="ICU procedures such as ventilation, lines and dialysis.",
    id_columns=("subject_id", "hadm_id", "stay_id", "orderid", "linkorderid"), entity_column="stay_id",
    event_time="starttime", available_time="storetime", end_time="endtime",
    variable_column="itemid", value_column="value",
    datetime_columns=("starttime", "endtime", "storetime"),
    hindsight_columns=("statusdescription",),
    notes="value is often a duration that runs to endtime.",
))
MIMIC_IV.add(TableSpec(name="d_items", module=_I, timing=DICTIONARY, description="ICU item dictionary."))
MIMIC_IV.add(TableSpec(name="caregiver", module=_I, timing=DICTIONARY, description="Caregiver identifiers."))
