"""Hand-written reason phrases for common (feature, value) pairs.

Features in this dataset are often pre-binned strings, so phrases are coarse
attribution labels — not causal explanations.
"""

from __future__ import annotations

# (feature, value_as_str) -> short reason phrase
REASON_PHRASES: dict[tuple[str, str], str] = {
    ("Fault", "Policy Holder"): "Policy holder at fault",
    ("Fault", "Third Party"): "Third party at fault",
    ("BasePolicy", "Liability"): "Liability base policy",
    ("BasePolicy", "Collision"): "Collision base policy",
    ("BasePolicy", "All Perils"): "All-perils base policy",
    ("AccidentArea", "Urban"): "Urban accident area",
    ("AccidentArea", "Rural"): "Rural accident area",
    ("PoliceReportFiled", "No"): "No police report filed",
    ("PoliceReportFiled", "Yes"): "Police report filed",
    ("WitnessPresent", "No"): "No witness present",
    ("WitnessPresent", "Yes"): "Witness present",
    ("AgentType", "External"): "External agent",
    ("AgentType", "Internal"): "Internal agent",
    ("Sex", "Male"): "Male policy holder",
    ("Sex", "Female"): "Female policy holder",
    ("VehicleCategory", "Sedan"): "Sedan vehicle category",
    ("VehicleCategory", "Sport"): "Sport vehicle category",
    ("VehicleCategory", "Utility"): "Utility vehicle category",
    ("PastNumberOfClaims", "none"): "No past claims",
    ("PastNumberOfClaims", "1"): "One past claim",
    ("PastNumberOfClaims", "2 to 4"): "Several past claims (2–4)",
    ("PastNumberOfClaims", "more than 4"): "Many past claims (>4)",
    ("AddressChange_Claim", "no change"): "No address change near claim",
    ("AddressChange_Claim", "under 6 months"): "Address changed within 6 months of claim",
    ("AddressChange_Claim", "1 year"): "Address changed about 1 year before claim",
    ("AddressChange_Claim", "2 to 3 years"): "Address changed 2–3 years before claim",
    ("AddressChange_Claim", "4 to 8 years"): "Address changed 4–8 years before claim",
    ("NumberOfSuppliments", "none"): "No supplements",
    ("NumberOfSuppliments", "more than 5"): "Many supplements (>5)",
    ("AgeOfVehicle", "new"): "New vehicle",
    ("AgeOfVehicle", "2 years"): "Vehicle about 2 years old",
    ("AgeOfVehicle", "3 years"): "Vehicle about 3 years old",
    ("AgeOfVehicle", "4 years"): "Vehicle about 4 years old",
    ("AgeOfVehicle", "5 years"): "Vehicle about 5 years old",
    ("AgeOfVehicle", "6 years"): "Vehicle about 6 years old",
    ("AgeOfVehicle", "7 years"): "Vehicle about 7 years old",
    ("AgeOfVehicle", "more than 7"): "Vehicle older than 7 years",
    ("Days_Policy_Accident", "none"): "No days-policy-accident bin (none)",
    ("Days_Policy_Claim", "none"): "No days-policy-claim bin (none)",
    ("Days_Policy_Accident", "1 to 7"): "Accident within 1–7 days of policy",
    ("Days_Policy_Claim", "1 to 7"): "Claim within 1–7 days of policy",
    ("DriverRating", "1"): "Driver rating 1",
    ("DriverRating", "4"): "Driver rating 4",
    ("Deductible", "400"): "Deductible 400",
    ("Deductible", "500"): "Deductible 500",
    ("Deductible", "700"): "Deductible 700",
    ("MaritalStatus", "Single"): "Single marital status",
    ("MaritalStatus", "Married"): "Married marital status",
    ("NumberOfCars", "1 vehicle"): "One vehicle on policy",
    ("NumberOfCars", "2 vehicles"): "Two vehicles on policy",
    ("NumberOfCars", "3 to 4"): "3–4 vehicles on policy",
    ("NumberOfCars", "5 to 8"): "5–8 vehicles on policy",
    ("NumberOfCars", "more than 8"): "More than 8 vehicles on policy",
}


def reason_for(feature: str, value) -> str:
    """Map (feature, value) to a phrase; fall back to 'feature = value'."""
    key = (str(feature), str(value))
    if key in REASON_PHRASES:
        return REASON_PHRASES[key]
    return f"{feature} = {value}"
