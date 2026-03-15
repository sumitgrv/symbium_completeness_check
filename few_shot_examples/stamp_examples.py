# ruff: noqa: E501
import os

from dotenv import load_dotenv

load_dotenv()

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_stamp_dir = os.path.join(_project_root, "examples", "stamp")
stamp_img_dir = os.getenv("STAMP_EXAMPLES_PATH", _default_stamp_dir)

# Expected JSON for PE-only stamp (checkStampPresence, ProfessionalEngineeringStamp)
PE_YES = {"checkStampPresence": "Yes", "CheckStampType": [{"ProfessionalEngineeringStamp": "Yes"}]}
PE_NO = {"checkStampPresence": "No", "CheckStampType": [{"ProfessionalEngineeringStamp": "No"}]}

# Few-shot PE stamp exemplars (PE-only check; City stamps are not counted as stamp presence)
STAMP_EXAMPLES = [
    {
        "image_path": os.path.join(stamp_img_dir, "stamp1.png"),
        "description": "Professional Engineer stamp with state - 'STATE OF UTAH' and profession - 'PROFESSIONAL ENGINNER' on outer ring",
        "expected_response": PE_YES,
    },
    {
        "image_path": os.path.join(stamp_img_dir, "stamp2.jpg"),
        "description": "PE Stamp having circular seal with license number & exp inside the circular ring, printed state - 'STATE OF NEW JERSEY' & profession - 'PROFESSIONAL ENGINNER' on outer ring",
        "expected_response": PE_YES,
    },
    {
        "image_path": os.path.join(stamp_img_dir, "stamp3.jpg"),
        "description": "Valid circular approval mark used for civil engineering license",
        "expected_response": PE_YES,
    },
    {
        "image_path": os.path.join(stamp_img_dir, "stamp6.jpg"),
        "description": "Example of a PE stamp with discipline - 'CIVIL' inside the circular ring and printed state - 'STATE OF CALIFORNIA' & profession - 'LICENSED PROFESSIONAL ENGINNER' on outer ring",
        "expected_response": PE_YES,
    },
    {
        "image_path": os.path.join(stamp_img_dir, "stamp7.jpg"),
        "description": "Example of a rectangular PE certification block with jurisdiction — 'STATE OF MARYLAND' — and profession — 'LICENSED PROFESSIONAL ENGINEER' — mentioned in the body of the text. Includes fields for License Number and Expiration Date, typically filled manually",
        "expected_response": PE_YES,
    },
    {
        "image_path": os.path.join(stamp_img_dir, "stamp8.jpg"),
        "description": "Example of a square PE stamp featuring the state emblem of Rhode Island with text indicating the jurisdiction — 'STATE OF RHODE ISLAND' — embedded within the seal. The stamp includes the engineer's name ('JOHN DOE'), license number ('00000'), and the discipline — 'CIVIL' — listed at the bottom. The title 'REGISTERED PROFESSIONAL ENGINEER' is printed prominently in uppercase, confirming licensure",
        "expected_response": PE_YES,
    },
    {
        "image_path": os.path.join(stamp_img_dir, "stamp9.jpg"),
        "description": "Example of a rectangular engineering firm identification block used in Texas. It includes the firm name ('John Doe Engineering') and indicates registration as a Texas Registered Engineering Firm with a firm number (e.g., 'F-00000')",
        "expected_response": PE_NO,
    },
    {
        "image_path": os.path.join(stamp_img_dir, "stamp10.png"),
        "description": "City approval stamp with 'APPROVED PLANS' and central date field",
        "expected_response": PE_NO,
    },
    {
        "image_path": os.path.join(stamp_img_dir, "stamp11.jpg"),
        "description": "City Stamp - Fire department conditional approval block with bold header and date",
        "expected_response": PE_NO,
    },
    {
        "image_path": os.path.join(stamp_img_dir, "stamp12.jpg"),
        "description": "City Stamp - County 'APPROVED FOR ISSUANCE' job‑copy notice with date and signer",
        "expected_response": PE_NO,
    },
    {
        "image_path": os.path.join(stamp_img_dir, "stamp14.jpg"),
        "description": "City Stamp with municipal routing table stamp with 'ACCEPTED' statuses and received dates",
        "expected_response": PE_NO,
    },
    {
        "image_path": os.path.join(stamp_img_dir, "stamp15.jpg"),
        "description": "City Stamp - City building approvals block with permit number, received mark, and reviewer/date",
        "expected_response": PE_NO,
    },
]
