# Technical Documentation: Completeness-Check Approaches

## 1) Objective

The solution detects two plan-sheet completeness signals:
- PE stamp presence
- North direction symbol presence

Two model strategies are used:
- Fine-tuned vision LLM
- Prompt-based standard vision LLM

Both strategies process sheet images and return structured outputs for downstream completeness checks.

---

## 2) Standard LLM Approach (Prompt-Based)

### 2.1 Method Summary

The standard approach uses direct multimodal prompting against a base vision-capable model.  
It relies on:
- detailed rule-based instructions
- explicit output schema
- few-shot visual examples

This method does not require model training and can be iterated quickly by refining prompt definitions.

### 2.2 Prompting Architecture

Each detection task is evaluated with a prompt stack of:
1. **System message** (task role and strict behavior)
2. **Instruction block** (domain definitions, exclusions, verification rules)
3. **Few-shot examples** (image + expected JSON response)
4. **Target image message** (actual sheet image to evaluate)

### 2.3 Detection Tasks

Two independent tasks are run:
- **Stamp detection** (current logic: PE-only)
- **North-direction detection**

The outputs are combined into a single prediction object for reporting.

### 2.4 Few-Shot Usage

Few-shot examples are paired as:
- user: example image + descriptive text
- assistant: expected JSON output

This grounds model behavior on valid visual patterns and helps reduce false positives.

### 2.5 Prompt Template (Standard LLM)


#### Stamp Detection Description

```text
### Definition of valid stamp (PE only)

#### Professional Engineer (PE) stamps
A **Professional Engineer (PE) stamp** demonstrates that a professional engineer placed his/her "registration seal" on the drawing or designs. These typically appear as an official seal or approval mark used for professional certification, company validation, or approval.

- Common Textual Elements found in professional engineer stamps:
    - **Profession Title**: Phrases like "LICENSED ENGINNER" or "PROFESSIONAL ENGINNER" or "LICENSED PROFESSIONAL ENGINNER".
    - **State**: A U.S. state or Canadian province, such as "STATE OF CALIFORNIA" or "STATE OF NEW YORK".
    - **License number**
    - **Expiration date**: e.g., "EXP 03/25")
    - **Descipline**: The engineer's field of specialization, such as "CIVIL", "STRUCTURAL", "MECHANICAL", etc.
- Design Characteristics of **Professional Engineer (PE) stamps**:
    - **Shape**: **circular, rectangular, or any other shape**.
    - **Structure**: Contains **printed, structured text** rather than handwriting.
- **Note**: Don't Considered an item as professional engineer stamps if **Professional Title** are not available inside it.

#### City Stamps (AHJ / Department Approval Stamps)

A **City Stamp** is an official approval marking placed by a city, county, district, or authority having jurisdiction (AHJ) on permit plan sheets.

- **Mandatory Identification Rules** (must all be true to qualify as a city stamp):
    1. **Contains Approval Keywords**: Text must include one or more of the following words in **CAPITAL LETTERS** with **special formatting** (bold, larger font, or distinct styling):
        **"APPROVED"**, **"APPROVAL"**, **"RECEIVED"**, **"ACCEPTED"**, **"ISSUED"**, or **"CONDITIONAL APPROVAL"**.
    2. **Jurisdiction / Authority Reference**: Must explicitly reference a **city, county, district, or AHJ department name**. Examples:
        - "CITY OF SAN JOSE"
        - "SAN MATEO COUNTY"
        - "COASTSIDE FIRE PROTECTION DISTRICT"
        - "BUILDING DEPARTMENT", "FIRE DEPARTMENT", "PLANNING DIVISION"
    3. **Structured Printed Text**: Text must be formal, printed, and structured (not handwritten or freeform).

### Important Instructions
- Only detect structured, official **PE stamps**.
- Ignore handwritten marks or signatures by themselves; these are not stamps.
- Do not confuse logos, decorative symbols, or abstract shapes with stamps.
- A valid PE stamp may be partially covered by a signature—focus on the structured, printed portion.

---

You will be shown an image of a document. Based on the definition and characteristics above, determine whether the image contains a **valid PE stamp**.

---

### Output Format
Return a structured JSON response that follows this schema exactly:

{json_schema_str}

- Populate `checkStampPresence` with **"Yes"** if a **PE Stamp** is present on the sheet, and **"No"** otherwise.
- In `CheckStampType[0]`, set `ProfessionalEngineeringStamp` to **"Yes"** or **"No"** to indicate PE stamp detection.
```

#### Stamp System Prompt

```text
You are a stamp detector assistant. Examine the page carefully and determine if it contains an official **Professional Engineer (PE) stamp** based on structured printed text (e.g. profession title, state, license). Refer to the few-shot examples to see what a valid PE stamp looks like.
```

#### Stamp Output Schema

```json
{
  "checkStampPresence": "Yes or No",
  "CheckStampType": [
    {
      "ProfessionalEngineeringStamp": "Yes or No"
    }
  ]
}
```

#### North Direction Detection Description

```text
Carefully examine the image and determine whether it contains a geographical **North direction Symbol** and/or a **Scale Indicator**.

---

### Definition of a North Direction Symbol:
- **Definitions**: A **North Direction Symbol** is a geographical directional symbol used to indicate geographic orientation in technical diagrams, such as floor plans or site plans.
- **Expected values**: `"Detected"` if present else `"Not Detected"`.
- **Visual Indicators**:
    - A **North Direction Symbol** is often available with label "North" pointing in one direction.
    - May include a full compass rose (N, NE, E, etc.) or a single geographical direction symbol with just the letter "N".
    - Is often labeled with the word **"NORTH"** or simply the letter **"N"**.
    - May appear **near the scale indicator**, often in the **corner or edge** of a drawing.

**Assumption**: If a North Direction Symbol is present, a scale is **likely to co-occur nearby** or on the same page.

---

### Before concluding detection, verify:
- Does the arrow explicitly show geographic direction (not flow or diagram arrows)?
- Is the arrow paired with "NORTH"?
- Is it in a logical architectural position (e.g., corner, title block)?

---

### Important Instructions:
- **Only detect the North Direction Symbol and its associated scale**, if visible.
- Look for **standard architectural or civil drawing conventions** — geographical compass or north direction symbol and scales.
- Do **not** treat direction-like symbols embedded inside the site diagram or near object labels (e.g., house, driveway) as valid North Arrows.
- Do **not** interpret "(N)" in equipment labels (e.g., "(N) Inverter", "(N) Panel", "(E) House", "(N) PV Sytem) as a North direction symbol. These are **installation phase indicators** (e.g., New, Existing), not geographic directions.
- Do **not** consider direction arrow/annotations pointing towards the diagram as North Direction Symbol.
- Do **not** confuse decorative arrows, flow arrows, or compass-like logos with a true North Direction Symbol.
- Absolutely **do not treat any label in parentheses**, such as "(N)", "(E)", "(R)", or "(P)", as a North Direction Arrow. These are **not geographic directional symbols** — they indicate project phases:
    - (N) = New
    - (E) = Existing
    - (R) = Relocated
    - (P) = Proposed
- **Never treat label (N)** found in front of any equipment name (e.g., "(N) Inverter", "(N) PV System", "(E) Battery) as a North direction marker — even if they are near arrows or architectural elements.
- A valid **North Arrow must be a graphic directional symbol**, with an arrow clearly pointing and labeled with "NORTH", **not embedded in parentheses**.

---

### Output Format:
Detect the North direction symbol and scale indicator and return a structured JSON-like response based on what is detected:

{json_schema_str}
```

#### North Direction System Prompt

```text
You are a helpful assistant specialized in detecting geographic symbols in architectural, site, and construction drawings. Your task is to visually analyze the layout and determine whether the image contains a North Direction Symbol and/or a Scale Indicator. Refer to the few-shot examples to understand what a valid North direction symbol looks like.
```

#### North Direction Output Schema

```json
{
  "NorthDirectionSymbol": "Detected or Not Detected"
}
```

### 2.6 Standard LLM Strengths and Trade-offs

**Strengths**
- Fast iteration on behavior via prompt updates
- No training cycle required
- Easy to add or revise rules/few-shot examples

**Trade-offs**
- Higher sensitivity to prompt drift
- More variability across model versions
- Reliability can depend on prompt quality and example coverage

---

## 3) Fine-Tuned LLM Approach

### 3.1 Method Summary

The fine-tuned approach trains a vision model using labeled page-level examples so the model internalizes the task.  
At inference, the model uses a concise prediction instruction and returns a compact boolean JSON.

### 3.2 Training Data Strategy

Each labeled page is converted into one training record with:
- task system prompt
- prediction user instruction
- page image (base64 data URL)
- assistant target output (`stamp`, `north_arrow`)

### 3.3 Fine-Tuning Data Template

```json
{
  "messages": [
    {
      "role": "system",
      "content": "<SITE_PLAN_SYSTEM_PROMPT>"
    },
    {
      "role": "user",
      "content": "<PREDICTION_USER_INSTRUCTION>"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,<BASE64_IMAGE>"
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": "{\"stamp\": true, \"north_arrow\": false}"
    }
  ]
}
```

### 3.4 Ground-Truth Label Mapping

Labeling is page-based:
- page in `PE Stamp` list -> `stamp = true`
- page in `north_arrow` list -> `north_arrow = true`
- otherwise false

This mapping is transformed into supervised assistant outputs during dataset generation.

### 3.5 Fine-Tuned Prompt Design

#### Fine-Tuned System Prompt

```text
You are a site plan assistant. Every answer MUST follow the North arrow and PE Stamp definitions below—do not use looser rules.

Output JSON only, no other text: {"stamp": true/false, "north_arrow": true/false}
- **north_arrow**: true = "Detected", false = "Not Detected" for a North Direction Symbol (per definitions below).
- **stamp**: true ONLY if a PE Stamp OR a qualifying City/AHJ stamp is present per the sections "PE Stamp" and "City / AHJ stamp".

---

### Definition of a North Direction Symbol
- **Definitions**: A **North Direction Symbol** is a geographical directional symbol used to indicate geographic orientation in technical diagrams, such as floor plans or site plans.
- **Expected values** (in JSON): **north_arrow** true if present ("Detected"), else false ("Not Detected").
- **Visual indicators**:
  - A **North Direction Symbol** is often available with label "North" pointing in one direction.
  - May include a full compass rose (N, NE, E, etc.) or a single geographical direction symbol with just the letter "N".
  - Is often labeled with the word **"NORTH"** or simply the letter **"N"**.
  - May appear **near the scale indicator**, often in the **corner or edge** of a drawing.


---

### Before concluding north_arrow detection, verify:
- Does the arrow explicitly show geographic direction (not flow or diagram arrows)?
- Is the arrow paired with "NORTH" or clear compass/north convention?
- Is it in a logical architectural position (e.g., corner, title block)?

---

### North Direction Symbol — important instructions
- **Only detect the North Direction Symbol** (and use scale only as context for where to look).
- Look for **standard architectural or civil drawing conventions** — geographical compass or north direction symbol.
- Do **not** treat direction-like symbols embedded inside the site diagram or near object labels (e.g., house, driveway) as valid North arrows.
- Do **not** interpret "(N)" in equipment labels (e.g., "(N) Inverter", "(N) Panel", "(E) House", "(N) PV System") as a North direction symbol. These are **installation phase indicators** (e.g., New, Existing), not geographic directions.
- Do **not** consider direction arrows or annotations **pointing into the diagram** as a North Direction Symbol.
- Do **not** confuse decorative arrows, flow arrows, or compass-like logos with a true North Direction Symbol.
- Absolutely **do not treat any label in parentheses**, such as "(N)", "(E)", "(R)", or "(P)", as a North direction arrow. These are **not geographic directional symbols** — they indicate project phases:
  - (N) = New
  - (E) = Existing
  - (R) = Relocated
  - (P) = Proposed
- **Never treat "(N)"** in front of equipment names (e.g., "(N) Inverter", "(N) PV System", "(E) Battery") as a North direction marker — even if near arrows or architectural elements.
- A valid **North arrow must be a graphic directional symbol** with an arrow clearly pointing and labeled with "NORTH" or standard north/compass convention — **not embedded in parentheses** as a phase label.

---

## PE Stamp (Professional Engineer)
A **PE stamp** is an official engineer registration seal on the drawing.
- **Required**: Printed **professional title** such as LICENSED ENGINEER, PROFESSIONAL ENGINEER, or LICENSED PROFESSIONAL ENGINEER (if no such title, it is NOT a PE stamp).
- **Often also includes**: State/province (e.g. STATE OF CALIFORNIA), license number, expiration (e.g. EXP 03/25), discipline (CIVIL, STRUCTURAL, MECHANICAL).
- **Shape**: Circular, rectangular, or other; **structured printed text**, not handwriting alone.
- PE stamp may be partly covered by a signature—judge from the **printed** seal text.

---

## Rules
- Only structured official stamps count for **stamp**. Ignore standalone handwriting/signatures as stamps.
- Do not treat logos or decorative graphics as stamps or north arrows.
- Respond with JSON only: {"stamp": true/false, "north_arrow": true/false}.
```

#### Fine-Tuned User Instruction

```text
Analyze this plan sheet image. For north_arrow: apply ONLY the North Direction Symbol rules (reject (N)/(E) phase labels, flow arrows, diagram arrows). For stamp: apply ONLY the PE Stamp and City/AHJ stamp definitions. Reply with JSON only: {"stamp": true/false, "north_arrow": true/false}.
```

### 3.6 Fine-Tuning Workflow

1. Build training and validation JSONL sets
2. Upload datasets to model provider
3. Launch fine-tuning job on selected base model
4. Poll until completion
5. Save and use returned fine-tuned model ID for inference

### 3.7 Fine-Tuned Strengths and Trade-offs

**Strengths**
- Better task consistency for repeated document patterns
- Lower dependence on long runtime prompts
- Predictable output structure for known domains

**Trade-offs**
- Requires labeled data and training lifecycle
- Retraining needed for significant rule changes
- Operational overhead for dataset maintenance

---

## 4) Comparison: Standard vs Fine-Tuned

| Dimension | Standard LLM (Prompt-Based) | Fine-Tuned LLM |
|---|---|---|
| Setup cost | Low | Higher |
| Iteration speed | Very fast | Slower (train cycle) |
| Data requirement | Optional (few-shot) | Required (labeled dataset) |
| Consistency | Medium | High (domain-specific) |
| Prompt dependency | High | Moderate |
| Best use case | Rapid rule experimentation | Stable production behavior |

---

## 5) Output Contract

### Standard LLM
- Stamp output is schema-driven (`Yes/No`)
- North output is schema-driven (`Detected/Not Detected`)

### Fine-Tuned LLM
- Unified boolean JSON:

```json
{"stamp": true, "north_arrow": false}
```

---

## 6) Quality Controls and Practical Guidance

- Keep exclusion rules explicit to reduce false positives.
- Use representative few-shot images across drawing styles.
- Validate edge cases: partial stamps, noisy scans, title-block clutter.
- Track disagreement cases between approaches for continuous improvement.
- For high-confidence production workflows, prioritize fine-tuned inference and use prompt-based mode for rapid policy updates and diagnostics.
