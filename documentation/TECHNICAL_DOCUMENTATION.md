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

#### Stamp System Prompt (PE-only behavior)

```text
You are a stamp detector assistant. Examine the page carefully and determine if it contains an official Professional Engineer (PE) stamp based on structured printed text (e.g. profession title, state, license). Do not consider City or AHJ approval stamps. Refer to the few-shot examples to see what a valid PE stamp looks like.
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

#### Fine-Tuned System Prompt (core behavior)

The system prompt encodes:
- strict North-direction symbol definition
- strict PE-stamp and City/AHJ stamp definition
- explicit exclusions (logos, decorative marks, phase labels like `(N)/(E)`)
- required JSON output format

Expected output shape:

```json
{"stamp": true, "north_arrow": false}
```

#### Fine-Tuned User Instruction

```text
Analyze this plan sheet image.
For north_arrow: apply ONLY the North Direction Symbol rules (reject (N)/(E) phase labels, flow arrows, diagram arrows).
For stamp: apply ONLY the PE Stamp and City/AHJ stamp definitions.
Reply with JSON only: {"stamp": true/false, "north_arrow": true/false}.
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
