import re
import json

def create_xml_to_json_script_prompt(table_xml_str: str) -> str:
    return f"""
# XML to JSON Table Conversion Task

Generate a Python 3.11 script that converts OCR-extracted table XML into structured JSON format.

## Workflow Overview

You will analyze the XML, write a conversion script, and validate it using the `execute_json_script` tool:

1. **Analyze** the XML structure (Phase 1 below)
2. **Generate** a Python script based on your analysis (Phase 2 below)
3. **Test** your script using `execute_json_script` tool - **MANDATORY: You MUST call this tool at least once**
4. **Validate** the output:
   - Tool returns first 5 entries (full data for entries 1-2)
   - Verify the FIRST compound matches the FIRST data row in XML
   - Check for off-by-one errors, skipped rows, or included headers
   - If total_count is 0 but you can see valid data rows in the XML preview, your script has a bug - FIX IT and retry
   - Confirm field values match the XML source
5. **Refine**
   - If ANY part of the output from execute_json_script fails → FIX the script and call it again with fixed script.
   - Your last successful execute_json_script call contains the final script. Once validation passes, simply confirm completion - no need to repeat the script.

---

## PHASE 1: XML ANALYSIS

### 1.1 Identify Table Type

**TYPE A: Row-per-Compound (Standard)**
- Each row = one compound with multiple measurements
- Example structure:
```
  | Compound ID | Inhibition | IC50  |
  | 598206      | 45        | 12.3  |
  | 761909      | 67        | 8.5   |
```
- Output: One JSON entry per row

**TYPE B: Compounds-as-Columns (Needs Unpivoting)**  
- Compound IDs in column headers, measurements in cells
- Example structure:
```
  | Time | ISIS 181071 | ISIS 181080 | ISIS 29848 |
  | 2    | 91          | 0           | 0          |
  | 4    | 99          | 44          | 2          |
```
- Output: One JSON entry per cell (row × compound)

**Detection Rules:**
- TYPE B if: Compound/sample IDs in headers OR first column is a variable (Time, Dose)
- TYPE A if: First column contains compound identifiers

### 1.2 Locate Headers and Data

1. **Header rows**: Contain column names/units (may span multiple rows)
2. **First data row**: 
   - TYPE A: Row where first column has compound ID
   - TYPE B: Row after last header, where first column has variable value
3. **Record indices** for skipping headers in your script

### 1.3 Create Field Name Mappings

Transform column headers to field names using these rules IN ORDER:

1. Reconstruct Headers Across <tgroup>s: The full header for a data column is often split across different <tgroup> blocks. Your primary analysis task is to find these separate header pieces and combine them. Do not generate generic names like col1 or field2; this indicates a failed analysis.

2. **Standardize text**:
   - Lowercase all text
   - Replace spaces with underscores
   - Replace symbols with words:
     - `%` → `pct`
     - `/` → `per`  
     - `@` → `at`
     - Greek letters → English (α → alpha, β → beta)
   - Remove parentheses around units
   - Preserve decimal points in field names

3. **Examples**:
```
   "ALT (U/L) at 30 mg/kg" → "alt_u_per_l_at_30_mg_per_kg"
   "Body weight change (%)" → "body_weight_change_pct"
   "EC₅₀ (nM)" → "ec50_nm"
   "at 20.00 μM" → "at_20_00_um" (decimal preserved)
```

### 1.4 Identify Special Processing Needs

- **Split value concatenation**: Long values (sequences, URLs) split across multiple rows for display
  - Detection: Continuation rows with empty first column but text fragments in same column position
  - Example: "GCCTCTG" → "ATTCCCT" → "GAACTG" = concatenate to "GCCTCTGATTCCCTGAACTG"
  - Action: Concatenate sequential fragments until next compound ID
  
- **Spillover rows** (TYPE A): Empty first column but DIFFERENT measurement values
  - Detection: Complete values (not fragments) in continuation rows  
  - Action: Create separate JSON entries with same compound_id
  
- **Carry-down columns** (TYPE A): Constant properties that persist across related rows
  - Detection: Property columns (dose, treatment, concentration)
  - Action: Use last non-empty value for that compound
  
- **Unpivoting** (TYPE B): Transform column-based compounds into row-based entries

---

## PHASE 2: SCRIPT GENERATION

### 2.1 Script Structure
```python
def xml_to_json(xml_str: str) -> str:
    # Parse XML
    # Skip hardcoded header rows
    # Process data rows based on table type
    # Return JSON string
```

Requirements:
- Use only: `json`, `re`, `xml.etree.ElementTree`
- Return valid JSON via `json.dumps()`
- No `if __name__` block needed

### 2.2 Processing Logic by Table Type

#### TYPE A Processing:
```python
# Hardcode from your analysis:
header_rows = [0, 1]  # Rows to skip
col_mapping = {{0: "compound_id", 1: "inhibition_pct", 2: "ic50_nm"}}  # col_idx: field_name

# Process each data row:
for row in data_rows:
    if row[0]:  # Has compound ID
        current_compound = row[0]
        # Create new entry
    else:  # Spillover row
        # Use current_compound, create additional entry
```

**Value Parsing Rules:**
- Empty/"−"/"N/A"/"NA"/"ND" → `null`
- Numeric strings → `int` or `float` (preserve precision)
- HTML entities → decode (`&#x2003;` → space)
- Text → string

**Carry-Down Logic:**
- Apply ONLY to constant properties (dose, treatment, etc.)
- NOT for measurements/results
- Check column semantics to determine

#### TYPE B Processing:
```python
# Hardcode from your analysis:
compound_cols = {{2: "ISIS 181071", 3: "ISIS 181080"}}  # Keep exact header text
variable_field = "time_hours"  # What rows represent
value_field = "pct_inhibition"  # What's measured

# Unpivot the table:
for row in data_rows:
    for col_idx, compound_id in compound_cols.items():
        entries.append({{
            "compound_id": compound_id,
            "data": {{
                variable_field: parse(row[0]),
                value_field: parse(row[col_idx])
            }}
        }})
```

### 2.3 Output Format
```json
{{
    "entries": [
        {{
            "compound_id": "598206",  # Always string
            "data": {{
                "field1": value,  # int/float/string/null
                "field2": value
            }}
        }}
    ]
}}
```

Note: Same compound_id may appear multiple times (spillover rows or multiple measurements).

---

## Input XML Preview:

{table_xml_str}

"""

def create_helm_script_prompt(custom_id: str, table_data: dict) -> str:
    context_text = open(custom_id.replace('.xml', '_context.txt'), 'r').read()
    full_xml = open(custom_id).read()
    lines = full_xml.split('\n')
    table_xml_str = '\n'.join(lines[:150] + ['... (truncated) ...'] + lines[-150:]) if len(lines) > 300 else full_xml
    xml_text = context_text + '\n' + table_xml_str
    custom_id_data = table_data[custom_id]['compound_data']

    import random
    compound_ids = list(custom_id_data.keys())
    sample_size = min(5, len(compound_ids))
    # Use deterministic seed for cache consistency
    random.seed(42)
    sampled_ids = random.sample(compound_ids, sample_size)
    custom_id_example_inputs = [{'compound_id': cid, **custom_id_data[cid][0]} for cid in sampled_ids]

    return f"""
## Task
Write a Python 3.11 script to generate HELM notation strings for oligonucleotide compounds from patent table data.

## WORKFLOW
1. Write your script based on analysis of the context and example inputs
2. Call `execute_helm_script` with your script - it runs on the first 3 compounds automatically
3. Review results:
   - **All successful?** → Respond with just "Validated" (nothing else)
   - **Errors?** → Fix script and call tool again

**IMPORTANT**: Do NOT repeat the script in your final response. Just say "Validated" - we extract the script from your tool call.

## Important: Table-Specific Script
This script will ONLY process compounds from the single table shown below. Keep it simple:
- **Look at the actual example inputs** to see what fields are present (don't guess field names)
- **Check the context once** to find chemistry info (don't write complex parsing logic)
- **Hardcode values when you find them** - chemistry is usually consistent across all compounds in a table
- If chemistry varies per-compound, read it from `compound_data`; otherwise extract once from context

## Output Format
Return a Script object (or dictionary) containing:
- `pyscript`: Complete Python script as a single string

## Function Requirements
- Define function: `annotate_helm(compound_data: dict) -> str | None`
- Return one of:
    - Valid HELM notation string (ends with `}}$$$$`)
    - `None` if data is truly missing (e.g., no sequence field exists)
    - Descriptive error string for debugging (e.g., "Length mismatch: sequence has 20 bases but sugar_motif has 19 characters")
- Descriptive errors help you debug issues during testing - the tool will show you these messages
- No `if __name__ == '__main__'` block needed

## Analysis Steps

### Step 1: Find Sequence and Chemistry

**Sequence** - Check in this order:
1. Look at example inputs below - what field contains the sequence? (e.g., `sequence_5_to_3`, `seq`, `sequence`)
2. Read from that exact field name in `compound_data`
3. **Default assumption:** Sequences are 5' to 3' unless field name says otherwise

**Chemistry** - Check both locations:
1. **Context text first** - Does it describe chemistry that applies to ALL compounds? 
- Sugar motif pattern
- Backbone/Linkage motif pattern
- Base modifications (e.g., "each cytosine residue is a 5-methyl cytosine")
- If found in context → hardcode these values in your script
- Linkage/backbone motifs represent connections between nucleotides, so they will always be 1 character shorter than the sugar motif and sequence (N-1 linkages for N nucleotides)

2. **Compound data** - Do individual compounds have their own chemistry fields?
- Look at example inputs for fields like: `sugar_motif`, `linkage_motif`, `modification_pattern`
- If present → read per-compound from `compound_data`

**CRITICAL: Returning `None` is the CORRECT and EXPECTED behavior when data is missing.**

It is better to return `None` than to guess chemistry. If you cannot find:
- Explicit sugar motif information → return `None`
- Explicit backbone/linkage information → return `None`  
- Clear base modification rules → return `None`

**Never assume:**
- ❌ "ISIS compounds are usually MOE/PS, so..."
- ❌ "This is probably phosphorothioate because..."
- ❌ "Therapeutic oligos typically use..."

**Only use what is explicitly stated in context or compound_data.**

### Step 2: Construct HELM Notation

**HELM Structure:**
```
RNA1{{{{[sugar](base)[backbone].[sugar](base)[backbone]....[sugar](base)}}}}$$$$
```

**HELM Notation Reference:**

*Sugars* (motif characters → HELM notation):
- `e` → `[moe]` (2'-O-methoxyethyl/MOE)
- `d` → `d` (2'-deoxy/DNA)
- `r` → `r` (ribose/RNA)
- `m` → `m` (2'-O-methyl)
- `k` → `[cet]` (cEt)
- `f` → `[fR]` (2'-fluoro)
- `l` → `[lna]` (LNA)
- Novel sugars described in context but not listed above → use `[?]` as the HELM sugar notation

*Backbones/Linkages* (motif characters → HELM notation):
- `s` → `[sp]` (phosphorothioate/PS)
- `o` → `.` (phosphodiester/PO - use only the dot, **do NOT write `[po]`**)
- `a` → `[am]` (phosphoramidate/PMO)
- Novel linkages described in context but not listed above → use `[?]` as the HELM backbone notation

**Note:** Use `[?]` when chemistry IS described but has no standard HELM code. Return `None` when chemistry data is missing entirely.

*Bases:*
- Standard: A, C, G, T, U
- `[5meC]` = 5-methylcytosine (if context says cytosines are methylated)

**Construction Rules:**
1. Each nucleotide format: `[sugar](base)[backbone].`
2. Last nucleotide has NO backbone connector - there are N-1 linkages for N nucleotides
3. Phosphodiester (PO): Use only `.` separator (no `[po]` notation)
4. Phosphorothioate (PS): Use `[sp].` between nucleotides
5. **Must terminate with exactly:** `}}}}$$$$`

**Example:**
```
RNA1{{{{[moe]([5meC])[sp].[moe](A).d(G)[sp].d(T)}}}}$$$$
```

**Construction Process:**
- For each position in sequence: determine sugar → base → backbone
- Align motif/pattern strings with sequence positions (must match length)
- Handle edge cases: empty sequences, incomplete data → return `None`

## Context and Table Header
*(May contain chemistry info that applies to all compounds)*
```xml
{xml_text}
```

### Example compound_data inputs from this table:
```python
{custom_id_example_inputs}
```

## Your Response
Provide the complete, executable Python script containing the `annotate_helm` function.
"""

def create_collate_script_prompt(custom_id: str, table_data: dict, data_schema: dict) -> str:
    context_text = open(custom_id.replace('.xml', '_context.txt'), 'r').read()
    full_xml = open(custom_id).read()
    lines = full_xml.split('\n')
    table_xml_str = '\n'.join(lines[:150] + ['... (truncated) ...'] + lines[-150:]) if len(lines) > 300 else full_xml
    xml_text = context_text + '\n' + table_xml_str
    custom_id_data = table_data[custom_id]['compound_data']
    import random
    compound_ids = list(custom_id_data.keys())
    sample_size = min(5, len(compound_ids))
    # Use deterministic seed for cache consistency
    random.seed(42)
    sampled_ids = random.sample(compound_ids, sample_size)
    examples_list = [{'compound_id': cid, **custom_id_data[cid][0]} for cid in sampled_ids]
    custom_id_example_inputs = ""
    for i, ex in enumerate(examples_list):
        custom_id_example_inputs += f"example_{i+1} = {json.dumps(ex, indent=2)}\n\n"

    data_schema_str = ""
    for name, description in data_schema.items():
        data_schema_str += f"- **{name}**: {description}\n"
    
    return f"""# Task: Generate a Data Mapping Script

Write a Python 3.11 script that maps pre-extracted compound data to this schema.

## Target Schema
{data_schema_str}

## WORKFLOW
1. Write your script based on analysis of the schema and example inputs
2. Call `execute_collate_script` with your script - it runs on the first 3 compounds automatically
3. Review results:
   - **All successful?** → Respond with just "Validated" (nothing else)
   - **Errors?** → Fix script and call tool again

**IMPORTANT**: Do NOT repeat the script in your final response. Just say "Validated" - we extract the script from your tool call.

## Two Data Sources

**Source 1: compound_data dict** (varies per compound)
- Pre-extracted fields from table rows
- Field names from column headers
- ALWAYS use these when available

**Source 2: Context text** (same for all compounds)
- Experimental conditions from prose paragraphs
- Hardcode values that apply to entire table
- Only use when field NOT in compound_data

## Context and Table
```xml
{xml_text}
```

## Example Inputs
```python
{custom_id_example_inputs}
```

## Function Requirements
```python
def map_data(compound_data: dict) -> list[dict]:
    \"\"\"
    Returns: List of dicts with schema keys (no None values).
    \"\"\"
```

### Mapping Rules

1. **For each schema field:**
   - Check if it exists in compound_data (use those field names from examples)
   - If not, scan context text for the value
   - Apply any unit conversions per schema description

2. **Multiple measurements:**
   - If compound_data has multiple related values (e.g., `inhibition_at_0.0625_um`, `inhibition_at_0.125_um`), return one dict per measurement

3. **Priority:**
   - compound_data ALWAYS wins if field exists in both places
   - Only include fields where data exists (no None values)

4. **Simple code:**
   - Look at the actual field names in the examples
   - Don't check for field variations not shown in examples
   - This script only runs on THIS table with THESE field names

5. **Missing data is expected:**
   - Partial mappings are correct—not every table has every schema field
   - Omit fields entirely if data doesn't exist; never guess or hallucinate

## Your Response
Provide the complete, executable Python script containing the `map_data` function via the tool call.
"""