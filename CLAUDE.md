## Code Style
* Keep comments concise and focused. Avoid over-explaining - prefer 2-3 line comments over paragraphs.
* No comments inside dictionaries - keep them clean
* Keep dictionaries on multiple lines for readability
* Avoid excessive inline comments on obvious code - use a single block comment above repetitive code sections instead
* For functions (other than plotting functions), especially dthose containing business logic, use Google-style docstrings and comments explaining non-obvious code.

## Function Design Principles
* KISS principle - "Keep it simple, stupid" - avoid over-engineering
* Minimal lines for simple operations - don't be verbose unless necessary
* Let errors "bubble up" instead of complex error handling - prefer simple code that fails fast
* Remove unnecessary defensive if-else checks - let operations fail clearly
* Question every utility function - inline simple operations when appropriate
* Use consistent patterns throughout (e.g., pandas everywhere, not mixed approaches)

## Code Implementation Style
* **Prefer minimal, compact code** - Use 2-5 lines instead of creating functions or verbose implementations
* **Inline simple operations** - Don't create helper functions for straightforward tasks
* **Direct approach** - Get straight to the solution without over-engineering
* **No unnecessary abstractions** - Avoid wrapping simple operations in functions unless specifically requested
* **Condensed code spacing** - Combine multiple lines into single statements when readable, prioritize vertical space efficiency
* **Multi-line function calls** - Put multiple parameters on same line when it improves readability and reduces vertical space

## Plotting Code Formatting
* **Single-line plotting calls** - Keep `fig.add_trace()`, `go.Scatter()`, `go.Bar()` etc. on single lines when possible
* **Consolidate parameters** - Put plotting parameters on same line rather than breaking unnecessarily
* **Avoid excessive line breaks** - Don't break lines in plotting code unless the line becomes unreadably long
* **Examples of preferred formatting:**
  - ✅ `fig.add_trace(go.Scatter(x=data['x'], y=data['y'], mode='markers', marker=dict(color='blue', size=8), name='Data'), row=1, col=1)`
  - ✅ `fig.update_layout(title="Chart Title", height=800, showlegend=True, hovermode='x unified')`
  - ❌ Avoid breaking simple calls across multiple lines unless truly necessary for readability

## Error Handling Philosophy
* Avoid try-except blocks unless handling expected scenarios (e.g., FileNotFoundError)
* Trust operations to work or fail with clear error messages
* No silent failures or defensive programming - fail fast and explicit

## Command Execution Guidelines

### Running Python Scripts
**Three ways to run Python code:**

1. **Interactive files with `# %%` markers** (e.g., `scratch/training.py`, `scratch/test_gpu_optimization.py`)
   - These are Jupyter-style files meant for VSCode's Python Interactive window
   - User runs these cell-by-cell - DO NOT run them as scripts
   - ❌ Never use: `python filename.py` or `conda run -n env python filename.py`
   - ✅ User will run them interactively in VSCode

2. **Standalone Python scripts** (no `# %%` markers)
   - Can be run directly from command line
   - ✅ Use: `conda run -n drl-crypto-portfolio python script.py`
   - Check correct conda environment name first with `conda env list`
   - Must handle imports correctly (see below)

3. **Inline Python commands** (for quick checks)
   - ✅ Use: `conda run -n drl-crypto-portfolio python -c "print('hello')"`
   - Keep simple - single line only
   - No multiline strings with conda run (causes parsing errors)
   - For complex operations, create a temporary .py file instead

### Common Python Execution Errors Encountered

**Import errors** (`ModuleNotFoundError: No module named 'src'`):
- **Cause**: Python can't find project modules
- **Fix**: Add `sys.path.insert(0, '.')` at top of script OR run from project root
- **Example**:
  ```python
  import sys
  sys.path.insert(0, '.')
  from src.policies import CNNPolicy  # Now works
  ```

**Wrong conda environment**:
- **Cause**: Using wrong environment name (e.g., `rl-portfolio` instead of `drl-crypto-portfolio`)
- **Fix**: Check with `conda env list` first, then use exact name
- **Example**: `conda run -n drl-crypto-portfolio python script.py`

**Conda multiline command errors**:
- **Cause**: `conda run` doesn't support multiline Python strings well
- **Fix**: Write code to a .py file first, then run the file
- ❌ Avoid: `conda run -n env python -c "line1\nline2\nline3"`
- ✅ Use: Create temp.py, then `conda run -n env python temp.py`

**Running interactive files as scripts**:
- **Cause**: Trying to run `# %%` files with `python filename.py`
- **Fix**: Don't run them - these are for user's interactive execution
- **Recognition**: If file has `# %%` markers anywhere, it's interactive

### General Command Guidelines
* **Avoid f-strings with {} in bash -c commands** - Bash interprets `${}` as variable substitution, causing "bad substitution" errors
* **No Unicode characters in command output** - Use ASCII only (avoid ✅❌⚠️ emojis) due to Windows codepage limitations (cp1252)
* **Safe string formatting patterns:**
  - ✅ Use: `print('Value:', variable)` or `'Text {}'.format(var)`
  - ❌ Avoid: `print(f'Value: ${variable}')` in bash commands
* **Quote escaping in multiline commands:**
  - Minimize nested quotes and complex string literals within bash -c
  - Test commands incrementally when building complex validation scripts
* **Alternative approaches for complex operations:**
  - Create temporary .py files for multi-step analysis
  - Use simple concatenation instead of f-strings when inside bash commands
  - Break complex validations into multiple simple bash calls

## Interest Rate Representation Standard
* **ALL interest rates, funding rates, and percentages MUST be stored as decimal numbers in variables and calculations**
* Examples of correct decimal representation:
  - 1% → store as `0.01`
  - 5.6 basis points → store as `0.00056`
  - 58.6% → store as `0.586`
  - 8-hourly funding rate of 2% → store as `0.02`
  - Annualized rate of 10% → store as `0.1`
* **Only multiply by 100 when displaying to users** (e.g., `f"{rate * 100:.4f}%"`)
* This prevents confusion between decimal and percentage representations in calculations
* All mathematical operations (addition, multiplication, comparisons) should work with decimal values

## UTC Timezone Standard
* **ALL timestamps, dates, and times MUST be recorded, stored, and reported in UTC**
* Never use local timezone or naive datetime objects - always specify UTC explicitly
* **When printing/logging datetime objects, always use timezone-aware objects to show +00:00 automatically**
* Examples of correct UTC usage:
  - `datetime.datetime.now(tz=datetime.timezone.utc)`
  - `datetime.datetime.fromtimestamp(ts/1000, tz=datetime.timezone.utc)`
  - `pd.to_datetime(df['timestamp'], unit='ms', utc=True)`
  - `print(f"Time: {datetime_obj}")` where datetime_obj has timezone info
* This ensures consistency across different systems and locations
* All time-based calculations and comparisons should work with UTC timestamps


## Development Guidelines
* Always clean up temporary test files after completing development work
* Remove any test scripts created during development using `rm` command when done
* Maintain existing code patterns and conventions in the codebase
* Implement tricky code changes step-by-step with testing after each modification, especially for logic changes that benefit from human review
* For simple changes (comments, docstrings, visualizations), update in one go rather than step-by-step
* Consolidate similar operations under unified approaches
* Document standards explicitly and apply uniformly

## Plan Mode Guidelines
* **Stay SUCCINCT** - Plans should be high-level overviews, not detailed specifications
* **NO CODE in plans** - At most use pseudocode or brief conceptual examples
* **Problem → Solution → Files** - State the issue, propose the fix, list affected files
* **One page maximum** - If the plan is longer than one page, it's too detailed
* **Bullet points preferred** - Use short, scannable bullet points over paragraphs

## Refactoring Requirements
* **PRECISE CODE COPYING** - When refactoring or moving code between files, copy it EXACTLY as-is
* Preserve all spacing, newlines, blank lines, and comments from the original
* Preserve the exact variable names the user chose - do NOT rename variables
* Preserve all assertions, even if they seem redundant
* Preserve unused variables - the user may have a reason for them
* Do NOT reformat, condense, or "improve" the code during refactoring
* Do NOT reorder statements or variable declarations
* You may ADD docstrings or function signature changes (parameters), but NEVER modify the function body logic
* Only modify the actual logic/behavior when explicitly requested
* If you notice issues (unused vars, inefficient code, etc.), mention them but do NOT fix them unless asked

## CSV Data Safety Guidelines
* **NEVER edit financial CSVs with spreadsheet software** (LibreOffice Calc, Excel) - they auto-convert decimals to percentages
* **Use Python scripts for bulk row deletion/filtering** instead of manual editing:
  ```python
  df = pd.read_csv('file.csv')
  df_filtered = df[your_filter_condition]
  df_filtered.to_csv('file.csv', index=False)
  ```
* **Text editors only for manual CSV editing** - Use VS Code with CSV extensions for safe viewing/editing
* **Always backup before manual operations** - Create timestamped backups before any data manipulation
* **Validate data after manual edits** - Check for corrupted decimal values (e.g., TTE > 5 years indicates corruption)
* **LibreOffice corruption pattern**: Small decimals like `0.025` become `25` due to percentage auto-formatting

## Interactive Development Boilerplate Requirements
**DO:** Create executable boilerplate code with realistic data for quick interactive testing
* Replace over-engineered classes/functions with direct, executable initialization code
* Use realistic dummy parameters that mirror actual usage patterns
* Provide working object instances that can immediately test methods
* Make code copy-pasteable into interactive environments

**DON'T:** Create abstract, commented-out, or overly-simplified mock implementations
* Avoid creating unnecessary classes when simple initialization suffices
* Don't provide lambda functions or minimal examples when full setup is needed
* Avoid commented-out code blocks - provide executable alternatives

## Testing and Validation Requirements
* Run intermediate tests for every function that gets modified during implementation
* After completing changes to a function, create and execute a test to validate:
  - Function still accepts expected inputs without errors
  - Function returns expected output types and shapes
  - Function behavior is consistent with original logic (where applicable)
  - Edge cases and error conditions are handled properly
* Use simple test cases with known expected results to verify correctness
* Document any behavioral changes or new requirements in function docstrings
* For mathematical functions, validate with simple hand-calculated examples when possible

## Output Formatting Preferences
* **Compact but informative** - Show essential information in 2-3 lines, not verbose multi-line reports
* **Consistent precision** - Use appropriate decimal places for the data type (whole numbers for counts/indices, 2 decimals for currency, 4 decimals for rates)
* **Meaningful units** - Include units when they add clarity (e.g., "days", "$")
* **Clean variable names** - Use full words over abbreviations in user-facing output ("error" not "err")
* **Progress indicators** - Use arrows (→) to show progression through steps
* **Parameter summaries** - Include key parameters with appropriate precision and units

## Visual Logging
* Easy scanning: Categories make it easy to find specific information
* Professional appearance: Clean, structured output
* Consistent data reporting: No conditional messaging inconsistencies

## Visualization Preferences
* **Maximize plot area** - Minimize dead white space, use tight margins and reduced spacing between subplots
* **Interactive plots preferred** - Use Plotly for complex multi-panel visualizations to enable zooming and synchronized navigation
* **Large figure sizes** - Prefer larger dimensions (1200+ height) for better readability and detail
* **Efficient layout** - Reduce vertical spacing and margins to pack more content into the visible area

## Project Conventions
* A year contains 365 days (not 365.25 days)
* (interest) rates and forward rates are assumed to be annualized and continuously compounded, unless stated otherwise.
* Funding rates are assumed to be simple rates. They may either by 8-hourly or annualized.