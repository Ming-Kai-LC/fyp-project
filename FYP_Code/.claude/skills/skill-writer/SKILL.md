---
name: skill-writer
description: Expert guide for creating Claude Code skills. Use when asked to create a skill, write a SKILL.md file, or teach about skill creation. Provides templates, best practices, and validation guidance for personal, project, and plugin skills.
---

# Skill Writer

This skill teaches Claude Code how to properly create and write skills according to the official Claude Code documentation standards.

## What are Skills?

Skills are specialized capabilities that extend Claude Code's functionality. They provide focused, domain-specific guidance through `SKILL.md` files that Claude autonomously discovers and applies when contextually relevant.

## When to Create a Skill

Create a skill when you need:
- Specialized domain knowledge (e.g., PDF processing, Excel analysis)
- Repeatable workflows with specific steps
- Consistent behavior across projects or team members
- Tool restrictions for read-only or security-sensitive operations

**Don't create a skill for**: One-off tasks, simple automations (use slash commands instead), or overly broad capabilities.

## SKILL.md File Structure

Every skill requires a `SKILL.md` file with this exact structure:

```yaml
---
name: skill-name-here
description: Brief description of what this Skill does and when to use it
---

# Skill Display Name

Clear introduction explaining the skill's purpose.

## Instructions

Step-by-step guidance for Claude to follow when this skill is activated.

1. First step with clear action
2. Second step with specific details
3. Continue with concrete instructions

## Examples

Concrete examples showing how to use this skill:

**Example 1: [Scenario Name]**
- Context: When this situation occurs
- Action: Do this specific thing
- Result: Expected outcome

**Example 2: [Another Scenario]**
- Context: Different use case
- Action: Alternative approach
- Result: What to expect

## Best Practices

- List important considerations
- Include common pitfalls to avoid
- Provide optimization tips

## Troubleshooting

Common issues and their solutions:
- **Issue**: Description
  - **Solution**: How to fix it
```

## Required Fields

### name
- **Format**: lowercase letters, numbers, and hyphens only
- **Max length**: 64 characters
- **Example**: `pdf-form-filler`, `excel-analyzer`, `skill-writer`
- **Invalid**: `PDF Filler`, `skill_name`, `SKILL-NAME`

### description
- **Max length**: 1024 characters
- **Must include**:
  1. What the skill does (functionality)
  2. When to use it (activation triggers)
- **Good example**: "Analyze Excel spreadsheets, generate pivot tables, create charts. Use when working with .xlsx files or when user requests data analysis from spreadsheets."
- **Bad example**: "Helps with data" (too vague, no trigger words)

## Optional Fields

### allowed-tools
Restricts which Claude Code tools are available during skill execution.

```yaml
---
name: read-only-analyzer
description: Analyzes code without making changes
allowed-tools:
  - Read
  - Grep
  - Glob
  - Bash
---
```

Use this for:
- Read-only analysis skills
- Security-sensitive operations
- Preventing accidental modifications

## File Locations

### Personal Skills
```
~/.claude/skills/skill-name/SKILL.md
```
For individual use only.

### Project Skills
```
.claude/skills/skill-name/SKILL.md
```
Shared with project team, version controlled.

### Plugin Skills
Distributed through Claude Code plugins. See plugins documentation.

## Supporting Files

Organize additional resources in the skill directory:

```
.claude/skills/my-skill/
├── SKILL.md           # Main skill file (required)
├── reference.md       # Detailed documentation
├── examples.md        # Extended examples
├── templates/         # Reusable templates
│   ├── template1.md
│   └── template2.py
└── scripts/           # Utility scripts
    └── helper.sh
```

Claude loads these progressively when contextually relevant.

## Writing Effective Descriptions

The description is crucial for autonomous skill discovery. Include:

1. **Action verbs**: "Analyze", "Generate", "Convert", "Process"
2. **Specific domains**: "Excel spreadsheets", "PDF forms", "JSON APIs"
3. **File types**: ".xlsx files", ".pdf documents", ".ipynb notebooks"
4. **User intent keywords**: "when user requests", "when asked to", "for tasks involving"

**Template**:
```
[Action] [specific domain/task], [additional capabilities]. Use when [trigger condition] or when [user intent].
```

**Examples**:
- ✅ "Convert PDF forms to fillable templates with validation. Use when working with .pdf files or when user needs interactive form creation."
- ✅ "Analyze Jupyter notebooks for data science workflows, identify performance issues, suggest optimizations. Use when working with .ipynb files or debugging ML pipelines."
- ❌ "Helps with PDFs" (too vague)
- ❌ "Does everything related to data" (too broad)

## Best Practices Checklist

- [ ] **One skill, one purpose**: Each skill addresses a single capability
- [ ] **Specific description**: Include concrete trigger terms and file types
- [ ] **Clear instructions**: Step-by-step guidance, no ambiguity
- [ ] **Concrete examples**: Show real-world usage scenarios
- [ ] **Valid YAML**: Use spaces (not tabs), proper indentation
- [ ] **Correct file path**: Verify location matches skill type (personal/project)
- [ ] **Test activation**: Verify skill loads with `claude --debug`
- [ ] **Version tracking**: Document changes in skill updates
- [ ] **Tool restrictions**: Use `allowed-tools` when appropriate

## Common Issues and Solutions

### Skill doesn't activate
- **Check description specificity**: Add more trigger keywords
- **Verify file path**: Ensure correct directory structure
- **Validate YAML syntax**: Check for tabs vs spaces, indentation
- **Run debug mode**: `claude --debug` to view loading errors

### Skill activates too often
- **Narrow description**: Make trigger conditions more specific
- **Split broad skills**: Break into focused sub-skills

### Instructions unclear
- **Add examples**: Show concrete scenarios
- **Be explicit**: Avoid assumptions about context
- **Test with teammates**: Get feedback on clarity

## Validation Checklist Before Saving

```yaml
# Verify YAML frontmatter:
# - name: lowercase, hyphens only, max 64 chars
# - description: clear functionality + triggers, max 1024 chars
# - allowed-tools: (optional) list of valid tool names
```

```markdown
# Verify Markdown content:
# - Clear heading structure
# - Step-by-step instructions
# - Concrete examples
# - Best practices section
# - Troubleshooting guidance (if applicable)
```

## Example: Complete Skill Template

```yaml
---
name: example-skill
description: [What it does]. [Additional capabilities]. Use when [trigger condition] or when working with [file types/domains].
---

# Example Skill Name

Introduction paragraph explaining the skill's purpose and value.

## Instructions

1. **First Step**: Specific action with clear outcome
   - Detail or consideration
   - Another important point

2. **Second Step**: Next action in the workflow
   - Use [specific tool] for this
   - Expected result

3. **Final Step**: Completion and validation
   - How to verify success
   - What to report to user

## Examples

**Example 1: Common Use Case**
```
User request: "Do X with Y"
Action: Apply step 1, 2, 3 from instructions
Result: Z is produced successfully
```

**Example 2: Edge Case**
```
User request: "Handle special scenario"
Action: Modified approach for this case
Result: Appropriate handling
```

## Best Practices

- Important consideration for optimal results
- Common pitfall to avoid
- Performance or quality tip

## Troubleshooting

**Issue: Skill doesn't work as expected**
- Solution: Check this specific thing
- Alternative: Try this approach

**Issue: Error occurs in step X**
- Solution: Verify Y before proceeding
- Prevention: Always do Z first
```

## Testing Your Skill

1. **Create the skill**: Place `SKILL.md` in correct directory
2. **Restart Claude Code**: Reload to discover new skill
3. **Test activation**: Use trigger keywords from description
4. **Verify behavior**: Ensure Claude follows instructions
5. **Debug if needed**: Run `claude --debug` to check for errors
6. **Iterate**: Refine description and instructions based on results

## Sharing Skills

### With Team (Project Skills)
1. Commit `.claude/skills/skill-name/` to version control
2. Team members automatically get skill on pull
3. Document skill purpose in project README

### Via Plugin (Plugin Skills)
1. Package skill following plugin structure
2. Publish to Claude Code plugin registry
3. Users install via plugin manager

## Quick Reference

| Aspect | Requirement |
|--------|-------------|
| Filename | `SKILL.md` (exact case) |
| Name format | `lowercase-with-hyphens` |
| Name max length | 64 characters |
| Description max | 1024 characters |
| Required fields | `name`, `description` |
| Optional fields | `allowed-tools` |
| YAML indentation | Spaces only (no tabs) |
| Personal path | `~/.claude/skills/skill-name/` |
| Project path | `.claude/skills/skill-name/` |

## When to Use This Skill

Activate this skill when:
- User asks to "create a skill"
- User requests "write a SKILL.md file"
- User wants to learn about "skill creation"
- User asks "how to make a skill"
- User needs "skill writing guidance"
- Task involves extending Claude Code capabilities through skills

## Final Notes

Skills are powerful because Claude discovers and applies them autonomously. Invest time in:
1. **Precise descriptions** - Enable correct activation
2. **Clear instructions** - Ensure consistent execution
3. **Concrete examples** - Demonstrate proper usage
4. **Iterative refinement** - Improve based on real-world use

Remember: A well-written skill becomes a permanent capability that enhances Claude Code for all future tasks in its scope.
