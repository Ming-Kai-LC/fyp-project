# Skills Usage Guide - FYP CrossViT COVID-19 Project

## Overview
This guide explains when to use each Claude Code skill available for this project. Use the right skill at the right time to maximize efficiency.

---

## 1. `crossvit-covid19-fyp`
**Complete context for TAR UMT Data Science FYP**

### When to Use:
- **Starting any notebook** - Load full project context first
- **Working with model architecture** - Need CrossViT specifications (input size, dual branches, parameters)
- **Setting up training code** - Need exact hyperparameters (learning rate, batch size, optimizer)
- **Memory issues** - Need RTX 4060 8GB VRAM constraints and workarounds
- **Data preprocessing** - Need exact CLAHE parameters, augmentation rules, normalization values
- **Academic requirements** - Need to know thesis standards, evaluation metrics, citation format

### Examples:
```
# Invoke when:
- "Help me start the modeling notebook"
- "What batch size should I use for CrossViT?"
- "How do I implement CLAHE preprocessing?"
- "What are the class weights for loss function?"
- "I'm getting CUDA out of memory errors"
```

### When NOT to Use:
- General Python/Jupyter questions (use `jupyter` instead)
- Statistical calculations only (use `fyp-statistical-validator` instead)

---

## 2. `fyp-jupyter`
**Data science research workflow for Jupyter notebooks**

### When to Use:
- **Starting a new analysis phase** - Need guidance on what to work on (Exploration vs Experimentation vs Analysis)
- **Following CRISP-DM methodology** - Need step-by-step workflow (Business Understanding → Data Understanding → Preparation → Modeling → Evaluation → Deployment)
- **MLflow experiment tracking** - Setting up tracking for 30+ experiment runs
- **Feature engineering decisions** - Creating interaction features, polynomial features, binning strategies
- **EDA workflow** - Systematic exploratory data analysis approach
- **Weekly planning** - What to prioritize in 10-week FYP timeline
- **Deciding which skill to use** - Need help choosing between different skills

### Examples:
```
# Invoke when:
- "I've finished data cleaning, what should I do next?"
- "How do I set up MLflow for tracking experiments?"
- "What feature engineering techniques should I try?"
- "I'm in Week 5, what should I focus on?"
- "Should I use crossvit-covid19-fyp or fyp-statistical-validator?"
```

### When NOT to Use:
- Project-specific questions (use `crossvit-covid19-fyp` instead)
- Just running code cells (use `jupyter` instead)
- Statistical validation only (use `fyp-statistical-validator` instead)

---

## 3. `fyp-statistical-validator`
**Statistical validation and hypothesis testing for TAR UMT thesis**

### When to Use:
- **Calculating confidence intervals** - Need 95% CI for model metrics (accuracy, precision, recall, F1)
- **Comparing models statistically** - Paired t-test, McNemar's test between CrossViT and baselines
- **Hypothesis testing** - Testing H₁, H₂, H₃, H₄ with p-values and effect sizes
- **Formatting results for thesis** - APA-formatted tables for Chapter 5
- **Multiple comparisons** - Need Bonferroni correction (comparing 6 models)
- **Reproducibility statements** - Generating documentation for Chapter 4
- **Medical metrics** - Calculating sensitivity, specificity, PPV, NPV with proper formatting

### Examples:
```
# Invoke when:
- "Calculate 95% confidence interval for CrossViT accuracy"
- "Compare CrossViT vs ResNet-50 with statistical tests"
- "Test if CrossViT is significantly better than baselines (H₁)"
- "Format results table for thesis Chapter 5"
- "Calculate Cohen's d effect size"
- "Generate reproducibility statement for experimental setup"
```

### When NOT to Use:
- Just training models (use `crossvit-covid19-fyp` instead)
- EDA or preprocessing (use `fyp-jupyter` instead)
- No statistical comparison needed

---

## 4. `jupyter`
**General Jupyter notebook assistance**

### When to Use:
- **Debugging notebook errors** - Cell execution errors, kernel crashes
- **Code optimization** - Making code run faster
- **Visualization help** - Creating plots, charts, confusion matrices
- **Cell management** - Organizing, editing, executing cells
- **General Python/ML questions** - Not specific to this FYP project
- **Quick tasks** - Simple code modifications without needing full project context

### Examples:
```
# Invoke when:
- "Why is this cell throwing a KeyError?"
- "How do I create a confusion matrix heatmap?"
- "Optimize this data loading code"
- "Clear all outputs and restart kernel"
- "Convert this notebook to Python script"
```

### When NOT to Use:
- FYP-specific questions (use `crossvit-covid19-fyp` instead)
- Statistical analysis (use `fyp-statistical-validator` instead)
- Research workflow planning (use `fyp-jupyter` instead)

---

## 5. `skill-writer`
**Expert guide for creating Claude Code skills**

### When to Use:
- **Creating new skills** - Writing custom SKILL.md files for your project
- **Learning skill syntax** - Understanding templates and best practices
- **Validating skills** - Checking if your skill follows correct format
- **Teaching about skills** - Understanding how skills work

### Examples:
```
# Invoke when:
- "Help me create a skill for my project"
- "How do I write a SKILL.md file?"
- "What are best practices for skill creation?"
- "Validate my skill file"
```

### When NOT to Use:
- Actually working on FYP tasks (use other skills)
- General coding help

---

## Quick Decision Tree

```
START: What do I need help with?
│
├─ Setting up/training CrossViT model?
│  └─ Use: crossvit-covid19-fyp
│
├─ Don't know what phase to work on next?
│  └─ Use: fyp-jupyter
│
├─ Comparing models with statistics?
│  └─ Use: fyp-statistical-validator
│
├─ General notebook/Python issue?
│  └─ Use: jupyter
│
└─ Creating a new skill?
   └─ Use: skill-writer
```

---

## Skill Combinations (Use Multiple Skills)

Some tasks benefit from using multiple skills in sequence:

### Scenario 1: Starting Modeling Notebook
1. **`crossvit-covid19-fyp`** - Load model specs, hyperparameters, memory constraints
2. **`jupyter`** - Help write training loop code

### Scenario 2: Validating Results for Thesis
1. **`fyp-statistical-validator`** - Calculate CI, run hypothesis tests
2. **`crossvit-covid19-fyp`** - Ensure results meet TAR UMT requirements

### Scenario 3: Weekly Planning
1. **`fyp-jupyter`** - Understand current phase and next steps
2. **`crossvit-covid19-fyp`** - Get specific implementation details

---

## Pro Tips

1. **Start broad, then narrow**: Use `fyp-jupyter` to plan → `crossvit-covid19-fyp` for specifics
2. **Context matters**: Skills with project context work better for FYP-specific questions
3. **Combine skills**: Don't hesitate to use multiple skills for complex tasks
4. **When in doubt**: Use `fyp-jupyter` to help decide which skill to use next

---

## Common Mistakes to Avoid

❌ Using `jupyter` for FYP-specific questions (use `crossvit-covid19-fyp`)
❌ Using `crossvit-covid19-fyp` for general Python help (use `jupyter`)
❌ Forgetting `fyp-statistical-validator` when writing Chapter 5 results
❌ Not using `fyp-jupyter` for phase planning (leads to inefficient workflow)

✅ Match the skill to the task type
✅ Use project-specific skills for project-specific questions
✅ Use general skills for general programming help
✅ Combine skills when needed
