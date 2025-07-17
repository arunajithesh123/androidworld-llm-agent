# Prompt Engineering Guidelines

## Best Practices

### 1. Clear Goal Definition
- Always include specific, actionable goals
- Use exact data (names, phone numbers, etc.)
- Specify completion criteria

### 2. Context Provision
- Current UI state and available elements
- Previous actions taken
- Step count and progress indicators

### 3. Action Format Consistency
- Use standardized action syntax: CLICK("element"), TYPE("text")
- Include available action types
- Provide clear examples

### 4. Reasoning Encouragement
- Ask for step-by-step thinking
- Include analysis questions for complex tasks
- Request confidence levels when appropriate

## Template Selection Guide

- **Basic Template**: Simple, fast execution
- **Few-Shot**: When you want consistent patterns
- **Self-Reflection**: For detailed reasoning and debugging
- **Memory-Enhanced**: For learning and context retention

## Common Pitfalls to Avoid

1. Vague or ambiguous goals
2. Missing UI element information
3. Inconsistent action formats
4. No completion validation
5. Lack of error handling guidance