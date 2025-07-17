# Memory-Enhanced Prompting Template

## Template with Memory Context
Goal: {goal}
Current Observation: {observation}
Available UI Elements:
{ui_elements}
Memory Context:
{memory_context}
Recent successful actions:
{recent_actions}
Relevant insights:
{semantic_insights}
Previous errors to avoid:
{error_patterns}
What is the best action to take towards achieving the goal?
Available Actions:

CLICK("element_name") - Click on UI element
TYPE("text") - Type text into field
SCROLL("direction") - Scroll up/down/left/right
WAIT() - Wait for screen

Action:

## Memory Types
- **Episodic**: Recent actions and their outcomes
- **Semantic**: Task-specific insights and patterns
- **Working**: Current context and goal state
- **Error**: Failed actions to avoid repeating