# Self-Reflection Prompting Template

## Template Structure
Goal: {goal}
Current Situation (Step {step_count}/{max_steps}):
{observation_text}
Available UI Elements:
{ui_elements_text}
Previous Actions Taken:
{prev_actions_text}
Think step by step about what you need to do to achieve this goal.
ANALYSIS QUESTIONS:

What specific data does the goal require? (names, numbers, text, etc.)
What is the logical sequence to complete this task?
Which UI element should I interact with next?
Am I ready to save/send/confirm, or do I need more data entry?

Available Actions:

CLICK("element_name") - Click on a UI element by name
TYPE("text") - Type text into currently focused field
SCROLL("up"|"down"|"left"|"right") - Scroll in direction
NAVIGATE_HOME() - Go to home screen
NAVIGATE_BACK() - Go back
WAIT() - Wait for screen to load

Format your response as:
Reasoning: [Analyze the goal, current state, and explain your logic for the next action]
Action: [Specific action in the format above]
Response:

## Usage Notes
- Use this template when you want detailed reasoning from the LLM
- Expect slower execution but richer explanations
- Good for debugging and understanding agent decision-making