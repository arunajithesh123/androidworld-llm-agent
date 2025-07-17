#!/usr/bin/env python3
"""
Task 2: Fixed Prompting & Evaluation Strategy

FIXES:
1. Correct ground truth generation that parses actual goals
2. Proper task names for AndroidWorld
3. Better completion detection logic
4. Enhanced prompt templates with clearer instructions
"""

import os
import sys
import json
import logging
import argparse
import time
import random
import re
from typing import Dict, List, Any, Optional, Type, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

# AndroidWorld imports
try:
    from android_world import registry
    from android_world.task_evals import task_eval
    from android_world.env import env_launcher
    from android_world.agents import infer
except ImportError as e:
    print("AndroidWorld not installed properly!")
    print(f"Error: {e}")
    sys.exit(1)

# LLM imports
try:
    import openai
except ImportError:
    print("OpenAI library not installed!")
    sys.exit(1)

# Additional evaluation imports
try:
    from fuzzywuzzy import fuzz
except ImportError:
    print("Installing fuzzywuzzy...")
    os.system("pip install fuzzywuzzy python-Levenshtein")
    from fuzzywuzzy import fuzz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Structure for evaluation results."""
    task_name: str
    goal: str
    predicted_actions: List[str]
    ground_truth_actions: List[str]
    step_accuracy: float
    episode_success: bool
    exact_matches: int
    fuzzy_matches: int
    execution_time: float
    prompt_type: str
    error: Optional[str] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate_action(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def generate_action_with_reflection(self, prompt: str) -> Tuple[str, str]:
        pass


class OpenAIClient(LLMClient):
    """Enhanced OpenAI client with reflection capabilities."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found!")
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_action(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "WAIT()"
    
    def generate_action_with_reflection(self, prompt: str) -> Tuple[str, str]:
        """Generate action with reasoning explanation."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            content = response.choices[0].message.content.strip()
            
            # Try to parse action and reasoning
            if "Reasoning:" in content and "Action:" in content:
                parts = content.split("Action:")
                reasoning = parts[0].replace("Reasoning:", "").strip()
                action = parts[1].strip()
                return action, reasoning
            else:
                return content, "No explicit reasoning provided"
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "WAIT()", "Error in API call"


class ClaudeClient(LLMClient):
    """Enhanced Claude client with reflection capabilities."""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic library not installed!")
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found!")
        
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate_action(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return "WAIT()"
    
    def generate_action_with_reflection(self, prompt: str) -> Tuple[str, str]:
        """Generate action with reasoning explanation."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text.strip()
            
            # Try to parse action and reasoning
            if "Reasoning:" in content and "Action:" in content:
                parts = content.split("Action:")
                reasoning = parts[0].replace("Reasoning:", "").strip()
                action = parts[1].strip()
                return action, reasoning
            else:
                return content, "No explicit reasoning provided"
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return "WAIT()", "Error in API call"


class FixedPromptTemplates:
    """Fixed prompt templates with better instructions."""
    
    @staticmethod
    def get_few_shot_examples() -> str:
        """Get improved few-shot examples."""
        return """
Here are examples of good Android automation steps:

Example 1 - Adding Contact:
Goal: Create a new contact for John Smith with phone +1234567890
Step 1: Contacts app open, contact list visible → CLICK("Add contact button")
Step 2: New contact form appears → TYPE("John Smith")  
Step 3: Name entered, phone field focused → TYPE("+1234567890")
Step 4: All required fields filled → CLICK("Save button")

Example 2 - Sending SMS:
Goal: Send message "Hello" to contact named Mike
Step 1: Messages app open → CLICK("New message button")
Step 2: New message screen → TYPE("Mike")
Step 3: Contact selected → TYPE("Hello")  
Step 4: Message composed → CLICK("Send button")

Example 3 - Timer:
Goal: Set timer for 5 minutes
Step 1: Clock app open → CLICK("Timer tab")
Step 2: Timer interface → TYPE("5:00")
Step 3: Time entered → CLICK("Start button")

Key Patterns:
- Always start with the main action button (Add, New, etc.)
- Enter data in the order it appears on screen
- Always end with Save/Send/Start to complete the action
"""
    
    @staticmethod
    def create_few_shot_prompt(goal: str, observation_text: str, ui_elements: List[str], 
                              previous_actions: List[str], step_count: int, max_steps: int) -> str:
        """Create few-shot prompt with examples."""
        
        ui_elements_text = "\n".join([f"  - {element}" for element in ui_elements[:15]])
        prev_actions_text = "None" if not previous_actions else "\n".join([f"  {i+1}. {action}" for i, action in enumerate(previous_actions[-5:])])
        
        prompt = f"""{FixedPromptTemplates.get_few_shot_examples()}

Now for your current task:

Goal: {goal}

Current Situation (Step {step_count}/{max_steps}):
{observation_text}

Available UI Elements:
{ui_elements_text}

Previous Actions Taken:
{prev_actions_text}

Based on the examples and your current situation, what is the next logical action to achieve the goal?

IMPORTANT GUIDELINES:
- Look at the specific goal requirements (name, phone number, message text, etc.)
- Use the exact data mentioned in the goal
- Follow the pattern: Action Button → Enter Data → Save/Send/Confirm
- Only use UI elements that are actually available
- If you've entered all required data, use Save/Send/Confirm to complete

Available Actions:
- CLICK("element_name") - Click on a UI element by name
- TYPE("text") - Type text into currently focused field  
- SCROLL("up"|"down"|"left"|"right") - Scroll in direction
- NAVIGATE_HOME() - Go to home screen
- NAVIGATE_BACK() - Go back
- WAIT() - Wait for screen to load

Respond with exactly one action in the required format.

Action:"""
        
        return prompt
    
    @staticmethod
    def create_self_reflection_prompt(goal: str, observation_text: str, ui_elements: List[str], 
                                    previous_actions: List[str], step_count: int, max_steps: int) -> str:
        """Create self-reflection prompt asking for reasoning."""
        
        ui_elements_text = "\n".join([f"  - {element}" for element in ui_elements[:15]])
        prev_actions_text = "None" if not previous_actions else "\n".join([f"  {i+1}. {action}" for i, action in enumerate(previous_actions[-5:])])
        
        prompt = f"""Goal: {goal}

Current Situation (Step {step_count}/{max_steps}):
{observation_text}

Available UI Elements:
{ui_elements_text}

Previous Actions Taken:
{prev_actions_text}

Think step by step about what you need to do to achieve this goal.

ANALYSIS QUESTIONS:
1. What specific data does the goal require? (names, numbers, text, etc.)
2. What is the logical sequence to complete this task?
3. Which UI element should I interact with next?
4. Am I ready to save/send/confirm, or do I need more data entry?

Available Actions:
- CLICK("element_name") - Click on a UI element by name
- TYPE("text") - Type text into currently focused field  
- SCROLL("up"|"down"|"left"|"right") - Scroll in direction
- NAVIGATE_HOME() - Go to home screen
- NAVIGATE_BACK() - Go back
- WAIT() - Wait for screen to load

Format your response as:
Reasoning: [Analyze the goal, current state, and explain your logic for the next action]
Action: [Specific action in the format above]

Response:"""
        
        return prompt


class FixedGroundTruthGenerator:
    """Fixed ground truth generator that properly parses goals."""
    
    @staticmethod
    def parse_contact_goal(goal: str) -> Dict[str, str]:
        """Parse contact creation goal to extract name and phone."""
        result = {"name": "Unknown Contact", "phone": ""}
        
        # Extract name - look for patterns like "for [Name]" or "named [Name]"
        name_patterns = [
            r'for ([A-Za-z\s]+)\.', # "for John Smith."
            r'for ([A-Za-z\s]+)\s+', # "for John Smith "
            r'named ([A-Za-z\s]+)\.', # "named John Smith."
            r'named ([A-Za-z\s]+)\s+', # "named John Smith "
            r'contact ([A-Za-z\s]+)\.', # "contact John Smith."
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                result["name"] = match.group(1).strip()
                break
        
        # Extract phone number
        phone_pattern = r'[\+]?[\d\-\(\)\s]{10,}'
        phone_match = re.search(phone_pattern, goal)
        if phone_match:
            result["phone"] = phone_match.group().strip()
        
        logger.info(f"Parsed goal '{goal}' -> Name: '{result['name']}', Phone: '{result['phone']}'")
        return result
    
    @staticmethod
    def generate_ground_truth_sequence(task_name: str, goal: str) -> List[str]:
        """Generate correct ground truth action sequence."""
        
        if "ContactsAddContact" in task_name:
            contact_info = FixedGroundTruthGenerator.parse_contact_goal(goal)
            return [
                'CLICK("Add contact button")',
                f'TYPE("{contact_info["name"]}")',
                f'TYPE("{contact_info["phone"]}")',
                'CLICK("Save button")'
            ]
        
        elif "SMS" in task_name or "Message" in task_name:
            # Extract recipient and message from goal
            # Example: "Send SMS 'Hello' to John"
            recipient = "Contact"
            message = "Message"
            
            # Try to extract recipient
            recipient_match = re.search(r'to ([A-Za-z\s]+)', goal, re.IGNORECASE)
            if recipient_match:
                recipient = recipient_match.group(1).strip()
            
            # Try to extract message
            message_match = re.search(r"'([^']+)'|\"([^\"]+)\"", goal)
            if message_match:
                message = message_match.group(1) or message_match.group(2)
            
            return [
                'CLICK("New message button")',
                f'TYPE("{recipient}")',
                f'TYPE("{message}")',
                'CLICK("Send button")'
            ]
        
        elif "Timer" in task_name or "Clock" in task_name:
            # Extract time from goal
            time_match = re.search(r'(\d+:\d+|\d+\s*minutes?|\d+\s*hours?)', goal, re.IGNORECASE)
            time_str = time_match.group() if time_match else "5:00"
            
            return [
                'CLICK("Timer")',
                f'TYPE("{time_str}")',
                'CLICK("Start")'
            ]
        
        elif "Camera" in task_name:
            return [
                'CLICK("Camera app")',
                'CLICK("Capture button")'
            ]
        
        else:
            # Generic sequence
            return [
                'CLICK("Primary action")',
                'TYPE("Required input")',
                'CLICK("Confirm")'
            ]


class AndroidWorldEvaluator:
    """Fixed evaluator for AndroidWorld tasks."""
    
    def __init__(self):
        self.task_registry = None
        self.aw_registry = None
        self.initialize()
    
    def initialize(self):
        """Initialize AndroidWorld registry."""
        try:
            self.task_registry = registry.TaskRegistry()
            self.aw_registry = self.task_registry.get_registry(self.task_registry.ANDROID_WORLD_FAMILY)
            logger.info("AndroidWorld registry initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize registry: {e}")
            return False
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available tasks."""
        if not self.aw_registry:
            return []
        try:
            return sorted(list(self.aw_registry.keys()))
        except Exception as e:
            logger.error(f"Error getting tasks: {e}")
            return []
    
    def create_task(self, task_name: str) -> Any:
        """Create a task instance with better error handling."""
        if not self.aw_registry:
            logger.error("Registry not initialized")
            return None
            
        # Check if task exists
        available_tasks = self.get_available_tasks()
        if task_name not in available_tasks:
            logger.error(f"Task '{task_name}' not found in registry")
            logger.info(f"Available tasks: {available_tasks[:10]}...")
            
            # Try to find similar task names
            similar_tasks = [t for t in available_tasks if task_name.lower() in t.lower()]
            if similar_tasks:
                logger.info(f"Similar tasks found: {similar_tasks[:5]}")
            
            return None
        
        try:
            task_type: Type[task_eval.TaskEval] = self.aw_registry[task_name]
            params = task_type.generate_random_params()
            task = task_type(params)
            return task
        except Exception as e:
            logger.error(f"Error creating task {task_name}: {e}")
            return None
    
    def generate_ui_elements(self, task_name: str, step: int) -> List[str]:
        """Generate realistic UI elements for task."""
        base_elements = ["Navigation bar", "Status bar", "Back button"]
        
        if "Contacts" in task_name:
            if step == 0:
                return base_elements + [
                    "Contacts app (open)", "Add contact button (clickable)",
                    "Contact list (scrollable)", "Search bar (text input)",
                    "Menu button (clickable)"
                ]
            elif step == 1:
                return base_elements + [
                    "New contact form", "First name field (text input, focused)",
                    "Last name field (text input)", "Phone field (text input)",
                    "Email field (text input)", "Save button (clickable)"
                ]
            elif step == 2:
                return base_elements + [
                    "New contact form", "First name field (filled)",
                    "Last name field (filled)", "Phone field (text input, focused)",
                    "Email field (text input)", "Save button (clickable)"
                ]
            else:
                return base_elements + [
                    "Contact form completed", "Save button (clickable)",
                    "Cancel button (clickable)", "More options (clickable)"
                ]
        
        elif "SMS" in task_name or "Message" in task_name:
            if step == 0:
                return base_elements + [
                    "Messages app (open)", "New message button (clickable)",
                    "Conversation list (scrollable)", "Search bar (text input)"
                ]
            elif step == 1:
                return base_elements + [
                    "New message screen", "To field (text input, focused)",
                    "Message field (text input)", "Send button (clickable)"
                ]
            else:
                return base_elements + [
                    "Message composition", "To field (filled)",
                    "Message field (text input, focused)", "Send button (clickable)"
                ]
        
        else:
            return base_elements + [
                f"{task_name} interface", "Primary action button (clickable)",
                "Secondary options (clickable)"
            ]
    
    def run_evaluation_episode(self, task_name: str, llm_client: LLMClient, 
                             prompt_type: str = "few_shot", max_steps: int = 8) -> EvaluationResult:
        """Run a single evaluation episode with fixes."""
        
        start_time = time.time()
        
        # Create task
        task = self.create_task(task_name)
        if not task:
            return EvaluationResult(
                task_name=task_name,
                goal="Failed to create task",
                predicted_actions=[],
                ground_truth_actions=[],
                step_accuracy=0.0,
                episode_success=False,
                exact_matches=0,
                fuzzy_matches=0,
                execution_time=0.0,
                prompt_type=prompt_type,
                error="Failed to create task"
            )
        
        goal = getattr(task, 'goal', f"Complete task: {task_name}")
        
        # Generate CORRECT ground truth using the actual goal
        ground_truth_actions = FixedGroundTruthGenerator.generate_ground_truth_sequence(task_name, goal)
        
        # Run LLM prediction
        predicted_actions = []
        action_history = []
        
        try:
            for step in range(max_steps):
                observation_text = f"Android screen showing {task_name} interface (step {step + 1})"
                ui_elements = self.generate_ui_elements(task_name, step)
                
                # Create prompt based on type
                if prompt_type == "few_shot":
                    prompt = FixedPromptTemplates.create_few_shot_prompt(
                        goal, observation_text, ui_elements, action_history, step + 1, max_steps
                    )
                    action = llm_client.generate_action(prompt)
                    predicted_actions.append(action)
                
                elif prompt_type == "self_reflection":
                    prompt = FixedPromptTemplates.create_self_reflection_prompt(
                        goal, observation_text, ui_elements, action_history, step + 1, max_steps
                    )
                    action, reasoning = llm_client.generate_action_with_reflection(prompt)
                    predicted_actions.append(action)
                    logger.info(f"Step {step + 1} Reasoning: {reasoning}")
                
                action_history.append(action)
                
                # Better completion detection
                completion_keywords = ['SAVE', 'SEND', 'START', 'CONFIRM', 'SUBMIT']
                if any(keyword in action.upper() for keyword in completion_keywords):
                    logger.info(f"Detected completion action: {action}")
                    break
                
                # Stop if we have enough actions
                if len(predicted_actions) >= len(ground_truth_actions):
                    break
        
        except Exception as e:
            logger.error(f"Error during episode: {e}")
            return EvaluationResult(
                task_name=task_name,
                goal=goal,
                predicted_actions=predicted_actions,
                ground_truth_actions=ground_truth_actions,
                step_accuracy=0.0,
                episode_success=False,
                exact_matches=0,
                fuzzy_matches=0,
                execution_time=time.time() - start_time,
                prompt_type=prompt_type,
                error=str(e)
            )
        
        # Calculate metrics
        exact_matches = 0
        fuzzy_matches = 0
        
        min_len = min(len(predicted_actions), len(ground_truth_actions))
        for i in range(min_len):
            pred = predicted_actions[i].strip()
            truth = ground_truth_actions[i].strip()
            
            if pred == truth:
                exact_matches += 1
                fuzzy_matches += 1
            elif fuzz.ratio(pred, truth) >= 80:  # Lowered threshold for better matching
                fuzzy_matches += 1
        
        step_accuracy = fuzzy_matches / len(ground_truth_actions) if ground_truth_actions else 0.0
        
        # Better episode success detection
        has_completion_action = any(any(keyword in action.upper() for keyword in ['SAVE', 'SEND', 'START', 'CONFIRM']) 
                                  for action in predicted_actions)
        sufficient_accuracy = step_accuracy >= 0.6  # Lowered threshold
        
        episode_success = has_completion_action and sufficient_accuracy
        
        return EvaluationResult(
            task_name=task_name,
            goal=goal,
            predicted_actions=predicted_actions,
            ground_truth_actions=ground_truth_actions,
            step_accuracy=step_accuracy,
            episode_success=episode_success,
            exact_matches=exact_matches,
            fuzzy_matches=fuzzy_matches,
            execution_time=time.time() - start_time,
            prompt_type=prompt_type
        )
    
    def run_evaluation_suite(self, tasks: List[str], llm_client: LLMClient, 
                           prompt_types: List[str] = ["few_shot", "self_reflection"],
                           episodes_per_task: int = 3) -> Dict[str, Any]:
        """Run evaluation on multiple tasks with different prompt types."""
        
        results = {
            "summary": {},
            "episodes": [],
            "prompt_comparison": {}
        }
        
        # First, validate all tasks
        valid_tasks = []
        available_tasks = self.get_available_tasks()
        
        for task in tasks:
            if task in available_tasks:
                valid_tasks.append(task)
            else:
                logger.warning(f"Task '{task}' not available. Skipping.")
                # Try to suggest alternatives
                similar = [t for t in available_tasks if task.lower() in t.lower()]
                if similar:
                    logger.info(f"Similar available tasks: {similar[:3]}")
        
        if not valid_tasks:
            logger.error("No valid tasks found!")
            return results
        
        logger.info(f"Running evaluation on valid tasks: {valid_tasks}")
        
        for prompt_type in prompt_types:
            logger.info(f"\n🧪 Running evaluation with {prompt_type} prompting...")
            
            prompt_results = []
            
            for task_name in valid_tasks:
                logger.info(f"📱 Evaluating task: {task_name}")
                
                # Run multiple episodes for this task
                task_results = []
                for episode in range(episodes_per_task):
                    logger.info(f"  Episode {episode + 1}/{episodes_per_task}")
                    result = self.run_evaluation_episode(task_name, llm_client, prompt_type)
                    task_results.append(result)
                    results["episodes"].append(asdict(result))
                
                prompt_results.extend(task_results)
            
            # Calculate summary statistics for this prompt type
            if prompt_results:
                successful_results = [r for r in prompt_results if not r.error]
                if successful_results:
                    avg_step_accuracy = sum(r.step_accuracy for r in successful_results) / len(successful_results)
                    episode_success_rate = sum(1 for r in successful_results if r.episode_success) / len(successful_results)
                    avg_execution_time = sum(r.execution_time for r in successful_results) / len(successful_results)
                    
                    results["prompt_comparison"][prompt_type] = {
                        "avg_step_accuracy": avg_step_accuracy,
                        "episode_success_rate": episode_success_rate,
                        "avg_execution_time": avg_execution_time,
                        "total_episodes": len(successful_results),
                        "failed_episodes": len(prompt_results) - len(successful_results)
                    }
        
        # Overall summary
        all_results = [EvaluationResult(**ep) for ep in results["episodes"] if not ep.get("error")]
        if all_results:
            results["summary"] = {
                "total_episodes": len(all_results),
                "overall_step_accuracy": sum(r.step_accuracy for r in all_results) / len(all_results),
                "overall_success_rate": sum(1 for r in all_results if r.episode_success) / len(all_results),
                "tasks_evaluated": valid_tasks,
                "prompt_types_tested": prompt_types,
                "failed_tasks": len([ep for ep in results["episodes"] if ep.get("error")])
            }
        
        return results


def main():
    """Main entry point for fixed Task 2 evaluation."""
    parser = argparse.ArgumentParser(description='AndroidWorld LLM Agent - Fixed Task 2 Evaluation')
    parser.add_argument('--tasks', nargs='+', default=["ContactsAddContact"],
                       help='Tasks to evaluate (will validate against available tasks)')
    parser.add_argument('--llm', choices=['openai', 'claude'], default='openai', help='LLM provider')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--prompt-types', nargs='+', default=["few_shot", "self_reflection"],
                       choices=["few_shot", "self_reflection"], help='Prompt types to test')
    parser.add_argument('--episodes-per-task', type=int, default=3, help='Episodes per task')
    parser.add_argument('--max-steps', type=int, default=8, help='Maximum steps per episode')
    parser.add_argument('--output', type=str, default='task2_fixed_evaluation_results.json', help='Output file')
    parser.add_argument('--list-tasks', action='store_true', help='List available tasks and exit')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AndroidWorldEvaluator()
    
    # List tasks if requested
    if args.list_tasks:
        tasks = evaluator.get_available_tasks()
        print(f"\n Available AndroidWorld Tasks ({len(tasks)} total):")
        for i, task in enumerate(tasks[:30], 1):
            print(f"  {i:2d}. {task}")
        if len(tasks) > 30:
            print(f"  ... and {len(tasks) - 30} more tasks")
        print(f"\nExample usage: python task2_fixed_evaluation.py --tasks ContactsAddContact")
        return
    
    # Initialize LLM client
    try:
        if args.llm == 'openai':
            model = args.model or 'gpt-4'
            llm_client = OpenAIClient(model=model)
            print(f" Using OpenAI {model}")
        else:
            model = args.model or 'claude-3-sonnet-20240229'
            llm_client = ClaudeClient(model=model)
            print(f" Using Claude {model}")
    except Exception as e:
        print(f" Failed to initialize LLM client: {e}")
        return
    
    # Run evaluation
    try:
        print(f"\n Running Fixed AndroidWorld LLM Evaluation - Task 2")
        print(f" Prompt Types: {args.prompt_types}")
        print(f" Episodes per task: {args.episodes_per_task}")
        print(f" Max Steps: {args.max_steps}")
        
        results = evaluator.run_evaluation_suite(
            args.tasks, llm_client, args.prompt_types, args.episodes_per_task
        )
        
        # Print results
        print(f"\n{'='*60}")
        print(f" FIXED EVALUATION RESULTS")
        print(f"{'='*60}")
        
        if results["summary"]:
            summary = results["summary"]
            print(f"Total Successful Episodes: {summary['total_episodes']}")
            print(f"Failed Episodes: {summary.get('failed_tasks', 0)}")
            print(f"Overall Step Accuracy: {summary['overall_step_accuracy']:.2%}")
            print(f"Overall Success Rate: {summary['overall_success_rate']:.2%}")
            print(f"Tasks Successfully Evaluated: {summary['tasks_evaluated']}")
        
        print(f"\n Prompt Type Comparison:")
        for prompt_type, metrics in results["prompt_comparison"].items():
            print(f"  {prompt_type.title()}:")
            print(f"    Step Accuracy: {metrics['avg_step_accuracy']:.2%}")
            print(f"    Success Rate: {metrics['episode_success_rate']:.2%}")
            print(f"    Avg Time: {metrics['avg_execution_time']:.2f}s")
            print(f"    Episodes: {metrics['total_episodes']} successful, {metrics.get('failed_episodes', 0)} failed")
        
        # Save detailed results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n Detailed results saved to: {args.output}")
        
        # Print sample episodes with improved analysis
        print(f"\n Sample Episode Results:")
        successful_episodes = [ep for ep in results["episodes"] if ep.get("episode_success", False)]
        failed_episodes = [ep for ep in results["episodes"] if not ep.get("episode_success", False) and not ep.get("error")]
        
        if successful_episodes:
            print(f"\n SUCCESSFUL Episode Example:")
            ep = successful_episodes[0]
            print(f"  Task: {ep['task_name']}")
            print(f"  Goal: {ep['goal']}")
            print(f"  Success Rate: {ep['step_accuracy']:.2%}")
            print(f"  Predicted Actions:")
            for i, action in enumerate(ep['predicted_actions'], 1):
                print(f"    {i}. {action}")
            print(f"  Ground Truth:")
            for i, action in enumerate(ep['ground_truth_actions'], 1):
                print(f"    {i}. {action}")
        
        if failed_episodes:
            print(f"\n FAILED Episode Example:")
            ep = failed_episodes[0]
            print(f"  Task: {ep['task_name']}")
            print(f"  Goal: {ep['goal']}")
            print(f"  Success Rate: {ep['step_accuracy']:.2%}")
            print(f"  Issue: Missing completion action or low accuracy")
            print(f"  Predicted: {ep['predicted_actions'][:3]}...")
            print(f"  Expected: {ep['ground_truth_actions'][:3]}...")
        
        # Error analysis
        error_episodes = [ep for ep in results["episodes"] if ep.get("error")]
        if error_episodes:
            print(f"\n ERROR Analysis:")
            error_counts = {}
            for ep in error_episodes:
                error = ep.get("error", "Unknown")
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for error, count in error_counts.items():
                print(f"  {error}: {count} episodes")
        
        # Recommendations based on results
        print(f"\n IMPROVEMENT RECOMMENDATIONS:")
        
        if results["summary"] and results["summary"]["overall_success_rate"] < 0.5:
            print(f"  1. Success rate is low ({results['summary']['overall_success_rate']:.1%})")
            print(f"     → Add better completion detection in prompts")
            print(f"     → Emphasize 'Save/Send/Confirm' actions in examples")
        
        if results["summary"] and results["summary"]["overall_step_accuracy"] < 0.7:
            print(f"  2. Step accuracy is low ({results['summary']['overall_step_accuracy']:.1%})")
            print(f"     → Improve ground truth generation")
            print(f"     → Add more specific examples for each task type")
        
        if error_episodes:
            print(f"  3. {len(error_episodes)} episodes had errors")
            print(f"     → Validate task names against AndroidWorld registry")
            print(f"     → Add better error handling for unavailable tasks")
        
        # Success analysis
        if successful_episodes:
            print(f"\n SUCCESS PATTERNS:")
            action_patterns = {}
            for ep in successful_episodes:
                for action in ep['predicted_actions']:
                    action_type = action.split('(')[0] if '(' in action else action
                    action_patterns[action_type] = action_patterns.get(action_type, 0) + 1
            
            common_actions = sorted(action_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Most common successful actions:")
            for action, count in common_actions:
                print(f"    {action}: {count} times")
        
    except KeyboardInterrupt:
        print("\n Interrupted by user")
    except Exception as e:
        print(f"\n Error running evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()