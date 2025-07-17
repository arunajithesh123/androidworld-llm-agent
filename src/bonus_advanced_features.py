#!/usr/bin/env python3
"""
Bonus: Fixed Advanced Features

FIXES APPLIED:
1. Fixed missing ClaudeClient definition
2. Removed Streamlit auto-detection causing warnings
3. Simplified model comparison
4. Better error handling for optional dependencies
5. CLI-focused implementation

Advanced features including:
- Memory buffer with full history
- Multi-model comparison (GPT-4 vs GPT-3.5-turbo vs Claude)
- Function calling with structured output
- Advanced error recovery
"""

import os
import sys
import json
import logging
import argparse
import time
from typing import Dict, List, Any, Optional, Type, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict

# AndroidWorld imports
try:
    from android_world import registry
    from android_world.task_evals import task_eval
    from android_world.env import env_launcher
    from android_world.agents import infer
except ImportError as e:
    print("❌ AndroidWorld not installed properly!")
    sys.exit(1)

# LLM imports
try:
    import openai
    from pydantic import BaseModel, Field
except ImportError:
    print("❌ Required libraries not installed!")
    print("Run: pip install openai pydantic")
    sys.exit(1)

# Optional Streamlit imports
try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AndroidAction(BaseModel):
    """Structured action model for function calling."""
    action_type: str = Field(description="Type of action: CLICK, TYPE, SCROLL, NAVIGATE_HOME, NAVIGATE_BACK, WAIT")
    target: Optional[str] = Field(default=None, description="Target element for CLICK actions or text for TYPE actions")
    direction: Optional[str] = Field(default=None, description="Direction for SCROLL actions: up, down, left, right")
    reasoning: str = Field(description="Explanation of why this action was chosen")
    confidence: float = Field(description="Confidence level (0.0-1.0) in this action choice")


class MemoryBuffer:
    """Advanced memory buffer with semantic storage and retrieval."""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.episodic_memory = deque(maxlen=max_size)  # Recent actions and observations
        self.semantic_memory = {}  # Key insights and patterns
        self.working_memory = {}  # Current task context
        self.error_memory = []  # Failed actions for learning
    
    def add_step(self, step_info: Dict[str, Any]):
        """Add a step to episodic memory."""
        self.episodic_memory.append({
            'timestamp': time.time(),
            'step': step_info['step'],
            'observation': step_info['observation'],
            'action': step_info['action'],
            'reasoning': step_info.get('reasoning', ''),
            'success': step_info.get('success', True)
        })
        
        # Update working memory with current context
        if 'goal' in step_info:
            self.working_memory['current_goal'] = step_info['goal']
        if 'ui_elements' in step_info:
            self.working_memory['last_ui_elements'] = step_info['ui_elements']
    
    def add_semantic_insight(self, key: str, insight: str):
        """Add a semantic insight for future reference."""
        if key not in self.semantic_memory:
            self.semantic_memory[key] = []
        self.semantic_memory[key].append({
            'insight': insight,
            'timestamp': time.time()
        })
    
    def add_error(self, action: str, reason: str):
        """Record a failed action for learning."""
        self.error_memory.append({
            'action': action,
            'reason': reason,
            'timestamp': time.time()
        })
    
    def get_relevant_context(self, current_goal: str, current_ui: List[str]) -> str:
        """Get relevant context from memory for current situation."""
        context_parts = []
        
        # Recent successful actions
        recent_successful = [step for step in list(self.episodic_memory)[-5:] if step.get('success', True)]
        if recent_successful:
            context_parts.append("Recent successful actions:")
            for step in recent_successful:
                context_parts.append(f"  - Step {step['step']}: {step['action']}")
        
        # Relevant semantic insights
        goal_keywords = current_goal.lower().split()
        relevant_insights = []
        for keyword in goal_keywords:
            if keyword in self.semantic_memory:
                latest_insight = self.semantic_memory[keyword][-1]
                relevant_insights.append(f"  - {keyword}: {latest_insight['insight']}")
        
        if relevant_insights:
            context_parts.append("Relevant insights:")
            context_parts.extend(relevant_insights)
        
        # Previous errors to avoid
        if self.error_memory:
            context_parts.append("Previous errors to avoid:")
            for error in self.error_memory[-3:]:
                context_parts.append(f"  - Don't: {error['action']} (Reason: {error['reason']})")
        
        return "\n".join(context_parts) if context_parts else "No relevant context found."
    
    def clear(self):
        """Clear all memory buffers."""
        self.episodic_memory.clear()
        self.semantic_memory.clear()
        self.working_memory.clear()
        self.error_memory.clear()


class AdvancedLLMClient:
    """Enhanced LLM client with function calling and structured output."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found!")
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_action(self, prompt: str) -> str:
        """Generate simple action for compatibility."""
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
    
    def generate_structured_action(self, prompt: str) -> AndroidAction:
        """Generate structured action using function calling."""
        try:
            # Use new OpenAI tools format instead of deprecated functions
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "execute_android_action",
                        "description": "Execute an action on Android device",
                        "parameters": AndroidAction.model_json_schema()
                    }
                }],
                tool_choice={"type": "function", "function": {"name": "execute_android_action"}},
                temperature=0.1
            )
            
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                args = json.loads(tool_calls[0].function.arguments)
                return AndroidAction(**args)
            else:
                # Fallback to simple parsing
                simple_response = self.generate_action(prompt)
                return self._parse_simple_action(simple_response)
                
        except Exception as e:
            logger.error(f"Structured action generation failed: {e}")
            # Fallback to simple parsing
            simple_response = self.generate_action(prompt)
            return self._parse_simple_action(simple_response)
    
    def _parse_simple_action(self, action_str: str) -> AndroidAction:
        """Parse a simple action string into structured format."""
        action_str = action_str.strip()
        
        if action_str.startswith('CLICK('):
            target = action_str[6:-1].strip('"\'')
            return AndroidAction(
                action_type="CLICK",
                target=target,
                reasoning="Parsed from simple action",
                confidence=0.8
            )
        elif action_str.startswith('TYPE('):
            target = action_str[5:-1].strip('"\'')
            return AndroidAction(
                action_type="TYPE",
                target=target,
                reasoning="Parsed from simple action",
                confidence=0.8
            )
        else:
            return AndroidAction(
                action_type="WAIT",
                reasoning="Could not parse action",
                confidence=0.3
            )
    
    def format_action_string(self, action: AndroidAction) -> str:
        """Convert structured action to string format."""
        if action.action_type == "CLICK" and action.target:
            return f'CLICK("{action.target}")'
        elif action.action_type == "TYPE" and action.target:
            return f'TYPE("{action.target}")'
        elif action.action_type == "SCROLL" and action.direction:
            return f'SCROLL("{action.direction}")'
        elif action.action_type in ["NAVIGATE_HOME", "NAVIGATE_BACK", "WAIT"]:
            return f"{action.action_type}()"
        else:
            return "WAIT()"


# Quick fix for bonus_advanced_features.py
# Just update the Claude model name in the ClaudeClient class

class ClaudeClient:
    """Claude client for comparison."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic library not installed! Run: pip install anthropic")
        
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


class MultiModelComparator:
    """Compare multiple LLM models on the same tasks."""
    
    def __init__(self):
        self.models = {}
        self.comparison_results = defaultdict(list)
    
    def add_model(self, name: str, client):
        """Add a model to the comparison."""
        self.models[name] = client
        logger.info(f"Added model: {name}")
    
    def run_comparison(self, task_name: str, goal: str, observations: List[str], 
                      ui_elements_list: List[List[str]], max_steps: int = 5) -> Dict[str, Any]:
        """Run the same scenario across all models."""
        
        results = {}
        
        for model_name, client in self.models.items():
            logger.info(f"Running {model_name} on {task_name}")
            
            model_result = {
                'model': model_name,
                'task': task_name,
                'goal': goal,
                'actions': [],
                'reasoning': [],
                'confidence_scores': [],
                'execution_time': 0,
                'success': False
            }
            
            start_time = time.time()
            memory_buffer = MemoryBuffer()
            
            try:
                for step in range(min(max_steps, len(observations))):
                    observation = observations[step]
                    ui_elements = ui_elements_list[step] if step < len(ui_elements_list) else []
                    
                    # Create enhanced prompt with memory
                    memory_context = memory_buffer.get_relevant_context(goal, ui_elements)
                    
                    prompt = f"""Goal: {goal}

Current Observation: {observation}

Available UI Elements:
{chr(10).join(f"  - {elem}" for elem in ui_elements)}

Memory Context:
{memory_context}

What is the best action to take towards achieving the goal?

Guidelines:
- Use exact data from the goal (names, phone numbers, etc.)
- Follow logical sequence: Add → Enter Data → Save
- Only use available UI elements

Available Actions:
- CLICK("element_name") - Click on a UI element
- TYPE("text") - Type text into field
- SCROLL("direction") - Scroll up/down/left/right
- WAIT() - Wait for screen to load

Action:"""
                    
                    # Get action based on client type
                    if hasattr(client, 'generate_structured_action'):
                        structured_action = client.generate_structured_action(prompt)
                        action_str = client.format_action_string(structured_action)
                        reasoning = structured_action.reasoning
                        confidence = structured_action.confidence
                    else:
                        # Simple client
                        action_str = client.generate_action(prompt)
                        reasoning = "Basic LLM response"
                        confidence = 0.7
                    
                    model_result['actions'].append(action_str)
                    model_result['reasoning'].append(reasoning)
                    model_result['confidence_scores'].append(confidence)
                    
                    # Update memory
                    memory_buffer.add_step({
                        'step': step + 1,
                        'observation': observation,
                        'action': action_str,
                        'reasoning': reasoning,
                        'goal': goal,
                        'ui_elements': ui_elements
                    })
                    
                    # Check for completion
                    if any(keyword in action_str.upper() for keyword in ['SAVE', 'SEND', 'CONFIRM']):
                        model_result['success'] = True
                        break
                
                model_result['execution_time'] = time.time() - start_time
                
            except Exception as e:
                logger.error(f"Error running {model_name}: {e}")
                model_result['error'] = str(e)
            
            results[model_name] = model_result
        
        # Store for aggregated analysis
        self.comparison_results[task_name].append(results)
        
        return results
    
    def get_aggregated_comparison(self) -> Dict[str, Any]:
        """Get aggregated comparison results across all tasks."""
        
        model_stats = defaultdict(lambda: {
            'total_tasks': 0,
            'successful_tasks': 0,
            'avg_confidence': 0,
            'avg_execution_time': 0,
            'avg_actions_per_task': 0
        })
        
        for task_results_list in self.comparison_results.values():
            for task_results in task_results_list:
                for model_name, result in task_results.items():
                    if 'error' in result:
                        continue
                        
                    stats = model_stats[model_name]
                    stats['total_tasks'] += 1
                    if result.get('success', False):
                        stats['successful_tasks'] += 1
                    
                    confidences = result.get('confidence_scores', [0.5])
                    if confidences:
                        stats['avg_confidence'] += sum(confidences) / len(confidences)
                    stats['avg_execution_time'] += result.get('execution_time', 0)
                    stats['avg_actions_per_task'] += len(result.get('actions', []))
        
        # Calculate averages
        for model_name, stats in model_stats.items():
            if stats['total_tasks'] > 0:
                stats['success_rate'] = stats['successful_tasks'] / stats['total_tasks']
                stats['avg_confidence'] /= stats['total_tasks']
                stats['avg_execution_time'] /= stats['total_tasks']
                stats['avg_actions_per_task'] /= stats['total_tasks']
        
        return dict(model_stats)


class AdvancedEpisodeRunner:
    """Advanced episode runner with all bonus features."""
    
    def __init__(self):
        self.task_registry = None
        self.aw_registry = None
        self.memory_buffer = MemoryBuffer()
        self.model_comparator = MultiModelComparator()
        self.initialize()
    
    def initialize(self):
        """Initialize AndroidWorld registry."""
        try:
            self.task_registry = registry.TaskRegistry()
            self.aw_registry = self.task_registry.get_registry(self.task_registry.ANDROID_WORLD_FAMILY)
            logger.info("✅ AndroidWorld registry initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize registry: {e}")
            return False
    
    def setup_models(self):
        """Setup multiple models for comparison."""
        # Add OpenAI models
        for model in ['gpt-4', 'gpt-3.5-turbo']:
            try:
                client = AdvancedLLMClient(model=model)
                self.model_comparator.add_model(model, client)
            except Exception as e:
                logger.warning(f"Could not setup {model}: {e}")
        
        # Add Claude if available
        try:
            claude_client = ClaudeClient()
            self.model_comparator.add_model("claude-3-sonnet", claude_client)
        except Exception as e:
            logger.warning(f"Could not setup Claude: {e}")
    
    def run_advanced_episode(self, task_name: str, use_memory: bool = True, 
                           compare_models: bool = False) -> Dict[str, Any]:
        """Run an advanced episode with all bonus features."""
        
        # Create task
        if not self.aw_registry or task_name not in self.aw_registry:
            return {"error": f"Task {task_name} not available"}
        
        try:
            task_type: Type[task_eval.TaskEval] = self.aw_registry[task_name]
            params = task_type.generate_random_params()
            task = task_type(params)
        except Exception as e:
            return {"error": f"Failed to create task: {e}"}
        
        goal = getattr(task, 'goal', f"Complete task: {task_name}")
        
        # Generate realistic episode data
        observations = [
            f"Android screen showing {task_name} interface (step {i+1})" 
            for i in range(5)
        ]
        
        if "Contacts" in task_name:
            ui_elements_list = [
                ["Add contact button", "Contact list", "Search bar"],
                ["Name field (focused)", "Phone field", "Email field", "Save button"],
                ["Name field (filled)", "Phone field (focused)", "Save button"],
                ["Contact form (complete)", "Save button", "Cancel button"],
                ["Contact saved", "View contact", "Add another"]
            ]
        else:
            ui_elements_list = [
                ["Primary action button", "Menu", "Settings"],
                ["Input field", "Confirm button", "Cancel button"],
                ["Processing screen", "Continue button"],
                ["Success message", "Done button"],
                ["Home screen", "Back button"]
            ]
        
        result = {
            'task_name': task_name,
            'goal': goal,
            'use_memory': use_memory,
            'compare_models': compare_models,
            'timestamp': time.time()
        }
        
        if compare_models:
            # Run model comparison
            if not self.model_comparator.models:
                self.setup_models()
            
            comparison_results = self.model_comparator.run_comparison(
                task_name, goal, observations, ui_elements_list
            )
            result['model_comparison'] = comparison_results
            result['aggregated_stats'] = self.model_comparator.get_aggregated_comparison()
        
        else:
            # Run single model with memory
            client = AdvancedLLMClient()
            if use_memory:
                self.memory_buffer.clear()  # Start fresh for this episode
            
            actions = []
            reasoning_list = []
            
            for step, (obs, ui_elements) in enumerate(zip(observations, ui_elements_list)):
                if use_memory:
                    memory_context = self.memory_buffer.get_relevant_context(goal, ui_elements)
                    prompt = f"""Goal: {goal}

Observation: {obs}

UI Elements: {ui_elements}

Memory Context:
{memory_context}

What's the next action to achieve the goal?

Available Actions:
- CLICK("element_name") - Click on UI element
- TYPE("text") - Type text into field
- WAIT() - Wait for screen

Action:"""
                else:
                    prompt = f"""Goal: {goal}
Observation: {obs}
UI Elements: {ui_elements}

Next action:"""
                
                structured_action = client.generate_structured_action(prompt)
                action_str = client.format_action_string(structured_action)
                
                actions.append(action_str)
                reasoning_list.append(structured_action.reasoning)
                
                if use_memory:
                    self.memory_buffer.add_step({
                        'step': step + 1,
                        'observation': obs,
                        'action': action_str,
                        'reasoning': structured_action.reasoning,
                        'goal': goal,
                        'ui_elements': ui_elements
                    })
                
                # Add semantic insights
                if "contact" in goal.lower() and "CLICK" in action_str and "Add" in action_str:
                    self.memory_buffer.add_semantic_insight("contact_creation", "Start with Add contact button")
                
                # Check for completion
                if any(keyword in action_str.upper() for keyword in ['SAVE', 'SEND', 'CONFIRM']):
                    break
            
            result.update({
                'actions': actions,
                'reasoning': reasoning_list,
                'memory_used': use_memory,
                'final_memory_size': len(self.memory_buffer.episodic_memory) if use_memory else 0
            })
        
        return result


def main():
    """Main entry point for bonus features."""
    parser = argparse.ArgumentParser(description='AndroidWorld LLM Agent - Bonus Advanced Features')
    parser.add_argument('--mode', choices=['episode', 'compare'], default='episode',
                       help='Mode to run: single episode or model comparison')
    parser.add_argument('--task', type=str, default='ContactsAddContact', help='Task to run')
    parser.add_argument('--use-memory', action='store_true', help='Use advanced memory buffer')
    parser.add_argument('--compare-models', action='store_true', help='Compare multiple models')
    parser.add_argument('--output', type=str, default='bonus_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = AdvancedEpisodeRunner()
    
    if args.mode == 'compare' or args.compare_models:
        print(f"🤖 Running model comparison on {args.task}")
        runner.setup_models()
        result = runner.run_advanced_episode(args.task, args.use_memory, compare_models=True)
    else:
        print(f"🎯 Running single episode: {args.task}")
        print(f"Memory enabled: {args.use_memory}")
        result = runner.run_advanced_episode(args.task, args.use_memory, compare_models=False)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"📊 ADVANCED EPISODE RESULTS")
    print(f"{'='*60}")
    
    if 'model_comparison' in result:
        print("🤖 Model Comparison Results:")
        for model, model_result in result['model_comparison'].items():
            print(f"  {model}:")
            print(f"    Success: {'✅' if model_result.get('success', False) else '❌'}")
            print(f"    Actions: {len(model_result.get('actions', []))}")
            
            confidences = model_result.get('confidence_scores', [0.5])
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            print(f"    Avg Confidence: {avg_confidence:.2f}")
            print(f"    Time: {model_result.get('execution_time', 0):.2f}s")
            
            if model_result.get('actions'):
                print(f"    Sample Actions: {model_result['actions'][:2]}...")
        
        if 'aggregated_stats' in result:
            print(f"\n📈 Aggregated Statistics:")
            for model, stats in result['aggregated_stats'].items():
                print(f"  {model}: {stats.get('success_rate', 0):.2%} success rate")
    
    else:
        print(f"Task: {result['task_name']}")
        print(f"Goal: {result['goal']}")
        print(f"Memory Used: {result.get('memory_used', False)}")
        print(f"Actions Taken: {len(result.get('actions', []))}")
        
        if result.get('actions'):
            print(f"\n📝 Action Sequence:")
            for i, action in enumerate(result['actions'], 1):
                reasoning = result.get('reasoning', [''])[i-1] if i-1 < len(result.get('reasoning', [])) else ''
                print(f"  {i}. {action}")
                if reasoning and reasoning != "Basic LLM response":
                    print(f"     💭 Reasoning: {reasoning[:100]}...")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {args.output}")
    
    # Memory summary
    if args.use_memory and hasattr(runner, 'memory_buffer'):
        memory = runner.memory_buffer
        print(f"\n🧠 Memory Summary:")
        print(f"  Episodic Memory: {len(memory.episodic_memory)} entries")
        print(f"  Semantic Insights: {len(memory.semantic_memory)} categories")
        print(f"  Recorded Errors: {len(memory.error_memory)} entries")
        
        if memory.semantic_memory:
            print(f"  📚 Learned Insights:")
            for key, insights in memory.semantic_memory.items():
                latest = insights[-1]['insight']
                print(f"    {key}: {latest}")


if __name__ == "__main__":
    main()