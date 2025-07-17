#!/usr/bin/env python3
"""
Bonus: Advanced Features with Integrated Visualization

Enhanced version with:
- Memory buffer with full history
- Multi-model comparison (GPT-4 vs GPT-3.5-turbo vs Claude)  
- Real-time CLI visualization with Rich
- Optional Streamlit dashboard
- Episode progress monitoring
- Interactive visualizations
"""

import os
import sys
import json
import logging
import argparse
import time
import threading
from typing import Dict, List, Any, Optional, Type, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict

# Visualization imports
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    from rich.tree import Tree
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    print("Installing rich for better visualization...")
    os.system("pip install rich")
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()

# Optional Streamlit
try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

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
        self.episodic_memory = deque(maxlen=max_size)
        self.semantic_memory = {}
        self.working_memory = {}
        self.error_memory = []
    
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
                simple_response = self.generate_action(prompt)
                return self._parse_simple_action(simple_response)
                
        except Exception as e:
            logger.error(f"Structured action generation failed: {e}")
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


class ProgressVisualizer:
    """Real-time progress visualization with Rich."""
    
    def __init__(self):
        self.console = Console()
        self.current_progress = None
        self.step_data = []
    
    def start_episode_visualization(self, task_name: str, goal: str, max_steps: int = 5):
        """Start episode visualization."""
        
        self.step_data = []
        
        # Create header
        header_table = Table(show_header=False, box=box.ROUNDED)
        header_table.add_column("Field", style="bold blue")
        header_table.add_column("Value", style="green")
        header_table.add_row("🎯 Task", task_name)
        header_table.add_row("📋 Goal", goal[:60] + "..." if len(goal) > 60 else goal)
        header_table.add_row("⏳ Status", "Starting...")
        
        self.console.print(Panel(header_table, title="🤖 Episode Progress", border_style="blue"))
        
        # Create progress bar
        self.current_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )
        
        self.progress_task = self.current_progress.add_task("Processing...", total=max_steps)
        
        return self.current_progress
    
    def update_step(self, step_num: int, action: str, reasoning: str, confidence: float, execution_time: float):
        """Update progress with new step."""
        
        # Store step data
        self.step_data.append({
            "step": step_num,
            "action": action,
            "reasoning": reasoning[:50] + "..." if len(reasoning) > 50 else reasoning,
            "confidence": confidence,
            "execution_time": execution_time,
            "timestamp": time.time()
        })
        
        # Update progress
        if self.current_progress:
            self.current_progress.update(
                self.progress_task,
                description=f"Step {step_num}: {action[:30]}...",
                advance=1
            )
        
        # Show step details
        step_table = Table(show_header=False, box=box.SIMPLE)
        step_table.add_column("Field", style="bold yellow")
        step_table.add_column("Value", style="white")
        
        confidence_color = "green" if confidence > 0.8 else "yellow" if confidence > 0.5 else "red"
        
        step_table.add_row("🎬 Action", action)
        step_table.add_row("💭 Reasoning", reasoning[:80] + "..." if len(reasoning) > 80 else reasoning)
        step_table.add_row("📊 Confidence", Text(f"{confidence:.2f}", style=confidence_color))
        step_table.add_row("⏱️ Time", f"{execution_time:.2f}s")
        
        self.console.print(Panel(step_table, title=f"Step {step_num} Details", border_style="green"))
    
    def finish_episode(self, success: bool, total_time: float):
        """Finish episode visualization."""
        
        if self.current_progress:
            self.current_progress.stop()
        
        # Show final summary
        status_icon = "✅" if success else "❌"
        status_text = "SUCCESS" if success else "FAILED"
        status_color = "green" if success else "red"
        
        summary_table = Table(show_header=False, box=box.DOUBLE)
        summary_table.add_column("Metric", style="bold blue")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("🎯 Status", Text(f"{status_icon} {status_text}", style=status_color))
        summary_table.add_row("📊 Steps Taken", str(len(self.step_data)))
        summary_table.add_row("⏱️ Total Time", f"{total_time:.2f}s")
        
        if self.step_data:
            avg_confidence = sum(step["confidence"] for step in self.step_data) / len(self.step_data)
            summary_table.add_row("📈 Avg Confidence", f"{avg_confidence:.2f}")
        
        self.console.print(Panel(summary_table, title="🏁 Episode Complete", border_style=status_color))
        
        # Show action sequence tree
        if self.step_data:
            action_tree = Tree("🎬 Action Sequence")
            
            for step in self.step_data:
                confidence_color = "green" if step["confidence"] > 0.8 else "yellow" if step["confidence"] > 0.5 else "red"
                step_text = f"Step {step['step']}: {step['action']}"
                
                step_node = action_tree.add(Text(step_text, style="bold"))
                step_node.add(f"💭 {step['reasoning']}")
                step_node.add(Text(f"📊 {step['confidence']:.2f}", style=confidence_color))
                step_node.add(f"⏱️ {step['execution_time']:.2f}s")
            
            self.console.print(action_tree)


class MultiModelComparator:
    """Compare multiple LLM models with visualization."""
    
    def __init__(self, visualize: bool = True):
        self.models = {}
        self.comparison_results = defaultdict(list)
        self.visualize = visualize
        self.visualizer = ProgressVisualizer() if visualize else None
    
    def add_model(self, name: str, client):
        """Add a model to the comparison."""
        self.models[name] = client
        if self.visualize:
            console.print(f"[green]✅ Added model: {name}[/green]")
    
    def run_comparison(self, task_name: str, goal: str, observations: List[str], 
                      ui_elements_list: List[List[str]], max_steps: int = 5) -> Dict[str, Any]:
        """Run the same scenario across all models with visualization."""
        
        results = {}
        
        if self.visualize:
            console.print(Panel(f"🚀 Starting multi-model comparison\n🎯 Task: {task_name}\n📋 Models: {list(self.models.keys())}", 
                              title="Model Comparison", border_style="magenta"))
        
        for model_name, client in self.models.items():
            if self.visualize:
                console.print(f"\n[bold blue]🤖 Running {model_name}...[/bold blue]")
            
            # Start visualization for this model
            if self.visualize:
                progress = self.visualizer.start_episode_visualization(
                    f"{task_name} ({model_name})", goal, max_steps
                )
                progress.start()
            
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
                    step_start_time = time.time()
                    
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

Available Actions:
- CLICK("element_name") - Click on UI element
- TYPE("text") - Type text into field
- SCROLL("direction") - Scroll up/down/left/right
- WAIT() - Wait for screen

Action:"""
                    
                    # Get action based on client type
                    if hasattr(client, 'generate_structured_action'):
                        structured_action = client.generate_structured_action(prompt)
                        action_str = client.format_action_string(structured_action)
                        reasoning = structured_action.reasoning
                        confidence = structured_action.confidence
                    else:
                        action_str = client.generate_action(prompt)
                        reasoning = "Basic LLM response"
                        confidence = 0.7
                    
                    step_execution_time = time.time() - step_start_time
                    
                    model_result['actions'].append(action_str)
                    model_result['reasoning'].append(reasoning)
                    model_result['confidence_scores'].append(confidence)
                    
                    # Update visualization
                    if self.visualize:
                        self.visualizer.update_step(step + 1, action_str, reasoning, confidence, step_execution_time)
                        time.sleep(0.5)  # Brief pause for visualization
                    
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
                
                # Finish visualization for this model
                if self.visualize:
                    progress.stop()
                    self.visualizer.finish_episode(model_result['success'], model_result['execution_time'])
                
            except Exception as e:
                logger.error(f"Error running {model_name}: {e}")
                model_result['error'] = str(e)
                
                if self.visualize:
                    progress.stop()
                    console.print(f"[red]❌ Error in {model_name}: {e}[/red]")
            
            results[model_name] = model_result
        
        # Show comparison summary
        if self.visualize:
            self._show_comparison_summary(results)
        
        return results
    
    def _show_comparison_summary(self, results: Dict[str, Any]):
        """Show comparison summary with Rich."""
        
        summary_table = Table(title="🏆 Model Comparison Summary", box=box.ROUNDED)
        summary_table.add_column("Model", style="bold blue")
        summary_table.add_column("Success", justify="center")
        summary_table.add_column("Actions", justify="center")
        summary_table.add_column("Avg Confidence", justify="center")
        summary_table.add_column("Time", justify="center")
        summary_table.add_column("Performance", justify="center")
        
        for model_name, result in results.items():
            if 'error' in result:
                summary_table.add_row(
                    model_name, 
                    "❌ Error", 
                    "-", "-", "-", 
                    Text("Failed", style="red")
                )
                continue
            
            success_icon = "✅" if result.get('success', False) else "❌"
            actions_count = len(result.get('actions', []))
            
            confidences = result.get('confidence_scores', [0.5])
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            confidence_color = "green" if avg_confidence > 0.8 else "yellow" if avg_confidence > 0.5 else "red"
            
            exec_time = result.get('execution_time', 0)
            time_color = "green" if exec_time < 5 else "yellow" if exec_time < 10 else "red"
            
            # Performance rating
            if result.get('success', False) and avg_confidence > 0.8 and exec_time < 5:
                performance = Text("⭐ Excellent", style="green")
            elif result.get('success', False):
                performance = Text("👍 Good", style="yellow")
            else:
                performance = Text("👎 Poor", style="red")
            
            summary_table.add_row(
                model_name,
                success_icon,
                str(actions_count),
                Text(f"{avg_confidence:.2f}", style=confidence_color),
                Text(f"{exec_time:.1f}s", style=time_color),
                performance
            )
        
        console.print(summary_table)


class AdvancedEpisodeRunner:
    """Advanced episode runner with visualization."""
    
    def __init__(self, visualize: bool = True):
        self.task_registry = None
        self.aw_registry = None
        self.memory_buffer = MemoryBuffer()
        self.model_comparator = MultiModelComparator(visualize=visualize)
        self.visualizer = ProgressVisualizer() if visualize else None
        self.visualize = visualize
        self.initialize()
    
    def initialize(self):
        """Initialize AndroidWorld registry."""
        try:
            self.task_registry = registry.TaskRegistry()
            self.aw_registry = self.task_registry.get_registry(self.task_registry.ANDROID_WORLD_FAMILY)
            if self.visualize:
                console.print("[green]✅ AndroidWorld registry initialized[/green]")
            return True
        except Exception as e:
            if self.visualize:
                console.print(f"[red]❌ Failed to initialize registry: {e}[/red]")
            return False
    
    def setup_models(self):
        """Setup multiple models for comparison."""
        for model in ['gpt-4', 'gpt-3.5-turbo']:
            try:
                client = AdvancedLLMClient(model=model)
                self.model_comparator.add_model(model, client)
            except Exception as e:
                logger.warning(f"Could not setup {model}: {e}")
        
        try:
            claude_client = ClaudeClient()
            self.model_comparator.add_model("claude-3-sonnet", claude_client)
        except Exception as e:
            logger.warning(f"Could not setup Claude: {e}")
    
    def run_advanced_episode(self, task_name: str, use_memory: bool = True, 
                           compare_models: bool = False) -> Dict[str, Any]:
        """Run an advanced episode with visualization."""
        
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
        observations = [f"Android screen showing {task_name} interface (step {i+1})" for i in range(5)]
        
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
            if not self.model_comparator.models:
                self.setup_models()
            
            comparison_results = self.model_comparator.run_comparison(
                task_name, goal, observations, ui_elements_list
            )
            result['model_comparison'] = comparison_results
            result['aggregated_stats'] = self._get_aggregated_comparison(comparison_results)
        
        else:
            # Single model with visualization
            client = AdvancedLLMClient()
            if use_memory:
                self.memory_buffer.clear()
            
            if self.visualize:
                progress = self.visualizer.start_episode_visualization(task_name, goal)
                progress.start()
            
            actions = []
            reasoning_list = []
            start_time = time.time()
            
            try:
                for step, (obs, ui_elements) in enumerate(zip(observations, ui_elements_list)):
                    step_start_time = time.time()
                    
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
                    
                    step_execution_time = time.time() - step_start_time
                    
                    actions.append(action_str)
                    reasoning_list.append(structured_action.reasoning)
                    
                    # Update visualization
                    if self.visualize:
                        self.visualizer.update_step(
                            step + 1, action_str, structured_action.reasoning, 
                            structured_action.confidence, step_execution_time
                        )
                        time.sleep(0.5)
                    
                    if use_memory:
                        self.memory_buffer.add_step({
                            'step': step + 1,
                            'observation': obs,
                            'action': action_str,
                            'reasoning': structured_action.reasoning,
                            'goal': goal,
                            'ui_elements': ui_elements
                        })
                    
                    if "contact" in goal.lower() and "CLICK" in action_str and "Add" in action_str:
                        self.memory_buffer.add_semantic_insight("contact_creation", "Start with Add contact button")
                    
                    # Check for completion
                    if any(keyword in action_str.upper() for keyword in ['SAVE', 'SEND', 'CONFIRM']):
                        break
                
                total_time = time.time() - start_time
                success = len(actions) > 0 and any(keyword in actions[-1].upper() for keyword in ['SAVE', 'SEND', 'CONFIRM'])
                
                # Finish visualization
                if self.visualize:
                    progress.stop()
                    self.visualizer.finish_episode(success, total_time)
                
            except Exception as e:
                if self.visualize:
                    progress.stop()
                    console.print(f"[red]❌ Error during episode: {e}[/red]")
                return {"error": str(e)}
            
            result.update({
                'actions': actions,
                'reasoning': reasoning_list,
                'memory_used': use_memory,
                'final_memory_size': len(self.memory_buffer.episodic_memory) if use_memory else 0,
                'success': success,
                'total_time': total_time
            })
        
        return result
    
    def _get_aggregated_comparison(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get aggregated comparison statistics."""
        
        model_stats = {}
        
        for model_name, result in comparison_results.items():
            if 'error' in result:
                continue
                
            confidences = result.get('confidence_scores', [0.5])
            
            model_stats[model_name] = {
                'success_rate': 1.0 if result.get('success', False) else 0.0,
                'avg_confidence': sum(confidences) / len(confidences) if confidences else 0.5,
                'avg_execution_time': result.get('execution_time', 0),
                'total_tasks': 1,
                'avg_actions_per_task': len(result.get('actions', []))
            }
        
        return model_stats


def create_streamlit_dashboard():
    """Create Streamlit dashboard for episode monitoring."""
    
    if not STREAMLIT_AVAILABLE:
        console.print("[red]❌ Streamlit not available. Install with: pip install streamlit plotly[/red]")
        return
    
    st.set_page_config(
        page_title="AndroidWorld Episode Monitor",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AndroidWorld Episode Progress Monitor")
    st.markdown("Real-time visualization of LLM agent episodes")
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Episode Results",
        type=['json'],
        help="Upload bonus_results.json or similar"
    )
    
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            
            # Display episode information
            st.subheader("📋 Episode Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Task:** {data.get('task_name', 'Unknown')}")
                st.write(f"**Goal:** {data.get('goal', 'No goal specified')}")
            
            with col2:
                st.write(f"**Memory Used:** {'✅' if data.get('memory_used', False) else '❌'}")
                st.write(f"**Success:** {'✅' if data.get('success', False) else '❌'}")
            
            # Model comparison visualization
            if 'model_comparison' in data:
                st.subheader("🤖 Model Comparison")
                
                # Create comparison dataframe
                comparison_data = []
                for model_name, model_data in data['model_comparison'].items():
                    if 'error' not in model_data:
                        confidences = model_data.get('confidence_scores', [0.5])
                        comparison_data.append({
                            'Model': model_name,
                            'Success': '✅' if model_data.get('success', False) else '❌',
                            'Actions': len(model_data.get('actions', [])),
                            'Avg Confidence': sum(confidences) / len(confidences) if confidences else 0.5,
                            'Execution Time': model_data.get('execution_time', 0)
                        })
                
                if comparison_data:
                    import pandas as pd
                    df = pd.DataFrame(comparison_data)
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_conf = px.bar(
                            df, x='Model', y='Avg Confidence',
                            title='Average Confidence by Model',
                            color='Avg Confidence',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)
                    
                    with col2:
                        fig_time = px.bar(
                            df, x='Model', y='Execution Time',
                            title='Execution Time by Model',
                            color='Execution Time',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig_time, use_container_width=True)
                    
                    # Data table
                    st.subheader("📊 Detailed Comparison")
                    st.dataframe(df, use_container_width=True)
            
            # Single episode visualization
            elif 'actions' in data:
                st.subheader("🎬 Episode Timeline")
                
                actions = data.get('actions', [])
                reasoning = data.get('reasoning', [])
                
                if actions:
                    # Create timeline data
                    timeline_data = []
                    for i, (action, reason) in enumerate(zip(actions, reasoning), 1):
                        timeline_data.append({
                            'Step': i,
                            'Action': action,
                            'Reasoning': reason[:100] + '...' if len(reason) > 100 else reason
                        })
                    
                    # Display as expandable steps
                    for step_data in timeline_data:
                        with st.expander(f"Step {step_data['Step']}: {step_data['Action']}", expanded=False):
                            st.write(f"**Action:** {step_data['Action']}")
                            st.write(f"**Reasoning:** {step_data['Reasoning']}")
            
            # Memory information
            if data.get('memory_used', False):
                st.subheader("🧠 Memory Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Memory Entries", data.get('final_memory_size', 0))
                
                with col2:
                    st.metric("Total Time", f"{data.get('total_time', 0):.2f}s")
            
            # Raw data
            with st.expander("🔍 Raw Episode Data"):
                st.json(data)
                
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    else:
        st.info("👆 Upload an episode results file to visualize progress")
        
        # Show sample dashboard
        st.subheader("📊 Sample Dashboard")
        st.write("This is what the dashboard looks like with episode data:")
        
        # Sample metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Models", "3")
        with col2:
            st.metric("Success Rate", "100%")
        with col3:
            st.metric("Avg Time", "8.2s")
        with col4:
            st.metric("Avg Confidence", "0.87")


def main():
    """Main entry point for advanced features with visualization."""
    
    parser = argparse.ArgumentParser(description='AndroidWorld LLM Agent - Advanced Features with Visualization')
    parser.add_argument('--mode', choices=['episode', 'compare', 'dashboard'], default='episode',
                       help='Mode to run')
    parser.add_argument('--task', type=str, default='ContactsAddContact', help='Task to run')
    parser.add_argument('--use-memory', action='store_true', help='Use advanced memory buffer')
    parser.add_argument('--compare-models', action='store_true', help='Compare multiple models')
    parser.add_argument('--no-visualization', action='store_true', help='Disable rich visualization')
    parser.add_argument('--output', type=str, default='bonus_results.json', help='Output file')
    
    args = parser.parse_args()
    
    if args.mode == 'dashboard':
        if STREAMLIT_AVAILABLE:
            console.print("🚀 Starting Streamlit dashboard...")
            console.print("Access at: http://localhost:8501")
            create_streamlit_dashboard()
        else:
            console.print("[red]❌ Streamlit not available. Install with: pip install streamlit plotly[/red]")
        return
    
    # Initialize runner with/without visualization
    visualize = not args.no_visualization and RICH_AVAILABLE
    runner = AdvancedEpisodeRunner(visualize=visualize)
    
    if args.mode == 'compare' or args.compare_models:
        if visualize:
            console.print(f"[bold blue]🤖 Running model comparison on {args.task}[/bold blue]")
        runner.setup_models()
        result = runner.run_advanced_episode(args.task, args.use_memory, compare_models=True)
    else:
        if visualize:
            console.print(f"[bold green]🎯 Running single episode: {args.task}[/bold green]")
            console.print(f"[yellow]Memory enabled: {args.use_memory}[/yellow]")
        result = runner.run_advanced_episode(args.task, args.use_memory, compare_models=False)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    if visualize:
        console.print(f"\n[green]💾 Results saved to: {args.output}[/green]")
        
        # Show final summary without visualization details
        if 'model_comparison' in result:
            console.print("\n[bold magenta]📈 Final Statistics:[/bold magenta]")
            if 'aggregated_stats' in result:
                for model, stats in result['aggregated_stats'].items():
                    success_rate = stats.get('success_rate', 0)
                    avg_time = stats.get('avg_execution_time', 0)
                    console.print(f"  [blue]{model}[/blue]: {success_rate:.1%} success, {avg_time:.1f}s avg")
        else:
            success = result.get('success', False)
            total_time = result.get('total_time', 0)
            actions_count = len(result.get('actions', []))
            
            console.print("\n[bold magenta]📈 Episode Summary:[/bold magenta]")
            console.print(f"  Success: {'✅' if success else '❌'}")
            console.print(f"  Actions: {actions_count}")
            console.print(f"  Time: {total_time:.2f}s")
            
            if result.get('memory_used', False):
                memory_size = result.get('final_memory_size', 0)
                console.print(f"  Memory: {memory_size} entries")
    
    else:
        # Fallback non-visual output
        print(f"\n{'='*60}")
        print(f"📊 EPISODE RESULTS")
        print(f"{'='*60}")
        
        if 'model_comparison' in result:
            print("🤖 Model Comparison Results:")
            for model, model_result in result['model_comparison'].items():
                success = '✅' if model_result.get('success', False) else '❌'
                actions = len(model_result.get('actions', []))
                time_taken = model_result.get('execution_time', 0)
                print(f"  {model}: {success} {actions} actions, {time_taken:.1f}s")
        else:
            print(f"Task: {result['task_name']}")
            print(f"Success: {'✅' if result.get('success', False) else '❌'}")
            print(f"Actions: {len(result.get('actions', []))}")
        
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()