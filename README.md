# AndroidWorld LLM Agent 🤖📱

A comprehensive framework for evaluating Large Language Model (LLM) agents in mobile automation tasks using the AndroidWorld benchmark environment.

##  Overview

This project implements and evaluates LLM-powered agents capable of navigating Android applications and completing complex mobile automation tasks. We achieved **100% success rates** on contact creation tasks through systematic prompt engineering, memory-enhanced architectures, and multi-model comparisons.

### Key Features

-  **Multi-LLM Support**: GPT-4, GPT-3.5-turbo, Claude-3.5-Sonnet
-  **Advanced Memory Systems**: Episodic, semantic, working, and error memory
-  **Comprehensive Evaluation**: Step accuracy, success rates, execution efficiency
-  **Multiple Prompt Strategies**: Basic, few-shot, self-reflection, memory-enhanced
-  **Detailed Analytics**: Failure analysis, model comparison, performance benchmarking

##  Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Results Summary](#results-summary)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [Contributing](#contributing)

##  Installation

### Prerequisites

- Python 3.8+
- AndroidWorld environment
- LLM API keys (OpenAI, Anthropic)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/androidworld-llm-agent.git
cd androidworld-llm-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt

# Core dependencies
pip install openai anthropic android-world
pip install fuzzywuzzy python-Levenshtein


```

3. **Set up API keys**
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

4. **Initialize AndroidWorld**
```bash
# Follow AndroidWorld setup instructions
# Ensure emulator/simulator access if needed
```

##  Quick Start

### Run a Basic Episode

```bash
# Task 1: Basic agent scaffold
python task1_agent_scaffold.py --task ContactsAddContact --llm openai

# List available tasks
python task1_agent_scaffold.py --list-tasks
```

### Evaluate Prompt Strategies

```bash
# Task 2: Compare prompting strategies
python task2_evaluation_strategy.py \
    --tasks ContactsAddContact \
    --prompt-types few_shot self_reflection \
    --episodes-per-task 5
```

### Run Comprehensive Benchmark

```bash
# Task 3: Full benchmark suite
python task3_fixed_benchmarking.py \
    --tasks ContactsAddContact \
    --models gpt-4 gpt-3.5-turbo \
    --episodes-per-task 10 \
    --generate-report \
    --generate-visualizations
```

### Advanced Features with Visualization

```bash
# Bonus: Memory-enhanced with visualization
python bonus_with_visualization.py \
    --task ContactsAddContact \
    --use-memory \
    --compare-models


```

##  Project Structure

```
androidworld-llm-agent/
├──  Core Implementation
│   ├── task1_agent_scaffold.py          # Basic agent framework
│   ├── task2_evaluation_strategy.py     # Prompt strategy evaluation
│   ├── task3_fixed_benchmarking.py      # Comprehensive benchmarking
│   ├── bonus_advanced_features.py       # Memory + multi-model comparison
│   └── bonus_with_visualization.py      # Rich CLI + Streamlit dashboard
│
├──  Templates & Configuration
│   ├── basic_template.txt               # Basic prompt template
│   ├── few_shot_examples.md             # Few-shot prompting examples
│   ├── self_reflection_template.md      # Self-reflection template
│   ├── memory_enhanced_template.md      # Memory-enhanced prompting
│   ├── prompt_guidelines.md             # Best practices guide
│   └── contact_examples.json            # Structured examples
│
├──  Results & Analysis
│   ├── task1_results_*.json             # Basic episode results
│   ├── task2_fixed_evaluation_*.json    # Prompt comparison results
│   ├── comprehensive_results_*.json     # Benchmark results
│   ├── bonus_results_*.json             # Advanced feature results
│   └── benchmark_report.md              # Generated analysis report
│

```

## 💡 Usage Examples

### Example 1: Single Episode with Memory

```python
from bonus_advanced_features import AdvancedEpisodeRunner

runner = AdvancedEpisodeRunner()
result = runner.run_advanced_episode(
    task_name="ContactsAddContact",
    use_memory=True,
    compare_models=False
)

print(f"Success: {result['success']}")
print(f"Actions: {result['actions']}")
```

### Example 2: Multi-Model Comparison

```python
from bonus_advanced_features import MultiModelComparator

comparator = MultiModelComparator()
# Models are auto-configured based on available API keys

results = comparator.run_comparison(
    task_name="ContactsAddContact",
    goal="Create contact for John Smith with +1234567890",
    observations=["Contact app open", "Form displayed"],
    ui_elements_list=[["Add button"], ["Name field", "Phone field"]]
)
```

### Example 3: Custom Prompt Strategy

```python
from task2_evaluation_strategy import AndroidWorldEvaluator

evaluator = AndroidWorldEvaluator()
result = evaluator.run_evaluation_episode(
    task_name="ContactsAddContact",
    llm_client=your_llm_client,
    prompt_type="few_shot"  # or "self_reflection"
)
```

## 📊 Results Summary

### Performance by Phase

| Phase | Approach | Step Accuracy | Success Rate | Avg Time | Innovation |
|-------|----------|---------------|--------------|----------|------------|
| 1 | Basic Scaffold | 92% | 100% | 3.8s | Task simulation framework |
| 2 | Enhanced Prompting | 87.5% | 100% | 7.2s | Prompt strategy comparison |
| 3 | Memory-Enhanced | 100% | 100% | 7.3s | Advanced memory systems |
| Bonus | Multi-Model + Viz | 100% | 100% | 8.2s | Production-ready features |

### Model Comparison

| Model | Success Rate | Avg Confidence | Execution Time | Reasoning Quality |
|-------|--------------|----------------|----------------|-------------------|
| GPT-4 | 100% | 1.0 | 10.8s | Excellent: Detailed, logical |
| GPT-3.5-turbo | 100% | 0.9 | 2.9s | Good: Concise, accurate |
| Claude-3.5-Sonnet | 100% | 0.7 | 10.1s | Verbose: Over-explanatory |

### Prompt Strategy Analysis

| Strategy | Step Accuracy | Success Rate | Avg Time | Key Strengths |
|----------|---------------|--------------|----------|---------------|
| Few-Shot | 100% | 100% | 3.9s | Fast, consistent patterns |
| Self-Reflection | 83% | 100% | 11.2s | Rich reasoning, learning potential |

## 🎛 Advanced Features

### Memory Architecture

- **Episodic Memory**: Recent actions and outcomes
- **Semantic Memory**: Task-specific insights and patterns  
- **Working Memory**: Current context and goal state
- **Error Memory**: Failed actions for learning

```python
# Enable memory in any script
runner.run_advanced_episode(
    task_name="ContactsAddContact", 
    use_memory=True
)
```



```bash
# Run with rich visualization
python bonus_with_visualization.py --task ContactsAddContact
```

### Streamlit Dashboard

```bash
# Launch interactive dashboard
python bonus_with_visualization.py --mode dashboard
# Open http://localhost:8501
```

### Function Calling & Structured Output

```python
# GPT-4 with function calling
client = AdvancedLLMClient(model="gpt-4")
structured_action = client.generate_structured_action(prompt)

# Returns AndroidAction with confidence scores
print(f"Action: {structured_action.action_type}")
print(f"Confidence: {structured_action.confidence}")
```

## ⚙️ Configuration

### Environment Variables

```bash
# Required API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional Configuration
export ANDROIDWORLD_DEBUG=true
export MAX_EPISODE_STEPS=15
```

### Customizing Prompts

Edit template files to customize prompting behavior:

- `basic_template.txt` - Simple goal-to-action prompting
- `few_shot_examples.md` - Add your own examples  
- `self_reflection_template.md` - Modify reasoning questions
- `memory_enhanced_template.md` - Customize memory integration

### Adding New Tasks

```python
# Extend ground truth generation
def generate_ground_truth_sequence(task_name: str, goal: str) -> List[str]:
    if "YourNewTask" in task_name:
        return [
            'CLICK("Start button")',
            'TYPE("Your input")',
            'CLICK("Finish button")'
        ]
```

## 🔧 CLI Reference

### Task 1: Basic Agent
```bash
python task1_agent_scaffold.py [OPTIONS]

Options:
  --task TEXT              Task name to run
  --list-tasks            List available tasks
  --max-steps INTEGER     Maximum steps per task (default: 15)
  --llm [openai|claude]   LLM provider (default: openai)
  --model TEXT            Specific model name
```

### Task 2: Evaluation
```bash
python task2_evaluation_strategy.py [OPTIONS]

Options:
  --tasks TEXT+                    Tasks to evaluate
  --prompt-types [few_shot|self_reflection]+  Prompt strategies
  --episodes-per-task INTEGER      Episodes per task (default: 3)
  --output TEXT                    Output file path
```

### Task 3: Benchmarking
```bash
python task3_fixed_benchmarking.py [OPTIONS]

Options:
  --tasks TEXT+                    Tasks to benchmark
  --models TEXT+                   Models to compare
  --episodes-per-task INTEGER      Episodes per task per model
  --generate-visualizations        Create performance plots
  --generate-report               Generate markdown report
  --output-dir TEXT               Output directory
```

### Bonus Features
```bash
python bonus_with_visualization.py [OPTIONS]

Options:
  --mode [episode|compare|dashboard]  Execution mode
  --task TEXT                         Task to run
  --use-memory                        Enable memory buffer
  --compare-models                    Multi-model comparison
  --no-visualization                  Disable rich output
```

## 📈 Understanding Results

### Metrics Explained

- **Step Accuracy**: Percentage of actions matching ground truth (≥80% fuzzy similarity)
- **Episode Success**: Task completion with proper goal achievement  
- **Execution Efficiency**: Time-to-completion and action count
- **Confidence Scores**: Model's self-assessed certainty (0.0-1.0)

### Output Files

- `*.json` - Raw results with full episode data
- `benchmark_report.md` - Human-readable analysis  
- `visualizations/*.png` - Performance charts
- `detailed_episodes` - Step-by-step breakdowns

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e .
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black .
flake8 .
mypy .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AndroidWorld Team** - For the excellent benchmark environment
- **OpenAI & Anthropic** - For powerful LLM APIs
- **Rich Library** - For beautiful CLI visualizations
- **Contributors** - For making this project better

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{androidworld-llm-agent,
  title={AndroidWorld LLM Agent: Comprehensive Evaluation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/androidworld-llm-agent}
}
```

## 🔗 Related Work

- [AndroidWorld Benchmark](https://github.com/android-world/android-world)
- [LLM Mobile Automation Survey](https://arxiv.org/abs/...)
- [Mobile Agent Evaluation Methods](https://arxiv.org/abs/...)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/androidworld-llm-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/androidworld-llm-agent/discussions)
- **Email**: your.email@domain.com

---

⭐ **Star this repo** if you find it useful! 

🐛 **Found a bug?** [Open an issue](https://github.com/yourusername/androidworld-llm-agent/issues/new)

💡 **Have an idea?** [Start a discussion](https://github.com/yourusername/androidworld-llm-agent/discussions/new)
