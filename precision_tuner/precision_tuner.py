import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
import requests
import subprocess
import gc
import logging
from dataclasses import dataclass

logger = logging.getLogger("PrecisionTuner")


@dataclass
class ModelConfig:
    """Model configuration based on actual benchmarks and constraints"""
    name: str
    memory_mb: int
    benchmark_specialty: str  # Based on real performance data
    system_prompt_focus: str  # What to emphasize in system prompt


class SimplifiedMemoryManager:
    """Simplified memory management without model deletion"""
    
    def __init__(self, max_memory_mb: int = 22000):
        self.max_memory_mb = max_memory_mb
        self.memory_history = []
        
    def get_current_memory_usage(self) -> int:
        """Get current GPU memory usage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=5
            )
            usage = int(result.stdout.strip())
            self.memory_history.append(usage)
            return usage
        except Exception as e:
            logger.warning(f"Cannot get memory usage: {e}")
            return 0
    
    def cleanup_if_needed(self):
        """cleanup without deleting models"""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        time.sleep(1)


class MemoryOptimizedOllamaClient:
    """UPDATED: 3 models configuration - all kept in memory simultaneously"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.current_loaded_model = None
        self.memory_manager = SimplifiedMemoryManager()
        
        # UPDATED: 3 available models with distinct personas
        self.models = [
            # Index 0 - Creative & Engaging
            ModelConfig(
                name="gemma3:1b", 
                memory_mb=815,  # Actual size
                benchmark_specialty="creative_storyteller", 
                system_prompt_focus="creative expression, engaging narratives, analogies, and accessible explanations"
            ),
            # Index 1 - Structured & Comprehensive
            ModelConfig(
                name="llama3.2:1b", 
                memory_mb=1300,  # Actual size
                benchmark_specialty="knowledge_synthesizer", 
                system_prompt_focus="comprehensive knowledge, balanced perspectives, systematic organization, and clear communication"
            ),
            # Index 2 - Precise & Technical
            ModelConfig(
                name="phi3:3.8b", 
                memory_mb=2200,  # Actual size
                benchmark_specialty="precision_specialist", 
                system_prompt_focus="precise instruction following, technical accuracy, constraint adherence, and structured output"
            )
        ]
        
        # Total memory usage: ~4.3GB (well within limits)
        total_memory = sum(model.memory_mb for model in self.models)
        logger.info(f"Total model memory: {total_memory}MB ({total_memory/1024:.1f}GB)")
        logger.info(f"3-model configuration optimized for available resources")
        
        self.available_models = self._check_available_models()
        logger.info(f"Initialized with {len(self.available_models)} models for round-robin assignment")
        
    def _check_available_models(self) -> List[ModelConfig]:
        """Check which models are actually available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_names = [model['name'] for model in response.json().get('models', [])]
                
                logger.info(f"Available models in Ollama:")
                for name in available_names:
                    logger.info(f"  ✓ {name}")
                
                # Filter to only available models
                available_models = []
                for model in self.models:
                    if any(model.name.startswith(name.split(':')[0]) or name.startswith(model.name.split(':')[0])
                           for name in available_names):
                        available_models.append(model)
                        logger.info(f"  → {model.name} ({model.benchmark_specialty})")
                
                if len(available_models) != 3:
                    logger.warning(f"Expected 3 models, found {len(available_models)}. Round-robin will be uneven.")
                
                return available_models
            
        except Exception as e:
            logger.error(f"Error checking models: {e}")
        
        return self.models  # Fallback to all models
    
    def get_model_for_sample(self, sample_index: int) -> ModelConfig:
        """UPDATED: Round-robin assignment for 3 models: sample_index % 3"""
        model_index = sample_index % len(self.available_models)
        return self.available_models[model_index]
    
    def _unload_current_model(self):
        """Unload without deleting"""
        if self.current_loaded_model:
            try:
                logger.info(f"Unloading {self.current_loaded_model.name} from memory")
                self.memory_manager.cleanup_if_needed()
                self.current_loaded_model = None
            except Exception as e:
                logger.error(f"Error unloading: {e}")
                self.current_loaded_model = None
    
    def _load_model(self, model: ModelConfig):
        """Load model without deletion"""
        if self.current_loaded_model and self.current_loaded_model.name == model.name:
            return  # Already loaded
        
        # Unload current if different
        if self.current_loaded_model:
            self._unload_current_model()
        
        logger.info(f"Loading {model.name} for {model.benchmark_specialty}")
        
        try:
            # Load by making a small request
            response = requests.post(f"{self.base_url}/api/generate", 
                json={
                    "model": model.name, 
                    "prompt": "Ready", 
                    "stream": False,
                    "options": {"num_predict": 1}
                }, 
                timeout=60
            )
            
            if response.status_code == 200:
                self.current_loaded_model = model
                logger.info(f"✓ {model.name} loaded successfully")
            else:
                logger.error(f"Failed to load {model.name}")
        
        except Exception as e:
            logger.error(f"Error loading {model.name}: {e}")
    
    def generate_response(self, model: ModelConfig, instruction: str, constraint_context: str) -> str:
        """Generate response with constraint-aware system prompt"""
        
        # Load model if needed
        if not self.current_loaded_model or self.current_loaded_model.name != model.name:
            self._load_model(model)
        
        if not self.current_loaded_model:
            return "Error: Failed to load model"
        
        # Create system prompt based on model specialty and constraint context
        system_prompt = self._create_constraint_system_prompt(model, constraint_context)
        
        payload = {
            "model": model.name,
            "prompt": instruction,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Consistent temperature
                "num_predict": 750,  # Increased for complete responses
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/generate", 
                json=payload, timeout=180)
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _create_constraint_system_prompt(self, model: ModelConfig, constraint_context: str) -> str:
        """Create system prompt focused on constraint progression with persona specialization"""
        
        # Enhanced persona-specific base prompts
        persona_prompts = {
            "creative_storyteller": """You are a Creative Storyteller AI - master of engaging communication and narrative craft.
Your strengths: Making complex topics accessible through stories, analogies, and vivid examples. You excel at creative expression while maintaining accuracy.""",
            
            "knowledge_synthesizer": """You are a Knowledge Synthesizer AI - expert at comprehensive analysis and systematic organization.
Your strengths: Connecting diverse information, providing balanced perspectives, and delivering well-structured, thorough explanations.""",
            
            "precision_specialist": """You are a Precision Specialist AI - master of technical accuracy and exact constraint adherence.
Your strengths: Following instructions with absolute precision, technical detail, systematic execution, and structured output."""
        }
        
        base_prompt = persona_prompts.get(model.benchmark_specialty, 
            f"You are an AI assistant specializing in {model.benchmark_specialty}. Focus on {model.system_prompt_focus}.")
        
        # Constraint level guidance
        constraint_guidance = {
            "level_1": "Master single constraint precision. Focus on exact adherence to the given requirement.",
            "level_2": "Balance dual constraints effectively. Both requirements must be satisfied simultaneously.", 
            "level_3": "Coordinate triple constraints seamlessly. Show advanced skill integration.",
            "level_4": "Demonstrate expert-level constraint mastery. Perfect coordination of all requirements."
        }
        
        constraint_instruction = constraint_guidance.get(constraint_context, constraint_guidance["level_1"])
        
        return f"""{base_prompt}

CONSTRAINT PROGRESSION: {constraint_context.upper()}
INSTRUCTION: {constraint_instruction}

CRITICAL REQUIREMENTS:
• Follow ALL constraints with absolute precision
• Complete your response fully without truncation
• Demonstrate progressive constraint mastery through your specialty
• Apply your unique perspective while meeting all requirements

Your response will be evaluated for exact constraint adherence and learning progression."""
    
    def cleanup_and_shutdown(self):
        """Clean shutdown preserving models"""
        logger.info("Shutting down - preserving all models")
        if self.current_loaded_model:
            self._unload_current_model()
        self.memory_manager.cleanup_if_needed()


# UPDATED: Constraint level mapping for 3 models
CONSTRAINT_PROGRESSION_LEVELS = {
    1: "level_1",  # Single constraint mastery
    2: "level_2",  # Dual constraint coordination  
    3: "level_3",  # Triple constraint integration
    4: "level_4"   # Expert constraint mastery
}

def get_constraint_level_context(constraint_level: int) -> str:
    """Map constraint level to progression context"""
    return CONSTRAINT_PROGRESSION_LEVELS.get(constraint_level, "level_1")


# Helper function for constraint difficulty descriptions
def get_constraint_description(constraint_level: int) -> str:
    """Get human-readable description of constraint level"""
    descriptions = {
        1: "L1: Single constraint mastery (length OR format)",
        2: "L2: Dual constraint coordination (length + forbidden)", 
        3: "L3: Triple constraint integration (format + length + forbidden)",
        4: "L4: Expert constraint mastery (all types combined)"
    }
    return descriptions.get(constraint_level, f"Level {constraint_level}")


# Model assignment verification function
def verify_round_robin_assignment_3models(total_samples: int) -> Dict[str, int]:
    """Verify round-robin assignment works correctly for 3 models"""
    assignment_counts = {"gemma3:1b": 0, "llama3.2:1b": 0, "phi3:3.8b": 0}
    model_names = list(assignment_counts.keys())
    
    for i in range(total_samples):
        model_index = i % 3
        model_name = model_names[model_index]
        assignment_counts[model_name] += 1
    
    logger.info("3-Model Round-Robin Verification:")
    for model, count in assignment_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"  {model}: {count} samples ({percentage:.1f}%)")
    
    return assignment_counts
