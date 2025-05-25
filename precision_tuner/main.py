"""
Corrected main.py - Constraint Progression with Round Robin Assignment
Aligns with the revised approach: constraint learning + simple assignment
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('precision_tuner_constraint_progression.log')
    ]
)

logger = logging.getLogger("PrecisionTuner.Main")

# CORRECTED imports for constraint progression approach
from precision_tuner import MemoryOptimizedOllamaClient
from constraint_evaluator import ConstraintEvaluator  # Keep your existing one
from dataset_generator import ConstraintProgressionDatasetGenerator  # ✅ Correct class name
from dataset_saver import DatasetSaver  # Keep your existing one


def main_precision_generation():
    """Main function with constraint progression + round-robin assignment"""
    
    logger.info("PRECISIONTUNER: CONSTRAINT PROGRESSION + ROUND-ROBIN")
    logger.info("=" * 60)
    logger.info("Strategy: Round-robin assignment (sample_index % 5)")
    logger.info("Learning: Constraint progression (L1→L2→L3→L4)")
    logger.info("L1: Single constraints (length OR format)")
    logger.info("L2: Dual constraints (length + forbidden)")
    logger.info("L3: Triple constraints (format + length + forbidden)")
    logger.info("L4: Expert constraints (all types combined)")
    logger.info("Distribution: Exactly 20% per model (guaranteed)")
    
    try:
        # Initialize constraint progression generator
        logger.info("Initializing Constraint Progression PrecisionTuner...")
        generator = ConstraintProgressionDatasetGenerator()  # ✅ Correct class
        
        # Show model assignment preview
        logger.info("Model Assignment Preview (first 10 samples):")
        for i in range(10):
            model = generator.ollama_client.get_model_for_sample(i)
            logger.info(f"  Sample {i}: {model.name} ({model.benchmark_specialty})")
        
        # Generate dataset with constraint progression
        logger.info("Starting constraint progression dataset generation...")
        dataset = generator.generate_precision_dataset(
            total_size=100,  # Your requested size
            constraint_distribution={  # ✅ Correct parameter name and levels
                1: 0.25,  # L1: Single constraints - 25 samples
                2: 0.30,  # L2: Dual constraints - 30 samples  
                3: 0.30,  # L3: Triple constraints - 30 samples
                4: 0.15   # L4: Expert constraints - 15 samples
            }
        )
        
        # Save dataset with constraint progression metadata
        logger.info("Saving dataset with constraint progression filtering...")
        hf_dataset = DatasetSaver.save_precision_dataset(dataset, "precision_instruct_constraint_progression_100")
        
        logger.info("CONSTRAINT PROGRESSION COMPLETE!")
        logger.info("=" * 40)
        logger.info("✅ Round-robin assignment completed")
        logger.info("✅ Constraint progression implemented (L1→L4)")
        logger.info("✅ Exactly 20% per model distribution")
        logger.info("✅ Filterable by constraint_level column")
        logger.info("✅ Benchmark-based specialties applied")
        
        # Verify round-robin distribution
        model_counts = {}
        constraint_counts = {}  # ✅ Changed from cognitive_counts
        
        for sample in dataset:
            model = sample['model_used']
            constraint_level = sample['constraint_level']  # ✅ Correct field name
            
            model_counts[model] = model_counts.get(model, 0) + 1
            constraint_counts[constraint_level] = constraint_counts.get(constraint_level, 0) + 1
        
        logger.info("Final Distribution Verification:")
        logger.info("Model Distribution:")
        total = len(dataset)
        for model, count in model_counts.items():
            percentage = (count / total) * 100
            logger.info(f"  {model}: {count}/{total} ({percentage:.1f}%)")
        
        logger.info("Constraint Level Distribution:")  # ✅ Changed from cognitive
        for constraint_level, count in constraint_counts.items():
            percentage = (count / total) * 100
            level_description = {
                1: "L1: Single constraints",
                2: "L2: Dual constraints", 
                3: "L3: Triple constraints",
                4: "L4: Expert constraints"
            }.get(constraint_level, f"L{constraint_level}")
            logger.info(f"  {level_description}: {count}/{total} ({percentage:.1f}%)")
        
        # Show sample with constraint progression metadata
        logger.info("Sample with Constraint Progression Metadata:")
        sample = dataset[0]
        logger.info(f"  Instruction: {sample['instruction']}")
        logger.info(f"  Constraint Level: {sample['constraint_level']}")  # ✅ Correct field
        logger.info(f"  Learning Focus: {sample['learning_focus']}")      # ✅ Correct field
        logger.info(f"  Experiential Progression: {sample['experiential_progression']}")  # ✅ New field
        logger.info(f"  Model Used: {sample['model_used']} ({sample['model_specialty']})")
        logger.info(f"  Sample Index: {sample['sample_index']}")
        logger.info(f"  Model Assignment Index: {sample['model_assignment_index']}")
        logger.info(f"  Quality Score: {sample['quality_score']:.3f}")
        logger.info(f"  Constraint Count: {sample['constraint_count']}")  # ✅ New field
        
        return dataset, hf_dataset
        
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        return None, None


def demonstrate_constraint_filtering(dataset):
    """Demonstrate filtering by constraint progression"""
    if not dataset:
        return
    
    logger.info("Constraint Progression Filtering Examples:")
    
    # Filter by constraint level
    single_constraint_samples = [s for s in dataset if s['constraint_level'] == 1]
    expert_constraint_samples = [s for s in dataset if s['constraint_level'] == 4]
    
    logger.info(f"L1 (Single constraint) samples: {len(single_constraint_samples)}")
    if single_constraint_samples:
        logger.info(f"  Example: {single_constraint_samples[0]['instruction']}")
        logger.info(f"  Learning focus: {single_constraint_samples[0]['learning_focus']}")
    
    logger.info(f"L4 (Expert constraint) samples: {len(expert_constraint_samples)}")
    if expert_constraint_samples:
        logger.info(f"  Example: {expert_constraint_samples[0]['instruction']}")
        logger.info(f"  Learning focus: {expert_constraint_samples[0]['learning_focus']}")
    
    # Filter by constraint type
    length_samples = [s for s in dataset if 'length' in s['constraint_type']]
    format_samples = [s for s in dataset if 'format' in s['constraint_type']]
    
    logger.info(f"Length constraint samples: {len(length_samples)}")
    logger.info(f"Format constraint samples: {len(format_samples)}")
    
    # Filter by learning focus
    mastery_samples = [s for s in dataset if 'mastery' in s['learning_focus']]
    coordination_samples = [s for s in dataset if 'coordination' in s['learning_focus']]
    
    logger.info(f"Mastery focus samples: {len(mastery_samples)}")
    logger.info(f"Coordination focus samples: {len(coordination_samples)}")


def verify_round_robin_assignment(dataset):
    """Verify that round-robin assignment worked correctly"""
    logger.info("Round-Robin Assignment Verification:")
    
    # Check assignment pattern
    assignment_pattern = []
    for sample in sorted(dataset, key=lambda x: x['sample_index']):
        model_index = sample['model_assignment_index']
        expected_index = sample['sample_index'] % 5
        assignment_pattern.append((sample['sample_index'], model_index, expected_index))
    
    # Check first 10 assignments
    logger.info("First 10 assignments (sample_index, assigned_model_index, expected_index):")
    for i, (sample_idx, assigned, expected) in enumerate(assignment_pattern[:10]):
        status = "✓" if assigned == expected else "✗"
        logger.info(f"  Sample {sample_idx}: assigned={assigned}, expected={expected} {status}")
    
    # Check if any assignments are wrong
    incorrect_assignments = [
        (sample_idx, assigned, expected) 
        for sample_idx, assigned, expected in assignment_pattern 
        if assigned != expected
    ]
    
    if incorrect_assignments:
        logger.warning(f"Found {len(incorrect_assignments)} incorrect assignments!")
        for sample_idx, assigned, expected in incorrect_assignments[:5]:
            logger.warning(f"  Sample {sample_idx}: got {assigned}, expected {expected}")
    else:
        logger.info("✅ Perfect round-robin assignment - all samples correctly assigned")


def analyze_constraint_progression(dataset):
    """Analyze constraint learning progression effectiveness"""
    logger.info("Constraint Learning Progression Analysis:")
    
    # Group by constraint level
    by_level = {}
    for sample in dataset:
        level = sample['constraint_level']
        if level not in by_level:
            by_level[level] = []
        by_level[level].append(sample)
    
    # Analyze progression
    for level in sorted(by_level.keys()):
        samples = by_level[level]
        avg_quality = sum(s['quality_score'] for s in samples) / len(samples)
        constraint_types = set(s['constraint_type'] for s in samples)
        
        logger.info(f"Level {level}: {len(samples)} samples, avg quality: {avg_quality:.3f}")
        logger.info(f"  Constraint types: {', '.join(constraint_types)}")
        
        # Show learning progression
        if samples:
            example = samples[0]
            logger.info(f"  Learning focus: {example['learning_focus']}")
            logger.info(f"  Constraint count: {example['constraint_count']}")


if __name__ == "__main__":
    # Run the constraint progression dataset generation
    dataset, hf_dataset = main_precision_generation()
    
    if dataset:
        logger.info(f"SUCCESS: {len(dataset)} samples generated with constraint progression!")
        logger.info("Check 'precision_instruct_constraint_progression_100' directory for all files")
        
        # Verify round-robin worked correctly
        verify_round_robin_assignment(dataset)
        
        # Demonstrate constraint filtering
        demonstrate_constraint_filtering(dataset)
        
        # Analyze constraint progression
        analyze_constraint_progression(dataset)
        
        # Show file structure
        logger.info("Generated Files:")
        output_dir = Path("precision_instruct_constraint_progression_100")
        if output_dir.exists():
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    logger.info(f"  ✓ {file_path}")
        
        logger.info("\nConstraint Progression Filtering Examples:")
        logger.info("# Filter by constraint level")
        logger.info("single_constraints = [s for s in dataset if s['constraint_level'] == 1]")
        logger.info("expert_constraints = [s for s in dataset if s['constraint_level'] == 4]")
        logger.info("")
        logger.info("# Filter by learning focus")  
        logger.info("mastery_samples = [s for s in dataset if 'mastery' in s['learning_focus']]")
        logger.info("coordination_samples = [s for s in dataset if 'coordination' in s['learning_focus']]")
        logger.info("")
        logger.info("# Filter by constraint complexity")
        logger.info("simple_constraints = [s for s in dataset if s['constraint_count'] == 1]")
        logger.info("complex_constraints = [s for s in dataset if s['constraint_count'] >= 3]")
        
    else:
        logger.error("Generation failed - check setup and try again")