"""
TTA Configuration Definitions

This module contains all TTA configuration combinations for systematic evaluation.
Configurations are organized by categories and performance expectations.
"""

# =============================================================================
# ORIGINAL BASELINE CONFIGURATIONS
# =============================================================================

def get_original_configurations():
    """Get the original 8 baseline configurations."""
    return [
        # Configuration 1: Baseline (current values)
        {
            'name': 'Baseline',
            'tta_config': {},
            'quality_config': {},
            'validation_config': {}
        },
        
        # Configuration 2: Strict Consensus (lower thresholds)
        {
            'name': 'Strict_Consensus',
            'tta_config': {
                'min_score_threshold': 0.8,
                'consensuation_distance_threshold': 15.0
            },
            'quality_config': {
                'high_quality_threshold': 0.9,
                'consistency_distance_base': 10.0
            },
            'validation_config': {
                'center_deviation_threshold': 5.0,
                'angle_deviation_threshold': 10.0,
                'area_deviation_threshold': 0.2
            }
        },
        
        # Configuration 3: Lenient Consensus (higher thresholds) - TOP PERFORMER
        {
            'name': 'Lenient_Consensus',
            'tta_config': {
                'min_score_threshold': 0.6,
                'consensuation_distance_threshold': 50.0
            },
            'quality_config': {
                'high_quality_threshold': 0.7,
                'consistency_distance_base': 30.0
            },
            'validation_config': {
                'center_deviation_threshold': 15.0,
                'angle_deviation_threshold': 25.0,
                'area_deviation_threshold': 0.6
            }
        },
        
        # Configuration 4: No Quality Filtering
        {
            'name': 'No_Quality_Filter',
            'tta_config': {
                'min_score_threshold': 0.5
            },
            'quality_config': {
                'high_quality_threshold': 0.0,  # Very low to include all
                'min_inclusion_quality': 0.0
            },
            'validation_config': {
                'center_deviation_threshold': 20.0,
                'angle_deviation_threshold': 30.0,
                'area_deviation_threshold': 0.5
            }
        },
        
        # Configuration 5: Conservative (high quality only) - WORST PERFORMER
        {
            'name': 'Conservative_High_Quality',
            'tta_config': {
                'min_score_threshold': 0.85
            },
            'quality_config': {
                'high_quality_threshold': 0.95,
                'min_inclusion_quality': 0.8,
                'fallback_to_single_best': True
            },
            'validation_config': {
                'center_deviation_threshold': 3.0,
                'angle_deviation_threshold': 5.0,
                'area_deviation_threshold': 0.15
            }
        },
        
        # Configuration 6: Disabled Validation (very high thresholds)
        {
            'name': 'No_Validation',
            'tta_config': {
                'min_score_threshold': 0.7
            },
            'quality_config': {
                'high_quality_threshold': 0.8
            },
            'validation_config': {
                'center_deviation_threshold': 1000.0,  # Effectively disabled
                'angle_deviation_threshold': 180.0,    # Effectively disabled
                'area_deviation_threshold': 10.0       # Effectively disabled
            }
        },
        
        # Configuration 7: Balanced Precision-Recall
        {
            'name': 'Balanced_PR',
            'tta_config': {
                'min_score_threshold': 0.75,
                'consensuation_distance_threshold': 25.0
            },
            'quality_config': {
                'high_quality_threshold': 0.85,
                'consistency_distance_base': 18.0,
                'quality_exponent': 1.5
            },
            'validation_config': {
                'center_deviation_threshold': 8.0,
                'angle_deviation_threshold': 12.0,
                'area_deviation_threshold': 0.3,
                'center_weight': 0.5,
                'angle_weight': 0.4,
                'area_weight': 0.1
            }
        },
        
        # Configuration 8: Area Focused (emphasize size consistency) - TOP PERFORMER
        {
            'name': 'Area_Focused',
            'tta_config': {
                'min_score_threshold': 0.7
            },
            'quality_config': {
                'high_quality_threshold': 0.8
            },
            'validation_config': {
                'center_deviation_threshold': 12.0,
                'angle_deviation_threshold': 20.0,
                'area_deviation_threshold': 0.15,  # Stricter area
                'center_weight': 0.3,
                'angle_weight': 0.3,
                'area_weight': 0.4  # Higher area weight
            }
        }
    ]

# =============================================================================
# LENIENT CONSENSUS VARIATIONS (Based on top performer)
# =============================================================================

def get_lenient_consensus_variations():
    """Get variations of the Lenient_Consensus configuration."""
    return [
        # Lenient v1: Even more permissive thresholds
        {
            'name': 'Lenient_Ultra',
            'tta_config': {
                'min_score_threshold': 0.5,  # Lower score threshold
                'consensuation_distance_threshold': 60.0  # Higher distance
            },
            'quality_config': {
                'high_quality_threshold': 0.65,  # Lower quality threshold
                'consistency_distance_base': 35.0,
                'min_inclusion_quality': 0.2  # Include more transforms
            },
            'validation_config': {
                'center_deviation_threshold': 18.0,
                'angle_deviation_threshold': 30.0,
                'area_deviation_threshold': 0.7,  # Very permissive area
                'center_weight': 0.4,  # Less emphasis on center
                'angle_weight': 0.4,
                'area_weight': 0.2
            }
        },
        
        # Lenient v2: Optimized weights
        {
            'name': 'Lenient_Balanced_Weights',
            'tta_config': {
                'min_score_threshold': 0.65,
                'consensuation_distance_threshold': 45.0
            },
            'quality_config': {
                'high_quality_threshold': 0.75,
                'consistency_distance_base': 28.0,
                'quality_exponent': 1.8  # More emphasis on quality
            },
            'validation_config': {
                'center_deviation_threshold': 12.0,
                'angle_deviation_threshold': 22.0,
                'area_deviation_threshold': 0.5,
                'center_weight': 0.33,  # Equal weights
                'angle_weight': 0.33,
                'area_weight': 0.34
            }
        },
        
        # Lenient v3: Focus on recall optimization
        {
            'name': 'Lenient_High_Recall',
            'tta_config': {
                'min_score_threshold': 0.55,
                'consensuation_distance_threshold': 55.0
            },
            'quality_config': {
                'high_quality_threshold': 0.7,
                'consistency_distance_base': 32.0,
                'fallback_to_single_best': False  # Always try consensus
            },
            'validation_config': {
                'center_deviation_threshold': 16.0,
                'angle_deviation_threshold': 28.0,
                'area_deviation_threshold': 0.65,
                'center_weight': 0.35,
                'angle_weight': 0.35,
                'area_weight': 0.3
            }
        },
        
        # Lenient v4: Medium permissive (between original and ultra)
        {
            'name': 'Lenient_Medium',
            'tta_config': {
                'min_score_threshold': 0.65,
                'consensuation_distance_threshold': 40.0
            },
            'quality_config': {
                'high_quality_threshold': 0.75,
                'consistency_distance_base': 25.0
            },
            'validation_config': {
                'center_deviation_threshold': 12.0,
                'angle_deviation_threshold': 20.0,
                'area_deviation_threshold': 0.45,
                'center_weight': 0.45,
                'angle_weight': 0.45,
                'area_weight': 0.1
            }
        }
    ]

# =============================================================================
# AREA FOCUSED VARIATIONS (Based on consistent performer)
# =============================================================================

def get_area_focused_variations():
    """Get variations of the Area_Focused configuration."""
    return [
        # Area v1: Even more area emphasis
        {
            'name': 'Area_Ultra_Focused',
            'tta_config': {
                'min_score_threshold': 0.7
            },
            'quality_config': {
                'high_quality_threshold': 0.8,
                'quality_exponent': 2.2  # Higher quality emphasis
            },
            'validation_config': {
                'center_deviation_threshold': 10.0,
                'angle_deviation_threshold': 18.0,
                'area_deviation_threshold': 0.1,  # Very strict area
                'center_weight': 0.2,  # Lower center weight
                'angle_weight': 0.2,   # Lower angle weight
                'area_weight': 0.6     # Dominant area weight
            }
        },
        
        # Area v2: Balanced area focus with higher thresholds
        {
            'name': 'Area_Lenient_Balanced',
            'tta_config': {
                'min_score_threshold': 0.65
            },
            'quality_config': {
                'high_quality_threshold': 0.75,
                'consistency_distance_base': 22.0
            },
            'validation_config': {
                'center_deviation_threshold': 15.0,  # More lenient center
                'angle_deviation_threshold': 25.0,   # More lenient angle
                'area_deviation_threshold': 0.2,     # Still strict area
                'center_weight': 0.35,
                'angle_weight': 0.25,
                'area_weight': 0.4
            }
        },
        
        # Area v3: Adaptive area thresholds
        {
            'name': 'Area_Adaptive',
            'tta_config': {
                'min_score_threshold': 0.7,
                'consensuation_distance_threshold': 35.0
            },
            'quality_config': {
                'high_quality_threshold': 0.8,
                'adaptive_threshold_multiplier': 1.5,  # More adaptive
                'consistency_distance_base': 20.0
            },
            'validation_config': {
                'center_deviation_threshold': 12.0,
                'angle_deviation_threshold': 15.0,
                'area_deviation_threshold': 0.18,
                'center_weight': 0.3,
                'angle_weight': 0.3,
                'area_weight': 0.4
            }
        },
        
        # Area v4: Size consistency with quality emphasis
        {
            'name': 'Area_Quality_Focused',
            'tta_config': {
                'min_score_threshold': 0.75
            },
            'quality_config': {
                'high_quality_threshold': 0.85,
                'min_inclusion_quality': 0.4,  # Include medium quality
                'quality_exponent': 2.5        # Strong quality emphasis
            },
            'validation_config': {
                'center_deviation_threshold': 8.0,
                'angle_deviation_threshold': 16.0,
                'area_deviation_threshold': 0.12,  # Very strict area
                'center_weight': 0.25,
                'angle_weight': 0.25,
                'area_weight': 0.5  # Dominant area weight
            }
        }
    ]

# =============================================================================
# HYBRID CONFIGURATIONS (Best of both worlds)
# =============================================================================

def get_hybrid_configurations():
    """Get hybrid configurations combining best aspects of top performers."""
    return [
        # Hybrid v1: Lenient consensus with area focus
        {
            'name': 'Hybrid_Lenient_Area',
            'tta_config': {
                'min_score_threshold': 0.65,
                'consensuation_distance_threshold': 45.0
            },
            'quality_config': {
                'high_quality_threshold': 0.75,
                'consistency_distance_base': 28.0,
                'quality_exponent': 2.0
            },
            'validation_config': {
                'center_deviation_threshold': 14.0,  # Lenient center
                'angle_deviation_threshold': 22.0,   # Lenient angle  
                'area_deviation_threshold': 0.18,    # Strict area (from Area_Focused)
                'center_weight': 0.3,
                'angle_weight': 0.3,
                'area_weight': 0.4  # Area emphasis
            }
        },
        
        # Hybrid v2: Best recall with size consistency
        {
            'name': 'Hybrid_Recall_Size',
            'tta_config': {
                'min_score_threshold': 0.6,   # Low for high recall
                'consensuation_distance_threshold': 50.0
            },
            'quality_config': {
                'high_quality_threshold': 0.7,
                'consistency_distance_base': 30.0,
                'fallback_to_single_best': False
            },
            'validation_config': {
                'center_deviation_threshold': 16.0,
                'angle_deviation_threshold': 25.0,
                'area_deviation_threshold': 0.15,  # Strict area
                'center_weight': 0.35,
                'angle_weight': 0.25,
                'area_weight': 0.4
            }
        },
        
        # Hybrid v3: Optimized balance
        {
            'name': 'Hybrid_Optimized',
            'tta_config': {
                'min_score_threshold': 0.68,
                'consensuation_distance_threshold': 42.0
            },
            'quality_config': {
                'high_quality_threshold': 0.78,
                'consistency_distance_base': 26.0,
                'quality_exponent': 1.9
            },
            'validation_config': {
                'center_deviation_threshold': 13.0,
                'angle_deviation_threshold': 19.0,
                'area_deviation_threshold': 0.22,
                'center_weight': 0.32,
                'angle_weight': 0.32,
                'area_weight': 0.36
            }
        }
    ]

# =============================================================================
# MAIN CONFIGURATION FUNCTIONS
# =============================================================================

def get_all_configurations():
    """Get all available configurations for evaluation."""
    all_configs = []
    
    # Add original baseline configurations
    all_configs.extend(get_original_configurations())
    
    # Add new variations based on top performers
    all_configs.extend(get_lenient_consensus_variations())
    all_configs.extend(get_area_focused_variations())
    all_configs.extend(get_hybrid_configurations())
    
    return all_configs


def get_top_performer_configurations():
    """Get only the top performing configurations and their variations."""
    configs = []
    
    # Original top performers
    originals = get_original_configurations()
    top_performers = [c for c in originals if c['name'] in ['Lenient_Consensus', 'Area_Focused', 'Baseline']]
    configs.extend(top_performers)
    
    # Their variations
    configs.extend(get_lenient_consensus_variations())
    configs.extend(get_area_focused_variations())
    configs.extend(get_hybrid_configurations())
    
    return configs


def get_baseline_configurations():
    """Get only the original 8 baseline configurations."""
    return get_original_configurations()


def get_experimental_configurations():
    """Get only the new experimental configurations."""
    configs = []
    configs.extend(get_lenient_consensus_variations())
    configs.extend(get_area_focused_variations())
    configs.extend(get_hybrid_configurations())
    return configs


# Configuration summary
CONFIGURATION_INFO = {
    'total_configs': len(get_all_configurations()),
    'baseline_configs': len(get_baseline_configurations()),
    'experimental_configs': len(get_experimental_configurations()),
    'top_performers': ['Lenient_Consensus', 'Area_Focused'],
    'worst_performers': ['Conservative_High_Quality'],
    'recommended_for_testing': get_top_performer_configurations()
}


if __name__ == "__main__":
    """Print configuration summary when run directly."""
    print("🔧 TTA Configuration Summary")
    print("=" * 50)
    print(f"📊 Total Configurations: {CONFIGURATION_INFO['total_configs']}")
    print(f"📋 Baseline Configurations: {CONFIGURATION_INFO['baseline_configs']}")
    print(f"🧪 Experimental Configurations: {CONFIGURATION_INFO['experimental_configs']}")
    print(f"🏆 Top Performers: {', '.join(CONFIGURATION_INFO['top_performers'])}")
    print(f"❌ Worst Performers: {', '.join(CONFIGURATION_INFO['worst_performers'])}")
    
    print(f"\n🔬 All Configuration Names:")
    for i, config in enumerate(get_all_configurations(), 1):
        category = "🏆" if any(perf in config['name'] for perf in ['Lenient', 'Area']) else "📋"
        print(f"{i:2d}. {category} {config['name']}")
