import json
import re
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger("PrecisionTuner.ConstraintEvaluator")


class ConstraintEvaluator:
    """Evaluates constraint adherence with detailed scoring and accuracy"""
    
    @staticmethod
    def evaluate_format_constraint(text: str, expected_format: str) -> Tuple[float, str]:
        """Evaluate format compliance with detailed feedback and improved accuracy"""
        if not text or not text.strip():
            return 0.0, "Empty or whitespace-only response"
        
        text = text.strip()
        
        try:
            if expected_format.lower() == "json":
                return ConstraintEvaluator._evaluate_json_format(text)
            elif expected_format.lower() == "markdown":
                return ConstraintEvaluator._evaluate_markdown_format(text)
            elif expected_format.lower() == "list":
                return ConstraintEvaluator._evaluate_list_format(text)
            elif expected_format.lower() == "xml":
                return ConstraintEvaluator._evaluate_xml_format(text)
            elif expected_format.lower() == "yaml":
                return ConstraintEvaluator._evaluate_yaml_format(text)
            else:
                return 1.0, f"Format '{expected_format}' accepted (no specific validation)"
                
        except Exception as e:
            return 0.0, f"Format evaluation error: {str(e)[:50]}"
    
    @staticmethod
    def _evaluate_json_format(text: str) -> Tuple[float, str]:
        """JSON format evaluation"""
        # Remove markdown code blocks if present
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        elif text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
        
        try:
            parsed = json.loads(text)
            
            # Additional JSON quality checks
            if isinstance(parsed, (dict, list)):
                if isinstance(parsed, dict) and len(parsed) == 0:
                    return 0.7, "Valid JSON but empty object"
                elif isinstance(parsed, list) and len(parsed) == 0:
                    return 0.7, "Valid JSON but empty array"
                else:
                    return 1.0, "Perfect JSON format with content"
            elif isinstance(parsed, str):
                return 0.8, "Valid JSON but single string value"
            else:
                return 0.9, f"Valid JSON ({type(parsed).__name__} value)"
                
        except json.JSONDecodeError as e:
            # Try to identify common JSON issues for better feedback
            if "Expecting ',' delimiter" in str(e):
                return 0.2, "Invalid JSON: Missing comma delimiter"
            elif "Expecting ':' delimiter" in str(e):
                return 0.2, "Invalid JSON: Missing colon delimiter"
            elif "Unterminated string" in str(e):
                return 0.1, "Invalid JSON: Unterminated string"
            elif "Expecting value" in str(e):
                return 0.1, "Invalid JSON: Missing value"
            else:
                return 0.0, f"Invalid JSON: {str(e)[:50]}"
    
    @staticmethod
    def _evaluate_markdown_format(text: str) -> Tuple[float, str]:
        """Markdown format evaluation"""
        lines = text.split('\n')
        
        # Check for headers
        headers = [line for line in lines if line.strip().startswith('#')]
        
        # Check for other markdown elements
        has_bold = '**' in text or '__' in text
        has_italic = '*' in text or '_' in text
        has_links = '[' in text and '](' in text
        has_code_blocks = '```' in text
        has_inline_code = '`' in text and not has_code_blocks
        has_lists = any(line.strip().startswith(('-', '*', '+')) or 
                       re.match(r'^\s*\d+\.', line) for line in lines)
        
        markdown_features = sum([
            len(headers) > 0,
            has_bold,
            has_italic,
            has_links,
            has_code_blocks,
            has_inline_code,
            has_lists
        ])
        
        if len(headers) >= 2 and markdown_features >= 3:
            return 1.0, f"Excellent markdown with {len(headers)} headers and {markdown_features} features"
        elif len(headers) >= 1 and markdown_features >= 2:
            return 0.9, f"Good markdown with {len(headers)} headers and {markdown_features} features"
        elif len(headers) >= 1:
            return 0.8, f"Valid markdown with {len(headers)} headers"
        elif markdown_features >= 2:
            return 0.6, f"Partial markdown with {markdown_features} features but no headers"
        elif markdown_features >= 1:
            return 0.4, f"Minimal markdown with {markdown_features} features"
        else:
            return 0.2, "Plain text with no markdown formatting"
    
    @staticmethod
    def _evaluate_list_format(text: str) -> Tuple[float, str]:
        """list format evaluation"""
        lines = text.split('\n')
        
        # Identify different list types
        bulleted_items = [line for line in lines if line.strip().startswith(('-', '*', '+'))]
        numbered_items = [line for line in lines if re.match(r'^\s*\d+\.', line.strip())]
        
        total_items = len(bulleted_items) + len(numbered_items)
        
        if total_items >= 5:
            list_type = "numbered" if numbered_items else "bulleted"
            return 1.0, f"Excellent {list_type} list with {total_items} items"
        elif total_items >= 3:
            list_type = "numbered" if numbered_items else "bulleted"
            return 0.9, f"Good {list_type} list with {total_items} items"
        elif total_items >= 1:
            list_type = "numbered" if numbered_items else "bulleted"
            return 0.6, f"Minimal {list_type} list with {total_items} items"
        else:
            # Check for other list-like structures
            if any(':' in line for line in lines):
                return 0.4, "Text contains colon-separated items (informal list)"
            elif any('•' in line for line in lines):
                return 0.3, "Text contains bullet points"
            else:
                return 0.1, "No recognizable list format"
    
    @staticmethod
    def _evaluate_xml_format(text: str) -> Tuple[float, str]:
        """XML format evaluation"""
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(text)
            return 1.0, "Valid XML format"
        except ET.ParseError as e:
            return 0.0, f"Invalid XML: {str(e)[:50]}"
        except ImportError:
            # Fallback basic check
            if text.startswith('<') and text.endswith('>'):
                return 0.8, "XML-like format (basic validation)"
            else:
                return 0.0, "Not XML format"
    
    @staticmethod
    def _evaluate_yaml_format(text: str) -> Tuple[float, str]:
        """YAML format evaluation"""
        try:
            import yaml
            yaml.safe_load(text)
            return 1.0, "Valid YAML format"
        except yaml.YAMLError as e:
            return 0.0, f"Invalid YAML: {str(e)[:50]}"
        except ImportError:
            # Fallback basic check
            if ':' in text and not text.startswith('{'):
                return 0.6, "YAML-like format (basic validation)"
            else:
                return 0.0, "Not YAML format"
    
    @staticmethod 
    def evaluate_length_constraint(text: str, target_length: int, tolerance: int = 10) -> Tuple[float, str]:
        """length evaluation with better word counting"""
        if not text or not text.strip():
            return 0.0, "Empty response"
        
        # Improved word counting - handle punctuation and special characters better
        # Remove code blocks for more accurate counting
        cleaned_text = text
        
        # Remove markdown code blocks
        cleaned_text = re.sub(r'```[\s\S]*?```', ' ', cleaned_text)
        
        # Remove inline code
        cleaned_text = re.sub(r'`[^`]+`', ' ', cleaned_text)
        
        # Split by whitespace and filter out empty strings
        words = [word for word in cleaned_text.split() if word.strip()]
        word_count = len(words)
        
        difference = abs(word_count - target_length)
        
        if difference == 0:
            return 1.0, f"Exact length: {word_count} words (target: {target_length})"
        elif difference <= tolerance:
            return 1.0, f"Perfect length: {word_count} words (target: {target_length}, within tolerance)"
        else:
            # More nuanced scoring based on percentage error
            error_rate = difference / target_length
            
            if error_rate <= 0.1:  # Within 10%
                score = 0.9
            elif error_rate <= 0.2:  # Within 20%
                score = 0.7
            elif error_rate <= 0.3:  # Within 30%
                score = 0.5
            elif error_rate <= 0.5:  # Within 50%
                score = 0.3
            else:
                score = max(0.1, 1.0 - error_rate)
            
            direction = "over" if word_count > target_length else "under"
            return score, f"Length {direction} target: {word_count} words (target: {target_length}, off by {difference})"
    
    @staticmethod
    def evaluate_forbidden_words(text: str, forbidden_words: List[str]) -> Tuple[float, str]:
        """forbidden word evaluation with better matching"""
        if not text or not text.strip():
            return 0.0, "Empty response"
        
        if not forbidden_words:
            return 1.0, "No forbidden words specified"
        
        text_lower = text.lower()
        violations = []
        violation_counts = {}
        
        for word in forbidden_words:
            word_lower = word.lower()
            
            # Use word boundaries for more accurate matching
            pattern = r'\b' + re.escape(word_lower) + r'\b'
            matches = re.findall(pattern, text_lower)
            
            if matches:
                violations.append(word)
                violation_counts[word] = len(matches)
        
        if not violations:
            return 1.0, "No forbidden words used"
        else:
            # More sophisticated scoring based on severity
            total_violations = sum(violation_counts.values())
            unique_violations = len(violations)
            
            # Penalty increases with both number of unique words and total occurrences
            base_penalty = unique_violations * 0.2
            frequency_penalty = min(0.6, total_violations * 0.1)
            total_penalty = min(1.0, base_penalty + frequency_penalty)
            
            score = max(0.0, 1.0 - total_penalty)
            
            violation_details = []
            for word in violations:
                count = violation_counts[word]
                if count > 1:
                    violation_details.append(f"{word}({count}x)")
                else:
                    violation_details.append(word)
            
            return score, f"Forbidden words found: {', '.join(violation_details)}"
    
    @staticmethod
    def evaluate_required_elements(text: str, required_elements: List[str]) -> Tuple[float, str]:
        """required element evaluation with better matching"""
        if not text or not text.strip():
            return 0.0, "Empty response"
        
        if not required_elements:
            return 1.0, "No required elements specified"
        
        text_lower = text.lower()
        found_elements = []
        missing_elements = []
        element_positions = {}
        
        for element in required_elements:
            element_lower = element.lower()
            
            # Try exact phrase matching first
            if element_lower in text_lower:
                found_elements.append(element)
                # Find position for context
                pos = text_lower.find(element_lower)
                element_positions[element] = pos
            else:
                # Try word-by-word matching for phrases
                element_words = element_lower.split()
                if len(element_words) > 1:
                    # Check if all words from the element are present
                    all_words_present = all(
                        re.search(r'\b' + re.escape(word) + r'\b', text_lower) 
                        for word in element_words
                    )
                    if all_words_present:
                        found_elements.append(element)
                        element_positions[element] = -1  # Indicate partial match
                    else:
                        missing_elements.append(element)
                else:
                    # Single word - use word boundary matching
                    if re.search(r'\b' + re.escape(element_lower) + r'\b', text_lower):
                        found_elements.append(element)
                        match = re.search(r'\b' + re.escape(element_lower) + r'\b', text_lower)
                        element_positions[element] = match.start() if match else -1
                    else:
                        missing_elements.append(element)
        
        if not required_elements:
            return 1.0, "No required elements specified"
        
        score = len(found_elements) / len(required_elements)
        
        if score == 1.0:
            # Bonus points for natural integration (not all bunched together)
            positions = [pos for pos in element_positions.values() if pos >= 0]
            if len(positions) > 1:
                spread = max(positions) - min(positions)
                if spread > len(text) * 0.3:  # Elements spread across text
                    return 1.0, f"All required elements found naturally: {', '.join(found_elements)}"
            
            return score, f"All required elements found: {', '.join(found_elements)}"
        elif score >= 0.5:
            return score, f"Most required elements found: {', '.join(found_elements)}. Missing: {', '.join(missing_elements)}"
        else:
            return score, f"Few required elements found: {', '.join(found_elements) if found_elements else 'none'}. Missing: {', '.join(missing_elements)}"
    
    @staticmethod
    def evaluate_response_completeness(text: str) -> Tuple[float, str]:
        """Evaluate if response appears complete (not truncated)"""
        if not text or not text.strip():
            return 0.0, "Empty response"
        
        text = text.strip()
        
        # Check for truncation indicators
        truncation_indicators = [
            text.endswith("..."),
            text.endswith(" and"),
            text.endswith(" or"),
            text.endswith(" the"),
            text.endswith(" of"),
            text.endswith(" in"),
            text.endswith(" for"),
            text.endswith(" with"),
            text.endswith(" to"),
            text.endswith(" a"),
            text.endswith(" an"),
            text.endswith(","),
            len(text.strip()) < 30,  # Very short responses
        ]
        
        truncation_count = sum(truncation_indicators)
        
        # Check for proper sentence endings
        proper_endings = text.endswith(('.', '!', '?', '"', "'", '`', '}', ']', ')'))
        
        # Check for complete structure in formatted responses
        format_complete = True
        if text.strip().startswith('{') and not text.strip().endswith('}'):
            format_complete = False
        elif text.strip().startswith('[') and not text.strip().endswith(']'):
            format_complete = False
        elif '```' in text and text.count('```') % 2 != 0:
            format_complete = False
        
        if truncation_count == 0 and proper_endings and format_complete:
            return 1.0, "Response appears complete"
        elif truncation_count <= 1 and format_complete:
            return 0.8, "Response mostly complete with minor issues"
        elif truncation_count <= 2 or not format_complete:
            return 0.5, "Response possibly incomplete"
        else:
            return 0.2, "Response likely truncated or incomplete"
    
    @staticmethod
    def evaluate_comprehensive(text: str, constraints: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """Comprehensive evaluation of all constraints with detailed feedback"""
        if not text or not text.strip():
            return 0.0, {"error": "Empty or invalid response"}
        
        scores = []
        feedback = {}
        
        # Always check completeness
        completeness_score, completeness_msg = ConstraintEvaluator.evaluate_response_completeness(text)
        feedback["completeness"] = completeness_msg
        
        # Only include completeness in score if it's significantly poor
        if completeness_score < 0.5:
            scores.append(completeness_score)
        
        # Evaluate each constraint
        for constraint in constraints:
            constraint_type = constraint.get("type", "unknown")
            
            if constraint_type == "format":
                score, msg = ConstraintEvaluator.evaluate_format_constraint(
                    text, constraint.get("format", "")
                )
                feedback["format"] = msg
            elif constraint_type == "length":
                score, msg = ConstraintEvaluator.evaluate_length_constraint(
                    text, 
                    constraint.get("target", 100), 
                    constraint.get("tolerance", 10)
                )
                feedback["length"] = msg
            elif constraint_type == "forbidden":
                score, msg = ConstraintEvaluator.evaluate_forbidden_words(
                    text, constraint.get("words", [])
                )
                feedback["forbidden"] = msg
            elif constraint_type == "required":
                score, msg = ConstraintEvaluator.evaluate_required_elements(
                    text, constraint.get("elements", [])
                )
                feedback["required"] = msg
            else:
                score = 1.0
                msg = f"Unknown constraint type: {constraint_type}"
                feedback[constraint_type] = msg
            
            scores.append(score)
        
        # Calculate overall score
        if scores:
            overall_score = sum(scores) / len(scores)
        else:
            overall_score = 0.0
        
        # Add summary information
        feedback["overall_score"] = f"{overall_score:.3f}"
        feedback["constraint_count"] = len(constraints)
        feedback["completeness_score"] = f"{completeness_score:.3f}"
        
        return overall_score, feedback
