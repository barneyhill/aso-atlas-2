import gzip
import json
import re
from typing import Dict, List, Tuple, Any

class HELMValidator:
    """Validates HELM notation strings"""
    
    @staticmethod
    def validate_helm(helm_string: str) -> Dict[str, Any]:
        """
        Validate a HELM notation string.
        
        Args:
            helm_string: The HELM string to validate
            
        Returns:
            Dictionary with validation results:
            {
                'is_valid': bool,
                'errors': list of error messages,
                'warnings': list of warning messages,
                'structure': parsed structure info
            }
        """
        errors = []
        warnings = []
        structure = {}
        
        if helm_string is None:
            return {
                'is_valid': False,
                'errors': ['No HELM annotation generated (None returned)'],
                'warnings': [],
                'structure': {}
            }

        # Check for proper termination
        if not helm_string.endswith('$$$$'):
            # If it doesn't look like HELM, treat it as a descriptive error message from the script
            return {
                'is_valid': False,
                'errors': [helm_string],  # Pass through the error message
                'warnings': [],
                'structure': {}
            }
        
        # Extract RNA polymer section
        # HELM uses single curly braces: RNA1{...}$$$$
        rna_match = re.match(r'(CHEM\d+\{.*?\}\|)?RNA1\{(.*?)\}(\|CHEM\d+\{.*?\})?\$\$\$\$', helm_string)
        
        if not rna_match:
            errors.append('Invalid HELM format: Could not parse RNA1{...}$$$$ structure')
            return {
                'is_valid': False,
                'errors': errors,
                'warnings': warnings,
                'structure': structure
            }
        
        has_5_conjugate = rna_match.group(1) is not None
        rna_sequence = rna_match.group(2)
        has_3_conjugate = rna_match.group(3) is not None
        
        structure['has_5_conjugate'] = has_5_conjugate
        structure['has_3_conjugate'] = has_3_conjugate
        
        # Parse nucleotides
        nucleotides = []
        
        # Split by periods, but be careful with modifications
        # Pattern: [modifications]base[backbone].
        parts = rna_sequence.split('.')
        
        for i, part in enumerate(parts):
            if not part.strip():
                continue
                
            # Parse each nucleotide unit
            # Pattern examples: [moe](T)[sp], d(A)[sp], [moe]([5meC])
            
            # Check for valid sugar modification
            sugar_mods = ['moe', 'fR', 'lna', 'cet', 'am']
            sugar_prefixes = ['d', 'r', 'm']  # d(), r(), m()
            
            has_valid_sugar = False
            for mod in sugar_mods:
                if f'[{mod}]' in part:
                    has_valid_sugar = True
                    break
            
            for prefix in sugar_prefixes:
                if f'{prefix}(' in part:
                    has_valid_sugar = True
                    break
            
            if not has_valid_sugar:
                warnings.append(f'Position {i+1}: No recognized sugar modification in "{part}"')
            
            # Check for valid base
            base_pattern = r'\(([A-Z5me\[\]]+)\)'
            base_match = re.search(base_pattern, part)
            
            if not base_match:
                errors.append(f'Position {i+1}: Could not find valid base in "{part}"')
            else:
                base = base_match.group(1)
                # Valid bases
                valid_bases = ['A', 'C', 'G', 'T', 'U', '[5meC]', '[m5C]']
                if base not in valid_bases and not any(vb in base for vb in valid_bases):
                    warnings.append(f'Position {i+1}: Unusual base "{base}"')
            
            # Check backbone (should not be on last position)
            has_backbone = '[sp]' in part or '[am]' in part
            is_last = (i == len(parts) - 1)
            
            if is_last and has_backbone:
                errors.append(f'Position {i+1}: Last nucleotide should not have backbone connector')
            elif not is_last and not has_backbone and '.' in rna_sequence[rna_sequence.find(part):]:
                # It's not the last, and no explicit backbone, check if it's phosphodiester (no notation)
                # This is acceptable
                pass
            
            nucleotides.append({
                'position': i + 1,
                'raw': part,
                'has_backbone': has_backbone
            })
        
        structure['nucleotide_count'] = len(nucleotides)
        structure['nucleotides'] = nucleotides
        
        # Check for common issues
        if len(nucleotides) == 0:
            errors.append('No nucleotides found in sequence')
        
        # Check for mismatched brackets
        if helm_string.count('[') != helm_string.count(']'):
            errors.append('Mismatched square brackets')
        
        if helm_string.count('{') != helm_string.count('}'):
            errors.append('Mismatched curly braces')
        
        if helm_string.count('(') != helm_string.count(')'):
            errors.append('Mismatched parentheses')
        
        # Final validation
        is_valid = len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'structure': structure
        }



def read_xml(xml_file: str) -> str:
    if xml_file.endswith('.gz'):
        with gzip.open(xml_file, 'rt', encoding='utf-8') as f:
            return f.read()
    else:
        with open(xml_file, 'r', encoding='utf-8') as f:
            return f.read()

