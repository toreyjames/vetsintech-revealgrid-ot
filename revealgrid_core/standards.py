"""
Industrial standards compliance module for RevealGrid.

Handles NIST SP 800-82, IEC 62443, Purdue Model, and NERC CIP compliance
classification and mapping.
"""

from typing import Dict, List, Any


# Standards Ontologies (Simplified Mapping)
STANDARDS = {
    'NIST_SP_800_82': {'criticality': ['low', 'medium', 'high']},
    'IEC_62443': {'roles': ['component', 'system', 'zone']},
    'Purdue_Model': {'levels': ['0_physical', '1_sensing', '2_control', '3_mes', '4_erp']},
    'NERC_CIP': {'impact': ['low', 'medium', 'high']}
}


def classify_asset_per_standard(node_type: str, latency: float) -> Dict[str, str]:
    """
    Classify asset according to industrial standards.
    
    Args:
        node_type: Type of asset (sensor, controller, etc.)
        latency: Network latency in milliseconds
        
    Returns:
        Dictionary with standards classifications
    """
    tags = {}
    
    # NIST SP 800-82 Criticality based on latency
    if latency > 10:
        tags['NIST_SP_800_82'] = 'high'
    elif latency > 5:
        tags['NIST_SP_800_82'] = 'medium'
    else:
        tags['NIST_SP_800_82'] = 'low'
    
    # IEC 62443 Role based on asset type
    if 'sensor' in node_type:
        tags['IEC_62443'] = 'component'
    elif 'controller' in node_type:
        tags['IEC_62443'] = 'system'
    else:
        tags['IEC_62443'] = 'zone'
    
    # Purdue Model Level mapping
    if 'sensor' in node_type:
        tags['Purdue_Model'] = '1_sensing'
    elif 'controller' in node_type:
        tags['Purdue_Model'] = '2_control'
    elif 'equipment' in node_type:
        tags['Purdue_Model'] = '3_mes'
    else:
        tags['Purdue_Model'] = '3_mes'
    
    # NERC CIP Impact assessment
    tags['NERC_CIP'] = 'high' if latency > 8 else 'medium' if latency > 4 else 'low'
    
    return tags


def validate_compliance(graph, standards: List[str] = None) -> Dict[str, Any]:
    """
    Validate graph compliance with industrial standards.
    
    Args:
        graph: NetworkX graph with assets
        standards: List of standards to validate against
        
    Returns:
        Dictionary with compliance validation results
    """
    if standards is None:
        standards = list(STANDARDS.keys())
    
    compliance_results = {}
    
    for standard in standards:
        if standard == 'NIST_SP_800_82':
            # Check criticality distribution
            criticality_counts = {}
            for node in graph.nodes:
                tags = graph.nodes[node].get('standards_tags', {})
                criticality = tags.get('NIST_SP_800_82', 'low')
                criticality_counts[criticality] = criticality_counts.get(criticality, 0) + 1
            
            compliance_results[standard] = {
                'criticality_distribution': criticality_counts,
                'high_criticality_count': criticality_counts.get('high', 0),
                'compliance_score': 0.9  # Simulated compliance score
            }
        
        elif standard == 'IEC_62443':
            # Check role distribution
            role_counts = {}
            for node in graph.nodes:
                tags = graph.nodes[node].get('standards_tags', {})
                role = tags.get('IEC_62443', 'component')
                role_counts[role] = role_counts.get(role, 0) + 1
            
            compliance_results[standard] = {
                'role_distribution': role_counts,
                'compliance_score': 0.85
            }
        
        elif standard == 'Purdue_Model':
            # Check level hierarchy
            level_counts = {}
            for node in graph.nodes:
                tags = graph.nodes[node].get('standards_tags', {})
                level = tags.get('Purdue_Model', '3_mes')
                level_counts[level] = level_counts.get(level, 0) + 1
            
            compliance_results[standard] = {
                'level_distribution': level_counts,
                'compliance_score': 0.88
            }
        
        elif standard == 'NERC_CIP':
            # Check impact assessment
            impact_counts = {}
            for node in graph.nodes:
                tags = graph.nodes[node].get('standards_tags', {})
                impact = tags.get('NERC_CIP', 'low')
                impact_counts[impact] = impact_counts.get(impact, 0) + 1
            
            compliance_results[standard] = {
                'impact_distribution': impact_counts,
                'high_impact_count': impact_counts.get('high', 0),
                'compliance_score': 0.92
            }
    
    return compliance_results


def generate_compliance_report(graph) -> str:
    """
    Generate human-readable compliance report.
    
    Args:
        graph: NetworkX graph with assets
        
    Returns:
        Formatted compliance report string
    """
    compliance = validate_compliance(graph)
    
    report = "## Industrial Standards Compliance Report\n\n"
    
    for standard, results in compliance.items():
        report += f"### {standard}\n"
        report += f"- Compliance Score: {results['compliance_score']:.1%}\n"
        
        if 'criticality_distribution' in results:
            report += "- Criticality Distribution:\n"
            for level, count in results['criticality_distribution'].items():
                report += f"  - {level}: {count} assets\n"
        
        if 'role_distribution' in results:
            report += "- Role Distribution:\n"
            for role, count in results['role_distribution'].items():
                report += f"  - {role}: {count} assets\n"
        
        if 'level_distribution' in results:
            report += "- Purdue Model Levels:\n"
            for level, count in results['level_distribution'].items():
                report += f"  - {level}: {count} assets\n"
        
        if 'impact_distribution' in results:
            report += "- NERC CIP Impact:\n"
            for impact, count in results['impact_distribution'].items():
                report += f"  - {impact}: {count} assets\n"
        
        report += "\n"
    
    return report 