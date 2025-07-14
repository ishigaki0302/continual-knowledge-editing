#!/usr/bin/env python3
"""
Report Generation Module for IRT-based Knowledge Editing Evaluation

This module generates comprehensive reports for IRT analysis results.
Includes HTML, PDF, and LaTeX formats with publication-ready tables and figures.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import warnings

# HTML/PDF generation
try:
    from jinja2 import Template, Environment, FileSystemLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    warnings.warn("jinja2 not available. HTML reports may not work.")

try:
    import weasyprint
    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False
    warnings.warn("weasyprint not available. PDF generation may not work.")

# Statistics
try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available. Some statistical tests may not work.")

logger = logging.getLogger(__name__)


class IRTReporter:
    """
    Generates comprehensive reports for IRT analysis results.
    
    Features:
    - HTML reports with interactive elements
    - PDF reports for publication
    - LaTeX tables and figures
    - Statistical summaries and interpretations
    - Research recommendations
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize reporter
        
        Args:
            template_dir: Directory containing Jinja2 templates
        """
        self.template_dir = template_dir or Path(__file__).parent / 'templates'
        self.template_dir = Path(self.template_dir)
        
        # Create templates directory if it doesn't exist
        self.template_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        if HAS_JINJA2:
            self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))
        else:
            self.env = None
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default HTML templates"""
        
        # Main report template
        main_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IRT Analysis Report - Knowledge Editing Evaluation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .subsection { margin: 15px 0; }
        .table-container { overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .highlight { background-color: #fffacd; }
        .figure { text-align: center; margin: 20px 0; }
        .figure img { max-width: 100%; height: auto; }
        .interpretation { background-color: #f0f8ff; padding: 15px; border-radius: 5px; }
        .recommendation { background-color: #f0fff0; padding: 15px; border-radius: 5px; }
        .warning { background-color: #fff0f0; padding: 15px; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 5px; }
        .footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.9em; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>IRT Analysis Report</h1>
        <h2>Knowledge Editing Evaluation</h2>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        <p><strong>Model:</strong> {{ model_info.model_type }} | <strong>Persons:</strong> {{ model_info.n_persons }} | <strong>Items:</strong> {{ model_info.n_items }}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="subsection">
            <h3>Key Findings</h3>
            <ul>
                {% for finding in key_findings %}
                <li>{{ finding }}</li>
                {% endfor %}
            </ul>
        </div>
        
        <div class="subsection">
            <h3>Performance Metrics</h3>
            {% for metric in performance_metrics %}
            <div class="metric">
                <strong>{{ metric.name }}:</strong> {{ metric.value }}
            </div>
            {% endfor %}
        </div>
    </div>

    <div class="section">
        <h2>Model Information</h2>
        <div class="table-container">
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Model Type</td><td>{{ model_info.model_type }}</td></tr>
                <tr><td>Estimation Method</td><td>{{ model_info.estimation_method }}</td></tr>
                <tr><td>Number of Persons</td><td>{{ model_info.n_persons }}</td></tr>
                <tr><td>Number of Items</td><td>{{ model_info.n_items }}</td></tr>
                <tr><td>Number of Observations</td><td>{{ model_info.n_observations }}</td></tr>
                <tr><td>Converged</td><td>{{ model_info.converged }}</td></tr>
                <tr><td>Iterations</td><td>{{ model_info.n_iterations }}</td></tr>
                <tr><td>Log-Likelihood</td><td>{{ "%.2f"|format(model_info.log_likelihood) }}</td></tr>
                <tr><td>AIC</td><td>{{ "%.2f"|format(model_info.aic) }}</td></tr>
                <tr><td>BIC</td><td>{{ "%.2f"|format(model_info.bic) }}</td></tr>
            </table>
        </div>
    </div>

    {% if parameter_summary %}
    <div class="section">
        <h2>Parameter Summary</h2>
        {% for param_name, param_data in parameter_summary.items() %}
        <div class="subsection">
            <h3>{{ param_data.label }}</h3>
            <div class="table-container">
                <table>
                    <tr><th>Statistic</th><th>Value</th></tr>
                    <tr><td>Mean</td><td>{{ "%.3f"|format(param_data.mean) }}</td></tr>
                    <tr><td>Standard Deviation</td><td>{{ "%.3f"|format(param_data.std) }}</td></tr>
                    <tr><td>Minimum</td><td>{{ "%.3f"|format(param_data.min) }}</td></tr>
                    <tr><td>Maximum</td><td>{{ "%.3f"|format(param_data.max) }}</td></tr>
                    <tr><td>Median</td><td>{{ "%.3f"|format(param_data.median) }}</td></tr>
                </table>
            </div>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if method_comparison %}
    <div class="section">
        <h2>Method Comparison</h2>
        <div class="table-container">
            <table>
                <tr>
                    <th>Method</th>
                    <th>Mean Ability (θ)</th>
                    <th>Std. Dev.</th>
                    <th>N Persons</th>
                    <th>Rank</th>
                </tr>
                {% for method in method_comparison %}
                <tr {% if method.rank == 1 %}class="highlight"{% endif %}>
                    <td>{{ method.method }}</td>
                    <td>{{ "%.3f"|format(method.mean_theta) }}</td>
                    <td>{{ "%.3f"|format(method.std_theta) }}</td>
                    <td>{{ method.n_persons }}</td>
                    <td>{{ method.rank }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
    </div>
    {% endif %}

    {% if condition_analysis %}
    <div class="section">
        <h2>Condition Analysis</h2>
        <div class="table-container">
            <table>
                <tr>
                    <th>Condition</th>
                    <th>Mean Difficulty (β)</th>
                    <th>Std. Dev.</th>
                    <th>N Items</th>
                    <th>Interpretation</th>
                </tr>
                {% for condition in condition_analysis %}
                <tr>
                    <td>{{ condition.condition }}</td>
                    <td>{{ "%.3f"|format(condition.mean_beta) }}</td>
                    <td>{{ "%.3f"|format(condition.std_beta) }}</td>
                    <td>{{ condition.n_items }}</td>
                    <td>{{ condition.interpretation }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
    </div>
    {% endif %}

    {% if interpretations %}
    <div class="section">
        <h2>Interpretations</h2>
        {% for interpretation in interpretations %}
        <div class="interpretation">
            <h3>{{ interpretation.title }}</h3>
            <p>{{ interpretation.content }}</p>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if recommendations %}
    <div class="section">
        <h2>Recommendations</h2>
        {% for recommendation in recommendations %}
        <div class="recommendation">
            <h3>{{ recommendation.title }}</h3>
            <p>{{ recommendation.content }}</p>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if figures %}
    <div class="section">
        <h2>Figures</h2>
        {% for figure in figures %}
        <div class="figure">
            <h3>{{ figure.title }}</h3>
            <img src="{{ figure.path }}" alt="{{ figure.title }}">
            <p><em>{{ figure.caption }}</em></p>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <div class="footer">
        <p>Report generated by IRT Knowledge Editing Evaluation System</p>
        <p>Generated on {{ timestamp }}</p>
    </div>
</body>
</html>
        """
        
        # Save template
        template_path = self.template_dir / 'report.html'
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(main_template)
    
    def generate_report(self, 
                       results: Dict[str, Any],
                       irt_data: pd.DataFrame,
                       figures_dir: Optional[Path] = None,
                       output_path: Optional[str] = None,
                       format: str = 'html') -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            results: IRT model results
            irt_data: IRT formatted data
            figures_dir: Directory containing generated figures
            output_path: Output file path
            format: Output format ('html', 'pdf', 'latex')
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating {format} report")
        
        # Prepare report data
        report_data = self._prepare_report_data(results, irt_data, figures_dir)
        
        # Generate report based on format
        if format == 'html':
            return self._generate_html_report(report_data, output_path)
        elif format == 'pdf':
            return self._generate_pdf_report(report_data, output_path)
        elif format == 'latex':
            return self._generate_latex_report(report_data, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _prepare_report_data(self, 
                           results: Dict[str, Any],
                           irt_data: pd.DataFrame,
                           figures_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Prepare data for report generation"""
        logger.info("Preparing report data")
        
        # Basic model information
        model_info = {
            'model_type': results.get('model_type', 'Unknown'),
            'estimation_method': results.get('estimation_method', 'Unknown'),
            'n_persons': results.get('n_persons', 0),
            'n_items': results.get('n_items', 0),
            'n_observations': results.get('n_observations', 0),
            'converged': results.get('converged', False),
            'n_iterations': results.get('n_iterations', 0),
            'log_likelihood': results.get('log_likelihood', 0),
            'aic': results.get('fit_statistics', {}).get('aic', 0),
            'bic': results.get('fit_statistics', {}).get('bic', 0)
        }
        
        # Parameter summary
        parameter_summary = self._create_parameter_summary(results)
        
        # Method comparison
        method_comparison = self._create_method_comparison(results, irt_data)
        
        # Condition analysis
        condition_analysis = self._create_condition_analysis(results, irt_data)
        
        # Key findings
        key_findings = self._extract_key_findings(results, irt_data)
        
        # Performance metrics
        performance_metrics = self._create_performance_metrics(results, irt_data)
        
        # Interpretations
        interpretations = self._create_interpretations(results, irt_data)
        
        # Recommendations
        recommendations = self._create_recommendations(results, irt_data)
        
        # Figures
        figures = self._list_figures(figures_dir) if figures_dir else []
        
        report_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': model_info,
            'parameter_summary': parameter_summary,
            'method_comparison': method_comparison,
            'condition_analysis': condition_analysis,
            'key_findings': key_findings,
            'performance_metrics': performance_metrics,
            'interpretations': interpretations,
            'recommendations': recommendations,
            'figures': figures
        }
        
        return report_data
    
    def _create_parameter_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create parameter summary statistics"""
        parameter_summary = {}
        
        param_labels = {
            'theta': 'Person Ability (θ)',
            'beta': 'Item Difficulty (β)',
            'alpha': 'Item Discrimination (α)',
            'gamma': 'Guessing Parameter (γ)'
        }
        
        for param_name, param_values in results.items():
            if param_name in param_labels and param_values is not None:
                param_values = np.array(param_values)
                parameter_summary[param_name] = {
                    'label': param_labels[param_name],
                    'mean': float(np.mean(param_values)),
                    'std': float(np.std(param_values)),
                    'min': float(np.min(param_values)),
                    'max': float(np.max(param_values)),
                    'median': float(np.median(param_values))
                }
        
        return parameter_summary
    
    def _create_method_comparison(self, results: Dict[str, Any], 
                                irt_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create method comparison table"""
        method_comparison = []
        
        if 'persons' in results and 'theta' in results:
            # Create mapping from person_id to theta
            person_theta = dict(zip(results['persons'], results['theta']))
            
            # Group by method
            method_stats = {}
            for _, row in irt_data.iterrows():
                person_id = row['person_id']
                method = row['method']
                
                if person_id in person_theta:
                    theta = person_theta[person_id]
                    
                    if method not in method_stats:
                        method_stats[method] = []
                    method_stats[method].append(theta)
            
            # Calculate statistics for each method
            for method, theta_values in method_stats.items():
                theta_array = np.array(theta_values)
                method_comparison.append({
                    'method': method,
                    'mean_theta': float(np.mean(theta_array)),
                    'std_theta': float(np.std(theta_array)),
                    'n_persons': len(theta_array)
                })
            
            # Sort by mean theta (descending) and add ranks
            method_comparison.sort(key=lambda x: x['mean_theta'], reverse=True)
            for i, method in enumerate(method_comparison):
                method['rank'] = i + 1
        
        return method_comparison
    
    def _create_condition_analysis(self, results: Dict[str, Any], 
                                 irt_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create condition analysis table"""
        condition_analysis = []
        
        if 'items' in results and 'beta' in results:
            # Create mapping from item_id to beta
            item_beta = dict(zip(results['items'], results['beta']))
            
            # Group by condition
            condition_stats = {}
            for _, row in irt_data.iterrows():
                item_id = row['item_id']
                condition = row['condition']
                
                if item_id in item_beta:
                    beta = item_beta[item_id]
                    
                    if condition not in condition_stats:
                        condition_stats[condition] = []
                    condition_stats[condition].append(beta)
            
            # Calculate statistics for each condition
            for condition, beta_values in condition_stats.items():
                beta_array = np.array(beta_values)
                mean_beta = float(np.mean(beta_array))
                
                # Interpret difficulty
                if mean_beta > 0.5:
                    interpretation = "High difficulty"
                elif mean_beta > 0:
                    interpretation = "Medium difficulty"
                else:
                    interpretation = "Low difficulty"
                
                condition_analysis.append({
                    'condition': condition,
                    'mean_beta': mean_beta,
                    'std_beta': float(np.std(beta_array)),
                    'n_items': len(beta_array),
                    'interpretation': interpretation
                })
            
            # Sort by mean beta
            condition_analysis.sort(key=lambda x: x['mean_beta'])
        
        return condition_analysis
    
    def _extract_key_findings(self, results: Dict[str, Any], 
                            irt_data: pd.DataFrame) -> List[str]:
        """Extract key findings from the analysis"""
        findings = []
        
        # Model convergence
        if results.get('converged', False):
            findings.append("Model converged successfully")
        else:
            findings.append("⚠️ Model did not converge - results may be unreliable")
        
        # Best performing method
        method_stats = {}
        if 'persons' in results and 'theta' in results:
            person_theta = dict(zip(results['persons'], results['theta']))
            
            for _, row in irt_data.iterrows():
                person_id = row['person_id']
                method = row['method']
                
                if person_id in person_theta:
                    theta = person_theta[person_id]
                    
                    if method not in method_stats:
                        method_stats[method] = []
                    method_stats[method].append(theta)
            
            if method_stats:
                method_means = {method: np.mean(values) for method, values in method_stats.items()}
                best_method = max(method_means, key=method_means.get)
                worst_method = min(method_means, key=method_means.get)
                
                findings.append(f"Best performing method: {best_method} (θ = {method_means[best_method]:.3f})")
                findings.append(f"Worst performing method: {worst_method} (θ = {method_means[worst_method]:.3f})")
        
        # Most difficult condition
        condition_stats = {}
        if 'items' in results and 'beta' in results:
            item_beta = dict(zip(results['items'], results['beta']))
            
            for _, row in irt_data.iterrows():
                item_id = row['item_id']
                condition = row['condition']
                
                if item_id in item_beta:
                    beta = item_beta[item_id]
                    
                    if condition not in condition_stats:
                        condition_stats[condition] = []
                    condition_stats[condition].append(beta)
            
            if condition_stats:
                condition_means = {cond: np.mean(values) for cond, values in condition_stats.items()}
                hardest_condition = max(condition_means, key=condition_means.get)
                easiest_condition = min(condition_means, key=condition_means.get)
                
                findings.append(f"Most difficult condition: {hardest_condition} (β = {condition_means[hardest_condition]:.3f})")
                findings.append(f"Easiest condition: {easiest_condition} (β = {condition_means[easiest_condition]:.3f})")
        
        # Model fit quality
        if 'fit_statistics' in results:
            aic = results['fit_statistics'].get('aic', 0)
            bic = results['fit_statistics'].get('bic', 0)
            
            if aic < 0:
                findings.append("Excellent model fit (AIC < 0)")
            elif aic < 1000:
                findings.append("Good model fit")
            else:
                findings.append("Model fit may be poor - consider alternative models")
        
        return findings
    
    def _create_performance_metrics(self, results: Dict[str, Any], 
                                  irt_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create performance metrics"""
        metrics = []
        
        if 'theta' in results:
            theta_array = np.array(results['theta'])
            metrics.append({
                'name': 'Mean Person Ability',
                'value': f"{np.mean(theta_array):.3f}"
            })
            metrics.append({
                'name': 'Person Ability Range',
                'value': f"[{np.min(theta_array):.3f}, {np.max(theta_array):.3f}]"
            })
        
        if 'beta' in results:
            beta_array = np.array(results['beta'])
            metrics.append({
                'name': 'Mean Item Difficulty',
                'value': f"{np.mean(beta_array):.3f}"
            })
            metrics.append({
                'name': 'Item Difficulty Range',
                'value': f"[{np.min(beta_array):.3f}, {np.max(beta_array):.3f}]"
            })
        
        if 'fit_statistics' in results:
            metrics.append({
                'name': 'Model AIC',
                'value': f"{results['fit_statistics']['aic']:.2f}"
            })
            metrics.append({
                'name': 'Model BIC',
                'value': f"{results['fit_statistics']['bic']:.2f}"
            })
        
        # Calculate overall accuracy
        if 'is_correct' in irt_data.columns:
            accuracy = irt_data['is_correct'].mean()
            metrics.append({
                'name': 'Overall Accuracy',
                'value': f"{accuracy:.3f} ({accuracy*100:.1f}%)"
            })
        
        return metrics
    
    def _create_interpretations(self, results: Dict[str, Any], 
                              irt_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create interpretations of the results"""
        interpretations = []
        
        # Model type interpretation
        model_type = results.get('model_type', 'Unknown')
        if model_type == '1PL':
            interpretations.append({
                'title': 'Model Type: 1PL (Rasch)',
                'content': 'This model assumes all items have equal discrimination. '
                          'Differences in response patterns are attributed only to person ability and item difficulty.'
            })
        elif model_type == '2PL':
            interpretations.append({
                'title': 'Model Type: 2PL',
                'content': 'This model allows items to have different discrimination parameters. '
                          'Some items are better at distinguishing between high and low ability persons.'
            })
        elif model_type == '3PL':
            interpretations.append({
                'title': 'Model Type: 3PL',
                'content': 'This model includes a guessing parameter, accounting for the possibility '
                          'that low-ability persons may still get items correct by chance.'
            })
        
        # Person ability interpretation
        if 'theta' in results:
            theta_array = np.array(results['theta'])
            theta_mean = np.mean(theta_array)
            theta_std = np.std(theta_array)
            
            if theta_mean > 0.5:
                ability_level = "above average"
            elif theta_mean > -0.5:
                ability_level = "average"
            else:
                ability_level = "below average"
            
            interpretations.append({
                'title': 'Person Ability Distribution',
                'content': f'The average person ability is {theta_mean:.3f} (SD = {theta_std:.3f}), '
                          f'indicating {ability_level} performance overall. '
                          f'The range suggests {"high" if theta_std > 1 else "moderate"} variability in editing method effectiveness.'
            })
        
        # Item difficulty interpretation
        if 'beta' in results:
            beta_array = np.array(results['beta'])
            beta_mean = np.mean(beta_array)
            beta_std = np.std(beta_array)
            
            if beta_mean > 0.5:
                difficulty_level = "high"
            elif beta_mean > -0.5:
                difficulty_level = "moderate"
            else:
                difficulty_level = "low"
            
            interpretations.append({
                'title': 'Item Difficulty Distribution',
                'content': f'The average item difficulty is {beta_mean:.3f} (SD = {beta_std:.3f}), '
                          f'indicating {difficulty_level} difficulty overall. '
                          f'The variability suggests {"diverse" if beta_std > 1 else "consistent"} difficulty levels across experimental conditions.'
            })
        
        return interpretations
    
    def _create_recommendations(self, results: Dict[str, Any], 
                              irt_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create research recommendations"""
        recommendations = []
        
        # Model convergence recommendation
        if not results.get('converged', True):
            recommendations.append({
                'title': 'Model Convergence',
                'content': 'The model did not converge. Consider increasing the number of iterations, '
                          'checking for data quality issues, or trying a different estimation method.'
            })
        
        # Sample size recommendation
        n_persons = results.get('n_persons', 0)
        n_items = results.get('n_items', 0)
        
        if n_persons < 100:
            recommendations.append({
                'title': 'Sample Size - Persons',
                'content': f'With {n_persons} persons, the sample size is relatively small for IRT analysis. '
                          'Consider collecting data from more method-model combinations to improve parameter estimation stability.'
            })
        
        if n_items < 20:
            recommendations.append({
                'title': 'Sample Size - Items',
                'content': f'With {n_items} items, consider adding more experimental conditions or edit sequences '
                          'to improve the precision of difficulty estimates.'
            })
        
        # Method comparison recommendation
        method_stats = {}
        if 'persons' in results and 'theta' in results:
            person_theta = dict(zip(results['persons'], results['theta']))
            
            for _, row in irt_data.iterrows():
                person_id = row['person_id']
                method = row['method']
                
                if person_id in person_theta:
                    theta = person_theta[person_id]
                    
                    if method not in method_stats:
                        method_stats[method] = []
                    method_stats[method].append(theta)
            
            if len(method_stats) > 1:
                method_means = {method: np.mean(values) for method, values in method_stats.items()}
                best_method = max(method_means, key=method_means.get)
                
                recommendations.append({
                    'title': 'Method Selection',
                    'content': f'Based on the IRT analysis, {best_method} shows the highest person ability estimates. '
                              'Consider focusing further research on this method while investigating why other methods underperform.'
                })
        
        # Condition analysis recommendation
        condition_stats = {}
        if 'items' in results and 'beta' in results:
            item_beta = dict(zip(results['items'], results['beta']))
            
            for _, row in irt_data.iterrows():
                item_id = row['item_id']
                condition = row['condition']
                
                if item_id in item_beta:
                    beta = item_beta[item_id]
                    
                    if condition not in condition_stats:
                        condition_stats[condition] = []
                    condition_stats[condition].append(beta)
            
            if condition_stats:
                condition_means = {cond: np.mean(values) for cond, values in condition_stats.items()}
                hardest_condition = max(condition_means, key=condition_means.get)
                
                recommendations.append({
                    'title': 'Experimental Design',
                    'content': f'Condition {hardest_condition} shows the highest difficulty. '
                              'Consider investigating what makes this condition challenging and whether '
                              'it provides unique insights into knowledge editing limitations.'
                })
        
        # Model selection recommendation
        if 'fit_statistics' in results:
            aic = results['fit_statistics']['aic']
            model_type = results.get('model_type', 'Unknown')
            
            if model_type == '1PL':
                recommendations.append({
                    'title': 'Model Complexity',
                    'content': 'Consider comparing with 2PL and 3PL models to determine if allowing '
                              'variable discrimination or guessing parameters improves model fit.'
                })
            elif model_type == '2PL':
                recommendations.append({
                    'title': 'Model Validation',
                    'content': 'The 2PL model provides a good balance of complexity and interpretability. '
                              'Consider validating results with cross-validation or holdout samples.'
                })
        
        return recommendations
    
    def _list_figures(self, figures_dir: Path) -> List[Dict[str, Any]]:
        """List available figures"""
        figures = []
        
        if figures_dir and figures_dir.exists():
            figure_files = list(figures_dir.glob('*.png')) + list(figures_dir.glob('*.jpg')) + list(figures_dir.glob('*.svg'))
            
            figure_titles = {
                'icc_plots': 'Item Characteristic Curves',
                'parameter_distributions': 'Parameter Distributions',
                'person_item_map': 'Person-Item Map',
                'method_performance': 'Method Performance Comparison',
                'condition_analysis': 'Condition Analysis',
                'summary_dashboard': 'Summary Dashboard'
            }
            
            for fig_file in figure_files:
                # Extract figure type from filename
                fig_name = fig_file.stem
                for key, title in figure_titles.items():
                    if key in fig_name:
                        figures.append({
                            'title': title,
                            'path': str(fig_file),
                            'caption': f'{title} showing results from IRT analysis'
                        })
                        break
        
        return figures
    
    def _generate_html_report(self, report_data: Dict[str, Any], 
                            output_path: Optional[str] = None) -> str:
        """Generate HTML report"""
        if not HAS_JINJA2:
            raise ImportError("jinja2 is required for HTML report generation")
        
        template = self.env.get_template('report.html')
        html_content = template.render(**report_data)
        
        if output_path is None:
            output_path = f"irt_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {output_path}")
        return output_path
    
    def _generate_pdf_report(self, report_data: Dict[str, Any], 
                           output_path: Optional[str] = None) -> str:
        """Generate PDF report"""
        if not HAS_WEASYPRINT:
            logger.warning("weasyprint not available. Generating HTML report instead.")
            return self._generate_html_report(report_data, output_path)
        
        # First generate HTML
        html_path = self._generate_html_report(report_data)
        
        # Convert to PDF
        if output_path is None:
            output_path = f"irt_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        try:
            weasyprint.HTML(filename=html_path).write_pdf(output_path)
            logger.info(f"PDF report saved to: {output_path}")
            
            # Clean up HTML file
            Path(html_path).unlink()
            
            return output_path
        except Exception as e:
            logger.error(f"PDF generation failed: {str(e)}")
            logger.info(f"HTML report available at: {html_path}")
            return html_path
    
    def _generate_latex_report(self, report_data: Dict[str, Any], 
                             output_path: Optional[str] = None) -> str:
        """Generate LaTeX report"""
        if output_path is None:
            output_path = f"irt_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        
        # Generate LaTeX content
        latex_content = self._create_latex_content(report_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        logger.info(f"LaTeX report saved to: {output_path}")
        return output_path
    
    def _create_latex_content(self, report_data: Dict[str, Any]) -> str:
        """Create LaTeX content"""
        latex_content = f"""
\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{geometry}}
\\usepackage{{hyperref}}
\\geometry{{margin=1in}}

\\title{{IRT Analysis Report: Knowledge Editing Evaluation}}
\\author{{IRT Evaluation System}}
\\date{{{report_data['timestamp']}}}

\\begin{{document}}

\\maketitle

\\section{{Executive Summary}}

\\subsection{{Key Findings}}
\\begin{{itemize}}
"""
        
        for finding in report_data.get('key_findings', []):
            latex_content += f"\\item {finding}\n"
        
        latex_content += """
\\end{itemize}

\\subsection{Performance Metrics}
\\begin{table}[h]
\\centering
\\begin{tabular}{lr}
\\toprule
Metric & Value \\\\
\\midrule
"""
        
        for metric in report_data.get('performance_metrics', []):
            latex_content += f"{metric['name']} & {metric['value']} \\\\\n"
        
        latex_content += """
\\bottomrule
\\end{tabular}
\\caption{Performance Metrics}
\\end{table}

\\section{Model Information}
\\begin{table}[h]
\\centering
\\begin{tabular}{lr}
\\toprule
Parameter & Value \\\\
\\midrule
"""
        
        model_info = report_data.get('model_info', {})
        latex_content += f"Model Type & {model_info.get('model_type', 'Unknown')} \\\\\n"
        latex_content += f"Number of Persons & {model_info.get('n_persons', 0)} \\\\\n"
        latex_content += f"Number of Items & {model_info.get('n_items', 0)} \\\\\n"
        latex_content += f"Log-Likelihood & {model_info.get('log_likelihood', 0):.2f} \\\\\n"
        latex_content += f"AIC & {model_info.get('aic', 0):.2f} \\\\\n"
        latex_content += f"BIC & {model_info.get('bic', 0):.2f} \\\\\n"
        
        latex_content += """
\\bottomrule
\\end{tabular}
\\caption{Model Information}
\\end{table}
"""
        
        # Add method comparison if available
        if report_data.get('method_comparison'):
            latex_content += """
\\section{Method Comparison}
\\begin{table}[h]
\\centering
\\begin{tabular}{lrrr}
\\toprule
Method & Mean Ability ($\\theta$) & Std. Dev. & Rank \\\\
\\midrule
"""
            
            for method in report_data['method_comparison']:
                latex_content += f"{method['method']} & {method['mean_theta']:.3f} & {method['std_theta']:.3f} & {method['rank']} \\\\\n"
            
            latex_content += """
\\bottomrule
\\end{tabular}
\\caption{Method Comparison}
\\end{table}
"""
        
        latex_content += """
\\section{Interpretations}
"""
        
        for interpretation in report_data.get('interpretations', []):
            latex_content += f"\\subsection{{{interpretation['title']}}}\n"
            latex_content += f"{interpretation['content']}\n\n"
        
        latex_content += """
\\section{Recommendations}
"""
        
        for recommendation in report_data.get('recommendations', []):
            latex_content += f"\\subsection{{{recommendation['title']}}}\n"
            latex_content += f"{recommendation['content']}\n\n"
        
        latex_content += """
\\end{document}
"""
        
        return latex_content
    
    def create_publication_tables(self, results: Dict[str, Any], 
                                irt_data: pd.DataFrame,
                                output_dir: str = 'tables') -> Dict[str, str]:
        """
        Create publication-ready tables
        
        Args:
            results: IRT model results
            irt_data: IRT formatted data
            output_dir: Output directory for tables
            
        Returns:
            Dictionary mapping table names to file paths
        """
        logger.info("Creating publication tables")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        tables = {}
        
        # Table 1: Model comparison
        if 'model_type' in results:
            model_table = self._create_model_comparison_table(results, irt_data)
            table_path = output_dir / 'model_comparison.csv'
            model_table.to_csv(table_path, index=False)
            tables['model_comparison'] = str(table_path)
        
        # Table 2: Method performance
        method_table = self._create_method_performance_table(results, irt_data)
        if method_table is not None:
            table_path = output_dir / 'method_performance.csv'
            method_table.to_csv(table_path, index=False)
            tables['method_performance'] = str(table_path)
        
        # Table 3: Condition analysis
        condition_table = self._create_condition_analysis_table(results, irt_data)
        if condition_table is not None:
            table_path = output_dir / 'condition_analysis.csv'
            condition_table.to_csv(table_path, index=False)
            tables['condition_analysis'] = str(table_path)
        
        # Table 4: Parameter summary
        param_table = self._create_parameter_summary_table(results)
        if param_table is not None:
            table_path = output_dir / 'parameter_summary.csv'
            param_table.to_csv(table_path, index=False)
            tables['parameter_summary'] = str(table_path)
        
        logger.info(f"Created {len(tables)} publication tables in {output_dir}")
        return tables
    
    def _create_model_comparison_table(self, results: Dict[str, Any], 
                                     irt_data: pd.DataFrame) -> pd.DataFrame:
        """Create model comparison table"""
        model_data = {
            'Model': [results.get('model_type', 'Unknown')],
            'Log-Likelihood': [results.get('log_likelihood', 0)],
            'AIC': [results.get('fit_statistics', {}).get('aic', 0)],
            'BIC': [results.get('fit_statistics', {}).get('bic', 0)],
            'Parameters': [results.get('fit_statistics', {}).get('n_parameters', 0)],
            'Persons': [results.get('n_persons', 0)],
            'Items': [results.get('n_items', 0)],
            'Converged': [results.get('converged', False)]
        }
        
        return pd.DataFrame(model_data)
    
    def _create_method_performance_table(self, results: Dict[str, Any], 
                                       irt_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create method performance table"""
        if 'persons' not in results or 'theta' not in results:
            return None
        
        person_theta = dict(zip(results['persons'], results['theta']))
        
        method_data = []
        method_stats = {}
        
        for _, row in irt_data.iterrows():
            person_id = row['person_id']
            method = row['method']
            
            if person_id in person_theta:
                theta = person_theta[person_id]
                
                if method not in method_stats:
                    method_stats[method] = []
                method_stats[method].append(theta)
        
        for method, theta_values in method_stats.items():
            theta_array = np.array(theta_values)
            method_data.append({
                'Method': method,
                'Mean_Theta': np.mean(theta_array),
                'Std_Theta': np.std(theta_array),
                'Min_Theta': np.min(theta_array),
                'Max_Theta': np.max(theta_array),
                'N_Persons': len(theta_array)
            })
        
        df = pd.DataFrame(method_data)
        df = df.sort_values('Mean_Theta', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        
        return df
    
    def _create_condition_analysis_table(self, results: Dict[str, Any], 
                                       irt_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create condition analysis table"""
        if 'items' not in results or 'beta' not in results:
            return None
        
        item_beta = dict(zip(results['items'], results['beta']))
        
        condition_data = []
        condition_stats = {}
        
        for _, row in irt_data.iterrows():
            item_id = row['item_id']
            condition = row['condition']
            
            if item_id in item_beta:
                beta = item_beta[item_id]
                
                if condition not in condition_stats:
                    condition_stats[condition] = []
                condition_stats[condition].append(beta)
        
        for condition, beta_values in condition_stats.items():
            beta_array = np.array(beta_values)
            condition_data.append({
                'Condition': condition,
                'Mean_Beta': np.mean(beta_array),
                'Std_Beta': np.std(beta_array),
                'Min_Beta': np.min(beta_array),
                'Max_Beta': np.max(beta_array),
                'N_Items': len(beta_array)
            })
        
        df = pd.DataFrame(condition_data)
        df = df.sort_values('Mean_Beta')
        
        return df
    
    def _create_parameter_summary_table(self, results: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Create parameter summary table"""
        param_data = []
        
        param_labels = {
            'theta': 'Person Ability (θ)',
            'beta': 'Item Difficulty (β)',
            'alpha': 'Item Discrimination (α)',
            'gamma': 'Guessing Parameter (γ)'
        }
        
        for param_name, param_values in results.items():
            if param_name in param_labels and param_values is not None:
                param_values = np.array(param_values)
                param_data.append({
                    'Parameter': param_labels[param_name],
                    'Mean': np.mean(param_values),
                    'Std': np.std(param_values),
                    'Min': np.min(param_values),
                    'Max': np.max(param_values),
                    'Median': np.median(param_values),
                    'N': len(param_values)
                })
        
        if param_data:
            return pd.DataFrame(param_data)
        else:
            return None


def main():
    """Example usage of IRTReporter"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate IRT analysis reports')
    parser.add_argument('--results', type=str, required=True,
                       help='IRT results JSON file')
    parser.add_argument('--irt-data', type=str, required=True,
                       help='IRT data CSV file')
    parser.add_argument('--figures-dir', type=str,
                       help='Directory containing figures')
    parser.add_argument('--output', type=str,
                       help='Output file path')
    parser.add_argument('--format', type=str, default='html',
                       choices=['html', 'pdf', 'latex'],
                       help='Output format')
    parser.add_argument('--tables-dir', type=str, default='tables',
                       help='Directory for publication tables')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    irt_data = pd.read_csv(args.irt_data)
    
    # Initialize reporter
    reporter = IRTReporter()
    
    # Generate main report
    figures_dir = Path(args.figures_dir) if args.figures_dir else None
    report_path = reporter.generate_report(
        results, irt_data, figures_dir, args.output, args.format
    )
    
    # Create publication tables
    tables = reporter.create_publication_tables(results, irt_data, args.tables_dir)
    
    print(f"\nReport generated successfully!")
    print(f"Main report: {report_path}")
    print(f"Publication tables: {len(tables)} tables in {args.tables_dir}/")
    
    for table_name, table_path in tables.items():
        print(f"  - {table_name}: {table_path}")


if __name__ == "__main__":
    main()