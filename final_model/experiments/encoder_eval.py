"""
Encoder Ablation Evaluation Script

This script performs Representational Similarity Analysis (RSA) and cluster validation
for encoder representations from ablation experiments.

Teile des Codes wurden mit ChatGPT Modell o4-mini (chatgpt.com) und Claude Sonnet 4 (claude.ai) generiert.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class Config:
    """Configuration class for evaluation parameters."""

    def __init__(self, config_path: Optional[str] = None):
        self.random_seed = 42
        self.bootstrap_samples = 1000
        self.confidence_level = 0.95
        self.max_samples_per_variant = 10000  # Memory management
        self.normalize_representations = True
        self.correlation_methods = ['pearson', 'spearman']
        self.output_formats = ['csv', 'json']

        if config_path and Path(config_path).exists():
            self.load_config(config_path)

    def load_config(self, config_path: str):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class EncoderAblationEvaluator:
    """Main evaluator class for encoder ablation experiments."""

    def __init__(self, config: Config):
        self.config = config
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)

        self.representations = {}
        self.variant_names = []
        self.expose_ids = []

    def load_all_variants(self, base_dir: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load all encoder representations from directory structure.

        Args:
            base_dir: Base directory containing variant subdirectories

        Returns:
            Dictionary with structure: {variant_name: {expose_id: representation_matrix}}
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Base directory {base_dir} does not exist")

        representations = defaultdict(dict)
        variant_dirs = [d for d in base_path.iterdir() if d.is_dir()]

        logger.info(f"Found {len(variant_dirs)} variant directories")

        for variant_dir in tqdm(variant_dirs, desc="Loading variants"):
            variant_name = variant_dir.name
            pt_files = list(variant_dir.glob("*.pt"))

            logger.info(f"Processing variant '{variant_name}' with {len(pt_files)} files")

            for pt_file in pt_files:
                try:
                    # Extract expose ID from filename (e.g., "expose_1_M.pt" -> "expose_1")
                    expose_id = pt_file.stem.replace("_M", "")

                    # Load the tensor data with weights_only=False for custom classes
                    # This is safe if you trust the source of your .pt files
                    data = torch.load(pt_file, map_location='cpu', weights_only=False)

                    # Validate data structure
                    if not self._validate_data_structure(data):
                        logger.warning(f"Invalid data structure in {pt_file}, skipping")
                        continue

                    # Extract representation matrix using mask
                    M = data['M']
                    mask = data['mask']

                    # Apply mask to get valid rows
                    valid_representation = M[mask.bool()]

                    # Memory management: subsample if too large
                    if len(valid_representation) > self.config.max_samples_per_variant:
                        indices = np.random.choice(
                            len(valid_representation),
                            self.config.max_samples_per_variant,
                            replace=False
                        )
                        valid_representation = valid_representation[indices]

                    representations[variant_name][expose_id] = valid_representation

                except Exception as e:
                    logger.error(f"Error loading {pt_file}: {e}")
                    continue

        # Store for later use
        self.representations = dict(representations)
        self.variant_names = list(self.representations.keys())

        # Get all unique expose IDs
        all_expose_ids = set()
        for variant_data in self.representations.values():
            all_expose_ids.update(variant_data.keys())
        self.expose_ids = sorted(list(all_expose_ids))

        logger.info(f"Loaded {len(self.variant_names)} variants with {len(self.expose_ids)} exposés")
        return self.representations

    def _validate_data_structure(self, data: Dict) -> bool:
        """Validate the structure of loaded data."""
        required_keys = ['M', 'mask']
        return all(key in data for key in required_keys)

    def _normalize_representation(self, representation: torch.Tensor) -> torch.Tensor:
        """Normalize representation using StandardScaler."""
        if not self.config.normalize_representations:
            return representation

        # Convert to numpy for sklearn
        repr_np = representation.numpy()
        scaler = StandardScaler()
        normalized = scaler.fit_transform(repr_np)
        return torch.from_numpy(normalized)

    def _align_representations(self, repr1: torch.Tensor, repr2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align two representations to same shape for comparison."""
        min_rows = min(repr1.shape[0], repr2.shape[0])
        min_cols = min(repr1.shape[1], repr2.shape[1])

        # Truncate to common dimensions
        aligned_repr1 = repr1[:min_rows, :min_cols]
        aligned_repr2 = repr2[:min_rows, :min_cols]

        return aligned_repr1, aligned_repr2

    def compute_rsa(self) -> pd.DataFrame:
        """
        Compute Representational Similarity Analysis between all variant pairs.

        Returns:
            DataFrame with correlation results between variants
        """
        logger.info("Computing Representational Similarity Analysis (RSA)")

        results = []
        variant_pairs = [(v1, v2) for i, v1 in enumerate(self.variant_names)
                         for v2 in self.variant_names[i:]]

        for variant1, variant2 in tqdm(variant_pairs, desc="Computing RSA"):
            correlations_pearson = []
            correlations_spearman = []

            # Find common exposé IDs
            common_exposes = set(self.representations[variant1].keys()) & \
                             set(self.representations[variant2].keys())

            if not common_exposes:
                logger.warning(f"No common exposés found for {variant1} and {variant2}")
                continue

            for expose_id in common_exposes:
                try:
                    repr1 = self.representations[variant1][expose_id]
                    repr2 = self.representations[variant2][expose_id]

                    # Normalize representations
                    repr1 = self._normalize_representation(repr1)
                    repr2 = self._normalize_representation(repr2)

                    # Align representations
                    repr1_aligned, repr2_aligned = self._align_representations(repr1, repr2)

                    # Flatten for correlation computation
                    flat1 = repr1_aligned.flatten().numpy()
                    flat2 = repr2_aligned.flatten().numpy()

                    # Compute correlations
                    if len(flat1) > 1 and len(flat2) > 1:
                        pearson_r, _ = pearsonr(flat1, flat2)
                        spearman_r, _ = spearmanr(flat1, flat2)

                        if not np.isnan(pearson_r):
                            correlations_pearson.append(pearson_r)
                        if not np.isnan(spearman_r):
                            correlations_spearman.append(spearman_r)

                except Exception as e:
                    logger.warning(f"Error computing correlation for {variant1}-{variant2}, {expose_id}: {e}")
                    continue

            if correlations_pearson and correlations_spearman:
                # Compute statistics
                pearson_mean = np.mean(correlations_pearson)
                spearman_mean = np.mean(correlations_spearman)

                # Bootstrap confidence intervals
                pearson_ci = self._bootstrap_confidence_interval(correlations_pearson)
                spearman_ci = self._bootstrap_confidence_interval(correlations_spearman)

                # Statistical significance test
                _, p_value_pearson = stats.ttest_1samp(correlations_pearson, 0)
                _, p_value_spearman = stats.ttest_1samp(correlations_spearman, 0)

                results.append({
                    'variant_1': variant1,
                    'variant_2': variant2,
                    'correlation_pearson': pearson_mean,
                    'correlation_spearman': spearman_mean,
                    'p_value_pearson': p_value_pearson,
                    'p_value_spearman': p_value_spearman,
                    'confidence_interval_pearson_low': pearson_ci[0],
                    'confidence_interval_pearson_high': pearson_ci[1],
                    'confidence_interval_spearman_low': spearman_ci[0],
                    'confidence_interval_spearman_high': spearman_ci[1],
                    'n_samples': len(correlations_pearson)
                })

        return pd.DataFrame(results)

    def compute_silhouette_scores(self) -> pd.DataFrame:
        """
        Compute silhouette scores for cluster validation.

        Returns:
            DataFrame with silhouette scores per variant
        """
        logger.info("Computing Silhouette Scores for cluster validation")

        results = []

        for variant_name in tqdm(self.variant_names, desc="Computing silhouette scores"):
            variant_data = self.representations[variant_name]

            if len(variant_data) < 2:
                logger.warning(f"Insufficient data for silhouette analysis in {variant_name}")
                continue

            # Prepare data and labels
            all_representations = []
            labels = []

            for expose_id, representation in variant_data.items():
                # Normalize representation
                normalized_repr = self._normalize_representation(representation)

                # Add to collection
                all_representations.append(normalized_repr.numpy())
                labels.extend([expose_id] * len(normalized_repr))

            if len(set(labels)) < 2:
                logger.warning(f"Need at least 2 different clusters for {variant_name}")
                continue

            try:
                # Concatenate all representations
                X = np.vstack(all_representations)

                # Subsample if too large for memory
                if len(X) > self.config.max_samples_per_variant:
                    indices = np.random.choice(len(X), self.config.max_samples_per_variant, replace=False)
                    X = X[indices]
                    labels = [labels[i] for i in indices]

                # Convert labels to numeric
                unique_labels = list(set(labels))
                label_to_int = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = [label_to_int[label] for label in labels]

                # Compute silhouette score
                silhouette_avg = silhouette_score(X, numeric_labels)

                # Bootstrap for confidence interval
                bootstrap_scores = []
                for _ in range(min(100, self.config.bootstrap_samples)):  # Reduced for memory
                    indices = np.random.choice(len(X), len(X), replace=True)
                    X_boot = X[indices]
                    labels_boot = [numeric_labels[i] for i in indices]

                    if len(set(labels_boot)) > 1:  # Need multiple clusters
                        score = silhouette_score(X_boot, labels_boot)
                        bootstrap_scores.append(score)

                # Compute confidence interval
                if bootstrap_scores:
                    ci_low, ci_high = self._bootstrap_confidence_interval(bootstrap_scores)
                    std_error = np.std(bootstrap_scores)
                else:
                    ci_low = ci_high = silhouette_avg
                    std_error = 0.0

                results.append({
                    'variant': variant_name,
                    'silhouette_score': silhouette_avg,
                    'std_error': std_error,
                    'n_samples': len(X),
                    'n_clusters': len(unique_labels),
                    'confidence_interval_low': ci_low,
                    'confidence_interval_high': ci_high
                })

            except Exception as e:
                logger.error(f"Error computing silhouette score for {variant_name}: {e}")
                continue

        return pd.DataFrame(results)

    def _bootstrap_confidence_interval(self, data: List[float]) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        if not data:
            return 0.0, 0.0

        bootstrap_samples = []
        n_bootstrap = min(self.config.bootstrap_samples, 1000)  # Memory management

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, len(data), replace=True)
            bootstrap_samples.append(np.mean(sample))

        alpha = 1 - self.config.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_samples, lower_percentile)
        ci_upper = np.percentile(bootstrap_samples, upper_percentile)

        return ci_lower, ci_upper

    def create_visualizations(self, rsa_results: pd.DataFrame, silhouette_results: pd.DataFrame,
                              output_dir: str) -> None:
        """Create visualizations for the results."""
        logger.info("Creating visualizations")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. RSA Heatmap (Pearson correlations)
        if not rsa_results.empty:
            plt.figure(figsize=(10, 8))

            # Create correlation matrix
            variants = sorted(set(rsa_results['variant_1'].tolist() + rsa_results['variant_2'].tolist()))
            corr_matrix = np.zeros((len(variants), len(variants)))

            for _, row in rsa_results.iterrows():
                i = variants.index(row['variant_1'])
                j = variants.index(row['variant_2'])
                corr_matrix[i, j] = row['correlation_pearson']
                corr_matrix[j, i] = row['correlation_pearson']  # Symmetric matrix

            # Fill diagonal with 1.0
            np.fill_diagonal(corr_matrix, 1.0)

            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                        xticklabels=variants, yticklabels=variants,
                        square=True, linewidths=0.5, cbar_kws={"shrink": .8})

            plt.title('Representational Similarity Analysis (RSA)\nPearson Correlations',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Encoder Variants', fontweight='bold')
            plt.ylabel('Encoder Variants', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_path / 'rsa_heatmap_pearson.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. RSA Heatmap (Spearman correlations)
            plt.figure(figsize=(10, 8))

            corr_matrix_spearman = np.zeros((len(variants), len(variants)))
            for _, row in rsa_results.iterrows():
                i = variants.index(row['variant_1'])
                j = variants.index(row['variant_2'])
                corr_matrix_spearman[i, j] = row['correlation_spearman']
                corr_matrix_spearman[j, i] = row['correlation_spearman']

            np.fill_diagonal(corr_matrix_spearman, 1.0)

            sns.heatmap(corr_matrix_spearman, annot=True, cmap='RdYlBu_r', center=0,
                        xticklabels=variants, yticklabels=variants,
                        square=True, linewidths=0.5, cbar_kws={"shrink": .8})

            plt.title('Representational Similarity Analysis (RSA)\nSpearman Correlations',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Encoder Variants', fontweight='bold')
            plt.ylabel('Encoder Variants', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_path / 'rsa_heatmap_spearman.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Silhouette Scores Bar Plot
        if not silhouette_results.empty:
            plt.figure(figsize=(12, 6))

            # Sort by silhouette score
            silhouette_sorted = silhouette_results.sort_values('silhouette_score', ascending=True)

            bars = plt.barh(range(len(silhouette_sorted)), silhouette_sorted['silhouette_score'],
                            color=sns.color_palette("viridis", len(silhouette_sorted)))

            # Calculate error bar values correctly to avoid negative values
            lower_errors = []
            upper_errors = []

            for _, row in silhouette_sorted.iterrows():
                score = row['silhouette_score']
                ci_low = row['confidence_interval_low']
                ci_high = row['confidence_interval_high']

                # Ensure error bars are non-negative
                lower_error = max(0, score - ci_low)
                upper_error = max(0, ci_high - score)

                lower_errors.append(lower_error)
                upper_errors.append(upper_error)

            # Add error bars with corrected values
            plt.errorbar(silhouette_sorted['silhouette_score'], range(len(silhouette_sorted)),
                         xerr=[lower_errors, upper_errors],
                         fmt='none', color='black', capsize=3, capthick=1)

            plt.yticks(range(len(silhouette_sorted)), silhouette_sorted['variant'])
            plt.xlabel('Silhouette Score', fontweight='bold')
            plt.ylabel('Encoder Variants', fontweight='bold')
            plt.title('Cluster Validation: Silhouette Scores by Encoder Variant',
                      fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)

            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, silhouette_sorted['silhouette_score'])):
                plt.text(score + 0.01, i, f'{score:.3f}', va='center', fontweight='bold')

            plt.tight_layout()
            plt.savefig(output_path / 'silhouette_scores.png', dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Visualizations saved to {output_path}")

    def export_results(self, rsa_results: pd.DataFrame, silhouette_results: pd.DataFrame,
                       output_dir: str) -> None:
        """Export results to CSV and JSON formats."""
        logger.info("Exporting results")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export RSA results
        if not rsa_results.empty:
            rsa_results.to_csv(output_path / 'rsa_results.csv', index=False)
            if 'json' in self.config.output_formats:
                # Convert DataFrame to JSON-serializable format
                rsa_json = rsa_results.copy()
                for col in rsa_json.columns:
                    if rsa_json[col].dtype in ['float32', 'float64']:
                        rsa_json[col] = rsa_json[col].astype(float)
                    elif rsa_json[col].dtype in ['int32', 'int64']:
                        rsa_json[col] = rsa_json[col].astype(int)
                rsa_json.to_json(output_path / 'rsa_results.json', orient='records', indent=2)

        # Export silhouette results
        if not silhouette_results.empty:
            silhouette_results.to_csv(output_path / 'silhouette_results.csv', index=False)
            if 'json' in self.config.output_formats:
                # Convert DataFrame to JSON-serializable format
                silhouette_json = silhouette_results.copy()
                for col in silhouette_json.columns:
                    if silhouette_json[col].dtype in ['float32', 'float64']:
                        silhouette_json[col] = silhouette_json[col].astype(float)
                    elif silhouette_json[col].dtype in ['int32', 'int64']:
                        silhouette_json[col] = silhouette_json[col].astype(int)
                silhouette_json.to_json(output_path / 'silhouette_results.json', orient='records', indent=2)

        # Create summary statistics
        summary_stats = self._compute_summary_statistics(rsa_results, silhouette_results)
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(output_path / 'summary_statistics.csv', index=False)

        if 'json' in self.config.output_formats:
            with open(output_path / 'summary_statistics.json', 'w') as f:
                json.dump(summary_stats, f, indent=2)

        logger.info(f"Results exported to {output_path}")

    def _compute_summary_statistics(self, rsa_results: pd.DataFrame,
                                    silhouette_results: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for the evaluation."""
        summary = {
            'n_variants': len(self.variant_names),
            'n_exposes': len(self.expose_ids),
            'variant_names': self.variant_names,
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }

        if not rsa_results.empty:
            summary.update({
                'rsa_mean_pearson_correlation': float(rsa_results['correlation_pearson'].mean()),
                'rsa_std_pearson_correlation': float(rsa_results['correlation_pearson'].std()),
                'rsa_mean_spearman_correlation': float(rsa_results['correlation_spearman'].mean()),
                'rsa_std_spearman_correlation': float(rsa_results['correlation_spearman'].std()),
                'rsa_n_comparisons': int(len(rsa_results))
            })

        if not silhouette_results.empty:
            summary.update({
                'silhouette_mean_score': float(silhouette_results['silhouette_score'].mean()),
                'silhouette_std_score': float(silhouette_results['silhouette_score'].std()),
                'silhouette_best_variant': str(silhouette_results.loc[
                    silhouette_results['silhouette_score'].idxmax(), 'variant'
                ]),
                'silhouette_best_score': float(silhouette_results['silhouette_score'].max())
            })

        return summary

    def run_evaluation(self, base_dir: str, output_dir: str) -> None:
        """Run the complete evaluation pipeline."""
        logger.info("Starting encoder ablation evaluation")

        try:
            # Load data
            self.load_all_variants(base_dir)

            # Perform RSA
            rsa_results = self.compute_rsa()
            logger.info(f"RSA completed with {len(rsa_results)} variant pairs")

            # Compute silhouette scores
            silhouette_results = self.compute_silhouette_scores()
            logger.info(f"Silhouette analysis completed for {len(silhouette_results)} variants")

            # Create visualizations
            self.create_visualizations(rsa_results, silhouette_results, output_dir)

            # Export results
            self.export_results(rsa_results, silhouette_results, output_dir)

            logger.info("Evaluation completed successfully")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Encoder Ablation Evaluation')
    parser.add_argument('--input_dir', help='Input directory with encoder outputs',
                        default="data/ablation_output")
    parser.add_argument('--output_dir', help='Output directory for results',
                        default="data/ablation_results")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)
    if args.seed:
        config.random_seed = args.seed

    # Initialize evaluator
    evaluator = EncoderAblationEvaluator(config)

    # Run evaluation
    evaluator.run_evaluation(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()