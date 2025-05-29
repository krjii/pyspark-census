
from pyspark.storagelevel import StorageLevel
from scipy.stats.mstats import winsorize

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


class ExploratoryAnalyzer(object):
    """Distributed computation of statistics and standardization using PySpark RDDs.
    Responsible for parsing the US Census csv files into RDDs.

    Handles:
        - Feature extraction for key indicators tied to gentrification risk.
            — percent renters
            - education index 
            - percent recent movers 
            - median household income

    All processing done without DataFrames or Pandas to support large-scaled distributed work flows using only PySpark RDDs.
        Keep your RDD long-formatted while doing:
        Normalization
        Feature-wise stats
        Winsorizing
        Filtering/extremes

    Convert to wide format (or DataFrame) after normalization for:
        Modeling (NeuralForecast, ML)
        Visualization
        Saving to Parquet/CSV
  
    """

    def __init__(self, standardized_rdd=None):
        """ RDD format is [(('GEO_ID', 'NAME'), {...})]
        """    
        # RDD format is ((geo_id, name), {field1: val1, field2: val2, ...})
        # No longer list
        self.all_features_rdd = standardized_rdd
        if standardized_rdd is not None:
            self.all_features_rdd = standardized_rdd.map(lambda x: (x[0], x[1]))
    

    def plot_standardized_outliers(self, outliers, threshold=3.0):
        """
        Plot standardized values and highlight outliers.
        Expects input from find_outliers_zscore: (rdd, feature_name)
        """
        rdd_feature_collected = outliers[0].collect()
        feature_name = outliers[1]
    
        values = [v[1] for v in rdd_feature_collected]
        labels = [f"{k[0]}-{k[1]}" for k, _ in rdd_feature_collected]
    
        outlier_indices = [i for i, v in enumerate(values) if abs(v) > threshold]
        outlier_values = [values[i] for i in outlier_indices]
        normal_indices = [i for i, v in enumerate(values) if abs(v) <= threshold]
        normal_values = [values[i] for i in normal_indices]
    
        plt.figure(figsize=(12, 6))
        plt.plot(normal_indices, normal_values, 'bo', label="Normal Values", markersize=3)
        plt.plot(outlier_indices, outlier_values, 'ro', label=f"Outliers (|z|>{threshold})", markersize=5)
        plt.axhline(y=threshold, color='g', linestyle='--', label=f"+{threshold}")
        plt.axhline(y=-threshold, color='g', linestyle='--', label=f"-{threshold}")
    
        plt.title(f"Standardized {feature_name} Values")
        plt.xlabel("Data Point Index")
        plt.ylabel("Z-Score")
        plt.legend()
        plt.grid(True)
        plt.show(block=True)
    
    def extract_raw_features(self, data_rdd, feature_list):
        """
        Extract multiple features from data_rdd into separate (Name, value) RDDs.
        Args:
            data_rdd: RDD of ((geo_id, name), {field: value, ...})
            feature_list: List of feature field names to extract
        Returns:
            dict: {feature_name: corresponding (year, value) RDD}
        """
        feature_rdds = {}
        
        for feature in feature_list:
            rdd = (
                data_rdd
                .filter(lambda x: feature in x[1])  # field must exist
                .map(lambda x: (x[0][1], x[1][feature]))  # (Name, value)
                .filter(lambda kv: str(kv[0]).isdigit() and str(kv[1]).strip() not in ('', 'null', 'N/A', None))  # clean
                .map(lambda kv: (int(kv[0]), float(kv[1])))  # type cast
                .filter(lambda kv: kv[1] >= 0)  # optional: no negatives
            )
            feature_rdds[feature] = rdd
        
        return feature_rdds
    
    def plot_data_distribution(self, data_rdd, features=[]):
        """
        data_rdd format = (('GEO_ID', 'NAME'), ResultIterable)
        Reiterable contains = [('B19013_001E', 76030), ('B07001_001E', 16633), ...]

        """
        feature_rdds = self.extract_raw_features(data_rdd, features)
        
        for feature, rdd in feature_rdds.items():
            values = [v for (Name, v) in rdd.collect()]
            
            plt.figure(figsize=(10,6))
            plt.hist(values, bins=50, color='skyblue', edgecolor='black')
            plt.title(f"Distribution of {feature}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.show(block=True)
        
        print()
        
        #Collect values for plotting
        #raw_values = [v for (y, v) in raw_feature_rdd.collect()]

        #Plot distribution
        # plt.figure(figsize=(12, 6))
        # plt.hist(raw_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        # plt.title("Distribution of B19013_001E (Raw Feature Values from merged_rdd)")
        # plt.xlabel("Value")
        # plt.ylabel("Frequency")
        # plt.grid(True)
        # plt.tight_layout()
        #
        # plt.show(block=True)
        
    @staticmethod
    def is_valid_number(v):
        if v is None:
            return False
        if isinstance(v, str):
            v = v.strip()
            if v in ('-', '', 'null', 'N/A'):
                return False
        try:
            float(v.replace(',', '').replace('+', ''))
            return True
        except:
            return False

    def distributed_summary(self, rdd_labeled, label="Feature"):
        """Distributed mean, std, min, max using Spark aggregations."""
        
        count = rdd_labeled.count()
        if count == 0:
            print(f"{label}: No data.")
            return

        sum_val = rdd_labeled.map(lambda x: x[1]).sum()
        mean = sum_val / count

        sum_sq_diff = rdd_labeled.map(lambda x: (x[1] - mean) ** 2).sum()
        std = (sum_sq_diff / count) ** 0.5

        min_val = rdd_labeled.map(lambda x: x[1]).min()
        max_val = rdd_labeled.map(lambda x: x[1]).max()

        print(f"\n--- {label} (Distributed Summary) ---")
        print(f"Count     : {count}")
        print(f"Mean      : {mean:.3f}")
        print(f"Std Dev   : {std:.3f}")
        print(f"Min       : {min_val:.3f}")
        print(f"Max       : {max_val:.3f}")

    def yearly_distribution(self, rdd_nested, label="Feature"):
        """
        Compute mean, min, max per year for a specified feature using the nested RDD format.
        Expects input RDD of (geo_id, {year: feature_dict}).
        """
        
        def extract_year_feature_tuples(record):
            geo_id, year_dicts = record
        
            if not isinstance(year_dicts, dict):
                return []  # or: return [(geo_id, None)] if you want to keep track
        
            return [
                ((geo_id, year), features)
                for year, features in year_dicts.items()
                if isinstance(features, dict)
            ]
    
        rdd_labeled = rdd_nested.flatMap(extract_year_feature_tuples)
    
        def seq_stats(acc, value):
            count, total, min_v, max_v = acc
            return (count + 1, total + value, min(min_v, value), max(max_v, value))
    
        def comb_stats(a, b):
            return (
                a[0] + b[0],
                a[1] + b[1],
                min(a[2], b[2]),
                max(a[3], b[3])
            )
    
        stats = (
            rdd_labeled
            .aggregateByKey(
                (0, 0.0, float("inf"), float("-inf")),
                seq_stats,
                comb_stats
            )
            .mapValues(lambda x: (x[1] / x[0], x[2], x[3]))  # mean, min, max
            .sortByKey()
        )
    
        stats_data = stats.collect()
        
        print(f"\n--- {label} by Year (Mean, Min, Max) ---")
        for year, (mean, min_v, max_v) in stats_data:
            print(f"{year}: Mean={mean:.2f}  Min={min_v:.2f}  Max={max_v:.2f}")
            
        return stats_data
    
    def zip_level_delta(self, rdd_labeled, label="Feature", top_n=10):
        """Compute (last - first) value per ZIP code and print top changers."""

        zip_series = (
            rdd_labeled
            .map(lambda x: (x[0][0], (x[0][1], x[1])))  # (zip, (year, value))
            .groupByKey()
            .mapValues(lambda series: sorted(series))
            .mapValues(lambda vals: vals[-1][1] - vals[0][1] if len(vals) > 1 else 0.0)
        )

        top_changes = zip_series.takeOrdered(top_n, key=lambda x: -abs(x[1]))

        print(f"\n--- {label} Top {top_n} ZIP Δ (Change Over Time) ---")
        for zcta, delta in top_changes:
            print(f"{zcta}: Δ = {delta:.2f}")

    def exploratory_stats(self, records):
        """                count    mean    std    min    25%    50%    75%    max    missing_values    zero_count    unique_values
            feature_1      1500    0.24    0.05    ...    ...    ...    ...    ...    0                    12            200
            feature_2      1500    0.02    0.01    ...    ...    ...
        """
        df = pd.DataFrame(records.collect())
        
        static_fields = ['unique_id', 'ds', 'year', 'y']
        features = [col for col in df.columns if col not in static_fields]
        
        exploratory_stats = df[features].describe().transpose()
        exploratory_stats['missing_values'] = df[features].isnull().sum()
        exploratory_stats['zero_count'] = (df[features] == 0).sum()
        exploratory_stats['unique_values'] = df[features].nunique()

        print(exploratory_stats)
        
        # Save to CSV
        os.makedirs("./exploratory", exist_ok=True)
        exploratory_stats.to_csv("./exploratory/exploratory_stats_overall.csv")
        
        # Per-year summaries
        per_year_stats = df.groupby("year")[features].describe().transpose()
        per_year_stats.to_csv("./exploratory/exploratory_stats_per_year.csv")
        
        # Histogram plots
        for feature in features:
            plt.figure(figsize=(8, 4))
            sns.histplot(data=df, x=feature, hue='year', kde=True, element="step")
            plt.title(f"Histogram of {feature}")
            plt.savefig(f"./exploratory/hist_{feature}.png")
            plt.close()
        
        # Pair plot
        pair_plot = sns.pairplot(df[features + ['year']], hue='year')
        pair_plot.savefig("./exploratory/pair_plot.png")
        plt.close()
        
        # Output file list for user reference
        output_files = os.listdir("./exploratory")
        output_files

    def feature_record(self, feature_dict, feature_keys=None):
        """
        Convert a dict of {(unique_id, year): feature_dict} into a list of records.
    
        Args:
            feature_dict (dict): {(unique_id, year): {feature_key: value, ...}}
            feature_keys (list): Optional list of keys to include from the features
    
        Returns:
            List of dicts sorted by 'ds' field (e.g., '2013-01-01')
        """
        feature_keys = feature_keys or []
        records = []
    
        for keys, features in feature_dict.items():
            if isinstance(keys, tuple) and len(keys) >= 2:
                unique_id, year = keys[0], keys[1]
            else:
                print(f"[SKIP] Unexpected key format: {keys}")
                continue
    
            if not isinstance(features, dict):
                print(f"[SKIP] Values not a dict: {features}")
                continue
    
            try:
                record = {
                    "unique_id": unique_id,
                    "ds": f"{year}-01-01",
                    "year": year
                }
    
                # Only include provided feature keys (if any)
                for key in feature_keys:
                    record[key] = features.get(key, None)
    
                records.append(record)
            except Exception as e:
                print(f"[ERROR] Failed to process {keys}: {e}")
    
        return sorted(records, key=lambda d: d["ds"])

    def plot_eda(self, records):
        """
            Avoids overplotting from extreme outliers.
        
            Adjusts scale per feature using quantiles.
        
            Keeps plots readable even for highly skewed data.
        """
            
        df = pd.DataFrame(records.collect())
        static_fields = ['unique_id', 'ds', 'year', 'y', "GEO_ID", "NAME", "region", "state", "year"]
        features = [col for col in df.columns if col not in static_fields]
        
        for feature in features:
            plt.figure(figsize=(8, 4))
            
            # Drop NaNs to avoid plotting issues
            values = df[feature].dropna()
        
            # Use 1st to 99th percentile to limit extreme outliers
            q_low, q_high = values.quantile([0.01, 0.99])
        
            # Optional: Clip values to better scale histogram
            clipped_values = values.clip(lower=q_low, upper=q_high)
        
            sns.histplot(clipped_values, kde=True, bins=30)
            plt.title(f'Histogram of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            
            # Only set xlim if range is not zero
            if clipped_values.min() != clipped_values.max():
                plt.xlim(clipped_values.min(), clipped_values.max())
            
            plt.tight_layout()
            plt.show(block=False)
            plt.savefig(f"./exploratory/hist_{feature}.png")
            #plt.close()

    def standardize_per_year(self, raw_data):
        """
        Standardize and winsorize feature values per year using Z-score.
    
        Input: 
            flattened_data: RDD[((geo_id, year), {feature: value, ...})]
    
        Output:
            final_normalized_rdd: RDD[((geo_id, year), {feature: {'raw_winsorized': val, 'z': z_val}, ...})]
        """
    
        excluded_keys = {"GEO_ID", "NAME", "region", "state", "year"}

        # Step 1: Compute mean and std per (year, feature)
        feature_triplets = raw_data.flatMap(
            lambda x: [
                ((x[0][1], feature), (val, val**2, 1))
                for feature, val in x[1].items()
                if feature not in excluded_keys and isinstance(val, (int, float))
            ]
        )
    
        # Step 1: Aggregate raw sums, sums of squares, and counts
        feature_stats_raw = feature_triplets.aggregateByKey(
            (0.0, 0.0, 0),
            lambda acc, val: (acc[0] + val[0], acc[1] + val[1], acc[2] + val[2]),
            lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])
        ).persist(StorageLevel.MEMORY_AND_DISK)
        
        # Step 2: Compute (mean, std) from aggregates
        stats_kv = feature_stats_raw.mapValues(lambda acc: (
            acc[0] / acc[2],
            ((acc[1] / acc[2]) - (acc[0] / acc[2]) ** 2) ** 0.5
        )).persist(StorageLevel.MEMORY_AND_DISK)
    
        # Step 2: Flatten for winsorizing
        flat_rdd = raw_data.flatMap(
            lambda x: [
                ((x[0][1], feature), (x[0][0], float(val)))
                for feature, val in x[1].items()
                if feature not in excluded_keys and isinstance(val, (int, float))
            ]
        )
    
        # Step 3: Combine values per (year, feature)
        def create_comb(val): return [val]
        def merge_val(acc, val): acc.append(val); return acc
        def merge_combs(a, b): a.extend(b); return a
    
        grouped = flat_rdd.combineByKey(create_comb, merge_val, merge_combs).persist(StorageLevel.MEMORY_AND_DISK)

        # Step 4: Winsorize
        winsorized = grouped.mapValues(lambda vals: {
            geo_id: wval for (geo_id, wval) in zip(
                [gid for gid, _ in vals],
                winsorize(np.array([val for _, val in vals], dtype=np.float64), limits=[0.05, 0.05])
            )
        }).persist(StorageLevel.MEMORY_AND_DISK)
    
        # Step 5: Join with (mean, std) stats
        z_input = winsorized.join(stats_kv).persist(StorageLevel.MEMORY_AND_DISK)
    
        # Step 6: Flatten and compute z-scores
        flat_z = z_input.flatMap(
            lambda x: [
                ((geo_id, x[0][0]), (x[0][1], val, x[1][1]))  # (geo_id, year), (feature, val, (mean, std))
                for geo_id, val in x[1][0].items()
            ]
        ).repartition(400)  # adjust based on executor count
    
        combined = flat_z.combineByKey(
            lambda x: [x],              # Create list with first feature tuple
            lambda acc, x: acc + [x],   # Append new feature tuple to list
            lambda a, b: a + b          # Merge two partitions' lists
        ).persist(StorageLevel.MEMORY_AND_DISK)
    
        # Step 7: Build nested dicts {feature: {'raw_winsorized': val, 'z': z}}
        z_scored = combined.mapValues(lambda features: {
            f: {
                'raw_winsorized': val,
                'z': 0.0 if std == 0 else (val - mean) / std
            } for f, val, (mean, std) in features
        }).persist(StorageLevel.MEMORY_AND_DISK)    
        
        # Step 8: Add back static fields
        static_fields_rdd = raw_data.map(
            lambda x: (x[0], {k: x[1][k] for k in ['GEO_ID', 'NAME', 'region', 'state'] if k in x[1]})
        ).persist(StorageLevel.MEMORY_AND_DISK)
    
        final_normalized_rdd = z_scored.join(static_fields_rdd).map(
            lambda x: (x[0], {**x[1][0], **x[1][1], 'year': x[0][1]})
        ).persist(StorageLevel.MEMORY_AND_DISK)
    
        return final_normalized_rdd
    
    def flatten_z_safe(self, record):
        flat = {}
        for k, v in record.items():
            if isinstance(v, dict) and 'z' in v:
                val = v['z']
                flat[k] = float(val) if isinstance(val, np.generic) else val
            else:
                flat[k] = v
        return flat

