
import json
import logging
import os
import shutil


class RddReaderWriter(object):
    """Class reads saved census data file into RDD
    """


    def __init__(self):
        logging.debug("Initializing RDD Reader")
    
    def save_raw_rdd(self, data_rdd, output_path: str = 'data_rdd_json'):
        
        # --- Save ---
        rdd_json = data_rdd.map(lambda x: json.dumps({
            "key": x[0],        # geo_id string
            "value": x[1]       # {year: {...}}
        }))
    
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        
        rdd_json.saveAsTextFile(output_path)
     

    def load_raw_rdd(self, sc, data_file):
        # --- Later, Load ---
        rdd_loaded = sc.textFile(data_file)
        
        rdd_reconstructed = rdd_loaded.map(lambda line: (
            tuple(json.loads(line)["key"]),
            dict(json.loads(line)["value"])
        )) 
        
        return rdd_reconstructed 
    
    def load_features_rdd(self, sc, file_path):
        # Load the saved JSON lines (entire folder if saved with saveAsTextFile)
        rdd_json = sc.textFile(file_path)
        
        # Parse each line and convert back to the original (tuple) format
        # Wide format
        computed_features_rdd = rdd_json.map(lambda line: json.loads(line)) \
             .map(lambda obj: (
                 tuple(obj["key"]) if isinstance(obj["key"], list) else obj["key"],
                 obj["value"]
             ))
        
        geo_year_pairs = computed_features_rdd.map(lambda x: (x[0][0], x[0][1]))  # (GEO_ID, year)
        #
        expected_years = {str(y) for y in range(2010, 2024)}
        #
        missing = expected_years - set(geo_year_pairs.map(lambda x: x[1]).distinct().collect())
        print("Years missing in geo_year_pairs:", sorted(missing))
        
        unique_years = geo_year_pairs.map(lambda x: x[1]).distinct().collect()
        
        feature_records = computed_features_rdd.mapPartitions(self.stats_computer.feature_record)
        
        return feature_records, unique_years 