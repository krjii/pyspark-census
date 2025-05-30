
from pyspark import SparkContext
from collections import defaultdict, ChainMap
import csv
import glob
from io import StringIO
import re
import os
from pyspark.rdd import RDD
from pyspark.storagelevel import StorageLevel


class CensusGeoReader():
    """Census GeoReader is developed to parse US census data csv files into Spark RDD data objects.
    This class is designed for Big Data applications that may require distributed processing of large
    U.S. Census data.
    """

    def __init__(self, desired_features: list = []):
        self.fips_to_state = {
                    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California",
                    "08": "Colorado", "09": "Connecticut", "10": "Delaware", "11": "District of Columbia", "12": "Florida",
                    "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois", "18": "Indiana",
                    "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
                    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota", "28": "Mississippi",
                    "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada", "33": "New Hampshire",
                    "34": "New Jersey", "35": "New Mexico", "36": "New York", "37": "North Carolina", "38": "North Dakota",
                    "39": "Ohio", "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
                    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas", "49": "Utah",
                    "50": "Vermont", "51": "Virginia", "53": "Washington", "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
                    "60": "American Samoa", "66": "Guam", "69": "Northern Mariana Islands", "72": "Puerto Rico",
                    "74": "U.S. Minor Outlying Islands", "78": "U.S. Virgin Islands"
                }
        
        self.essential_fields = desired_features + ['GEO_ID', 'NAME', 'region', 'state', 'year']
            

    def filter_geos_missing_years(self, long_rdd, n_consecutive_years = None):
        if n_consecutive_years is None:
            unique_years = long_rdd.map(lambda x: x[0][1]).countByValue()
            n_consecutive_years = int(len(unique_years) / 2)
       
        #Filter geo_ids that have ≥ n consecutive years
        def has_n_consec(years, n):
            years = sorted(int(y) for y in years)
            max_count = count = 1
            for i in range(1, len(years)):
                if years[i] == years[i - 1] + 1:
                    count += 1
                    max_count = max(max_count, count)
                else:
                    count = 1
            return max_count >= n
    
        geo_year_pairs = long_rdd.map(lambda x: (x[0][0], x[0][1]))

        # Using aggregateByKey to build a set of years per geo_id
        geo_to_years = geo_year_pairs.aggregateByKey(
            set(),                                 # zeroValue
            lambda acc, y: acc.union({y}),         # seqOp
            lambda a, b: a.union(b)                # combOp
        ).mapValues(lambda s: sorted(s)).persist(StorageLevel.MEMORY_AND_DISK)

        # 2. Keep only geo_ids with enough consecutive years
        valid_geo_ids_rdd = geo_to_years.filter(lambda x: has_n_consec(x[1], n_consecutive_years)).map(lambda x: (x[0], None))
        
        # 3. Prepare long_rdd for join: (geo_id, ((geo_id, year), feature_dict))
        long_rdd_keyed = long_rdd.map(lambda x: (x[0][0], x))
        
        # 4. Join to keep only valid geo_ids (no collect, scalable!)
        filtered_long_rdd = valid_geo_ids_rdd.join(long_rdd_keyed).map(lambda x: x[1][1])

    
        return filtered_long_rdd
    
 
    def get_state_from_geo_id(self, geo_id: str) -> str:
        """Processes Census GEO ID's to determine that state associated with data beiing processed.
        GEO_ID's that start with "1400000US" represent census tract level data
        GEO_ID's that start with "8600000US" represent Zip Tabulation Area (ZCTA) level data
        
        Parameters:
            geo_id (str): Geographic identifier

        Returns:
            str:    State string
        """
        if geo_id.startswith("1400000US") or geo_id.startswith("8600000US"):
            state_fips = geo_id[9:11]  # e.g., '06' → California
        elif geo_id.startswith("ZCTA5"):
            return "ZCTA"  # or handle separately
        else:
            state_fips = geo_id[-5:-3]  # fallback logic, if format is unknown
        
        return self.fips_to_state.get(state_fips, "Unknown")

    def filter_record_features(self, record: dict, desired_fields: list = []) -> dict:
        """Processes Census record and filter to keep only the desired fields
        
        Parameters:
            record (dict): Dictionary containing Key-Value pair for U.S. census feature
            desired_fields (list): dictionary entries to retain

        Returns:
            str:    State string
        """
        feature_records = {k: v for k, v in record.items() if k in self.essential_fields}       
    
        return feature_records
    
    def get_directories_in_path(self, path: str) -> list:
        """Returns a list of directories in the given path.

          Parameters:
            path (str): The path to the directory.

          Returns:
            list: A list of directory names in the given path.
        """
        directories = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]

        return directories

    def parse_census_file(self, file: str) -> list:
        """Parse a CSV file content into (zip_area, row_dict) entries.
        This method adds a state and region property for each record.
        It also skips any records that are missing both Geo_Id and Name fields

          Parameters:
            file (str): File name to be processed

          Returns:
            list: A list of directory names in the given path.
        
        """
    
        path, content = file
        match = re.search(r'ACSDT5Y(\d{4})', path)
        year = match.group(1) if match else None
    
        if not content:
            return []
    
        lines_io = StringIO(content)
        reader = csv.reader(lines_io)
    
        all_rows = list(reader)
        if len(all_rows) < 2:
            return []
    
        headers = [h.strip('"') for h in all_rows[0]]
        if headers[0].startswith('\ufeff'):
            headers[0] = headers[0].replace('\ufeff', '')
    
        data_rows = all_rows[2:]  # skip second row
        num_headers = len(headers)
    
        area_records = defaultdict(dict)
    
        for row in data_rows:
            if len(row) > num_headers:
                row = row[:num_headers]
            row_dict = {}
    
            for header, field in zip(headers, row):
                raw_value = field.strip('"').strip()
    
                if raw_value in ('-', 'null', ''):
                    field_value = None
                else:
                    if header in ('NAME', 'GEO_ID'):
                        field_value = raw_value
                    else:
                        is_negative = raw_value.endswith('-')
                        cleaned_value = raw_value.rstrip('+-').replace(',', '')
    
                        try:
                            num_value = float(cleaned_value)
                            if is_negative:
                                num_value = -num_value
                            field_value = int(num_value) if num_value.is_integer() else num_value
                        except ValueError:
                            field_value = None  # log if needed
    
                row_dict[header] = field_value
    
            row_dict = {k: v for k, v in row_dict.items() if k.strip() != ''}
            row_dict = {k: v for k, v in row_dict.items() if not k.endswith('M')}
            row_dict['year'] = year
    
            geo_area = row_dict.get('GEO_ID') or row_dict.get('NAME')
            if geo_area:
                state, region = self.add_region_tag(geo_area)
                
                row_dict['GEO_ID'] = geo_area  # Ensure GEO_ID is always present
                row_dict['region'] = region
                row_dict['state'] = state
                area_records[(geo_area, year)].update(row_dict)
            else:
                print(f"[SKIP] Missing both GEO_ID and NAME for year {year}: {row_dict}")
    
        return list(area_records.items())

    def process_data_directory(self, sc: SparkContext, data_directory) -> RDD:
        """Processes the directory provided and identifies any subdirectories of data
        The assumption this method makes is that you have a directory of U.S. Census Detailed Table 
        data files (stored as csv files) with each sub direcotry containing a file for each year of 
        table data. The method identifies all files in that directory which end with -Data.csv which
        is the naming convention fo U.S. Census data files. 

          Parameters:
            sc (SparkContext): spark context necessary for processing data into Spark object
            data_directory (str): name of directory that stores data files

          Returns:
            RDD: A pyspark RDD containing a long formatted dictionary of US Census data
        
        """
        list_data_dir = self.get_directories_in_path(data_directory)
    
        rdd_list = []
    
        for data_subdir in list_data_dir:
            # Replace with the actual directory path
            subdirectory_path = f"{data_directory}/{data_subdir}"
            file_pattern = "*-Data.csv"  # Replace with the desired file pattern
    
            # Parse them
            # Recursive glob
            all_data_files = glob.glob(f"{subdirectory_path}/**/{file_pattern}", recursive=True)
            
            # Recursive spark read
            rdd_all = sc.wholeTextFiles(",".join(all_data_files)).flatMap(self.parse_census_file)
            rdd_filtered = rdd_all.map(lambda x: (x[0], self.filter_record_features(x[1])))
            
            rdd_list.append(rdd_filtered)
    
        # Join all
        long_formatted_rdd = sc.union(rdd_list).reduceByKey(self.merge_dicts)
        
        return long_formatted_rdd

    def merge_dicts(self, d1, d2):
        merged_dict = dict(ChainMap(d2, d1))  # d2 overwrites d1
 
        return merged_dict

    def assign_region(self, state_fips) -> str:
        """Assign U.S. Census region by state FIPS code.
            Parameters:
                state_fips (str): Fips code use to label region based on state
                
            Returns:
                str: region associated with state
   
        """
        region = ""
        
        northeast = {'09', '23', '25', '33', '34', '36', '42', '44', '50'}
        midwest   = {'17', '18', '19', '20', '26', '27', '29', '31', '38', '39', '46', '55'}
        south     = {'01', '05', '10', '11', '12', '13', '21', '22', '24', '28',
                     '37', '40', '45', '47', '48', '51', '54'}
        west      = {'02', '04', '06', '08', '15', '16', '30', '32', '35', '41', '49', '53', '56'}
    
        if state_fips in northeast:
            region = 'Northeast'
        elif state_fips in midwest:
            region = 'Midwest'
        elif state_fips in south:
            region = 'South'
        elif state_fips in west:
            region = 'West'
        else:
            region = 'Unknown'

        return region

    def add_region_tag(self, geo_id: str) -> tuple:
        """
            Parameters:
                geo_id (str): Geographic identifier
            
            Returns:
                tuple: state, region associated with the geographic identifier
        """
        state_fips = geo_id[9:11] if isinstance(geo_id, str) and geo_id.startswith("1400000US") and len(geo_id[9:]) == 11 else None
        state = self.get_state_from_geo_id(geo_id)
        
        region = self.assign_region(state_fips) if state_fips else "Unknown"
        
        # Add region and return updated record
        return state, region

