# pyspark-census

**Modular PySpark-based ETL and clustering framework for big data analysis — demonstrated on U.S. Census datasets.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

---

## 🚀 Features

- ⚙️ Scalable ETL pipeline using PySpark
- 🧠 Built-in clustering algorithms (KMeans, DBSCAN, ...)
- 🔍 Easily extendable to other large-scale datasets
- 📊 Visualize clustering and dimensionality reduction (PCA)
- 🧪 Clean modular structure for experimentation

## 📦 Installation

```bash
pip install git+https://github.com/krjii/pyspark-census.git
```

Or clone and install with Poetry:

```bash
git clone https://github.com/krjii/pyspark-census.git
cd pyspark-census
poetry install
```

---

## 🛠️ Usage

```python
from pyspark_census.clustering.kmeans_clusterer import KMeansClusterer

kmeans = KMeansClusterer(k=6)
kmeans.fit(dataframe)
kmeans.plot_clusters()
```

Or run the CLI script:

```bash
python src/pyspark_census/run_pipeline.py --year 2020 --cluster dbscan
```

---

## 🧬 Project Structure

```
src/
└── pyspark_census/
    ├── data_processing/
    ├── clustering/
    │   ├── kmeans_clusterer.py
    │   └── dbscan_clusterer.py
    ├── visualization/
    └── __init__.py
```

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## 👤 Author

**Kevin James**  
📧 [krjii@indyhustles.com](mailto:krjii@indyhustles.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/krjii/)
