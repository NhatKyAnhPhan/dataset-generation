from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("EcommerceSyntheticGenerator")
    .master("local[*]")
    # memory cho Driver
    .config("spark.driver.memory", "16g")
    # Cho phép Spark xả bộ nhớ tạm xuống ổ đĩa thay vì giữ toàn bộ trên RAM
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.memory.offHeap.size", "4g")
    # dung lượng kết quả tối đa
    .config("spark.driver.maxResultSize", "8g")
    # số partition shuffle mặc định (giúp chia nhỏ dữ liệu khi Sort/Shuffle)
    .config("spark.sql.shuffle.partitions", "400")
    .getOrCreate()
)

input_path = "ecommerce_dataset_+1m.csv"

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(input_path)
)

print("Số dòng:", df.count())
print("Số cột:", len(df.columns))

df.printSchema()
