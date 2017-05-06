# Data Science Retreat - Applied Machine Learning with Apache Spark

- If you have a Linux installation and you feel like starting from scratch, you can go straight to Section 3. It should work with minimal changes if you are using a Mac

- However, if you are using Windows, you can choose between using a Docker container (Section 1), assuming you have already successfully installed Docker, or using an AWS EC2 Instance (Section 2), assuming you already have an AWS account.

## Using Docker 

### 1.0 Clone the Repository
- git clone https://github.com/dvgodoy/DSR-Spark-AppliedML.git

### 2.0 Run the container from the image at DockerHub (dvgodoy/dsr-spark-appliedml), naming it dsr-spark-appliedml, making your local folder with the repository and all its notebooks accessible inside the container in the folder /home/jovyan/work/DSR
```bash
docker run -v /path/to/DSR-Spark-AppliedML:/home/jovyan/work/DSR --name dsr-spark-appliedml -it --rm -p 8888:8888 dvgodoy/dsr-spark-appliedml:latest
```
- It will start a MySQL database server and the Jupyter Notebook. You should see a message like:
```bash
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=13f10e95d96d50eb156ca6e108f39ece7e0a8560eea11c44
```
- After copying the URL in your browser (please note token will be different from the example!), you should see a DSR folder containing the contents of the repository
- This image is built on top of of [Docker stacks' Pyspark Notebook](https://github.com/jupyter/docker-stacks/tree/master/pyspark-notebook)

### 3.0 If you want to access the container:
```bash
docker exec -it dsr-spark-appliedml bash
```

## Using AWS EC2 Image
### 2.0 Go for the EC2 menu

### 2.1 Click on Launch Instance

### 2.2 Look for the AMI ID ami-3bb56c5b in Community AMIs

### 2.3 When asked for, create a new key pair - download it and keep it safe!

### 2.4 When asked for, create a new security group with the following rules:
- SSH with source Anywhere
- HTTPS with source Anywhere
- Custom TCP Rule with Port 8888 and source Anywhere

### 2.5 After your instance is ready, you can SSH into it:
- ssh -i mykeypairfile.pem ubuntu@ec2-XX-XX-XX-XX.us-west-2.compute.amazonaws.com

### 2.6 Update and then install Git
- sudo apt-get update
- sudo apt-get install git

### 2.7 Clone the Repository
- git clone https://github.com/dvgodoy/DSR-SparkAppliedML.git

### 2.8 Run PySpark
- cd DSR-SparkAppliedML
- pyspark OR nohup pyspark

## Manual Installation
### 3.1 You should have Java 8 installed, otherwise:
- sudo add-apt-repository ppa:webupd8team/java
- sudo apt-get update
- sudo apt-get install oracle-java8-installer

### 3.2 You should have Anaconda installed, otherwise:
- wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh
- bash Anaconda2-4.2.0-Linux-x86_64.sh
- source ~/.bashrc

### 3.3 You should have PY4J and SPARK-SKLEARN packages installed, otherwise:
- pip install py4j
- pip install spark-sklearn

### 3.4 You should have MySQL installed, otherwise:
- sudo apt-get install mysql-server

### 3.5 You should have MySQL Connector/J JAR file available, otherwise:
- wget https://dev.mysql.com/get/Downloads/Connector-J/mysql-connector-java-5.1.39.tar.gz
- tar -xvf mysql-connector-java-5.1.39.tar.gz 

### 3.6 You should have Spark 2.0 installed, otherwise:
- wget http://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-hadoop2.7.tgz
- tar -xvf spark-2.0.0-bin-hadoop2.7.tgz
- mv spark-2.0.0-bin-hadoop2.7 spark

### 3.7 If you are performing the installation on an EC2 instance, you should follow these steps:
#### 3.7.0 Make sure the Security Group associated with your EC2 instance has the following rules:
- SSH with source Anywhere
- HTTPS with source Anywhere
- Custom TCP Rule with Port 8888 and source Anywhere

#### 3.7.1 Generate your own SSL Certificate
- mkdir certificates
- cd certificates
- openssl genrsa -out server.key 1024
- openssl req -new -key server.key -out server.csr
- openssl x509 -req -days 366 -in server.csr -signkey server.key -out server.crt
- cat server.crt server.key > server.pem

#### 3.7.2 Create Jupyter Notebook config file
- jupyter notebook --generate-config
- cd ~/.jupyter
- vi jupyter_notebook_config.py
	- c = get_config()
	- c.IPKernelApp.pylab = 'inline'
	- c.NotebookApp.certfile = '/home/ubuntu/certificates/mycert.pem'
	- c.NotebookApp.ip = '*'
	- c.NotebookApp.open_browser = False
	- c.NotebookApp.port = 8888

### 3.8 Apache Spark - you have to add packages/jars so Spark can handle XML and JDBC sources
- cd /home/ubuntu/spark/conf
- cp spark-defaults.conf.template spark-defaults.conf
- vi spark-defaults.conf
	- spark.jars.packages    com.databricks:spark-xml_2.11:0.4.0
	- spark.jars	         /home/ubuntu/mysql-connector-java-5.1.39/mysql-connector-java-5.1.39-bin.jar

### 3.9 Environment Variables - you have to add this variables, so you can easily run PySpark as a Jupyter Notebook
- vi ~/.bashrc
	- export JAVA_HOME="/usr/lib/jvm/java-8-oracle"
	- export SPARK_HOME="/home/ubuntu/spark"
	- export PATH="$SPARK_HOME/bin:$SPARK_HOME:$PATH"
	- export PYSPARK_DRIVER_PYTHON="jupyter"
	- export PYSPARK_DRIVER_PYTHON_OPTS="notebook"

### 3.10 Clone the Repository
- git clone https://github.com/dvgodoy/DSR-SparkAppliedML.git

### 3.11 Run PySpark
- cd DSR-SparkAppliedML
- pyspark OR nohup pyspark