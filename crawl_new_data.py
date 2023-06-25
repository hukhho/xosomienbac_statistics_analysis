import pyodbc

connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-SJS8V3V;DATABASE=hunglotto;UID=sa;PWD=123"  
conn = pyodbc.connect(connection_string)  
cursor = conn.cursor()  
from datetime import datetime, timedelta

# Function to insert data into the database
def update_data_to_db(date, value, cursor):
    try:
        date_object = datetime.strptime(date, '%d-%m-%Y')
        query = """
        UPDATE [hunglotto].[dbo].[lotto]
        SET value_column = ?
        WHERE date_column = ?
        """
        cursor.execute(query, (value, date_object))
        conn.commit()
    except pyodbc.Error as err:
        print(f"Error updating data in SQL Server for date {date}: {err}")

def execute_query(query, cursor, params=None):
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    result = cursor.fetchone()

    return result.value_column if result else None

def get_value_from_db(date):
    value = None;  
    try:
        #print("OKE")
        # Convert date string to datetime object
        from datetime import datetime
        date_object = datetime.strptime(date, '%d-%m-%Y')

        # Query date_column and value_column for a specific date
        query2 = """
        SELECT date_column, value_column
        FROM [hunglotto].[dbo].[lotto]
        WHERE date_column = ?
        """
        
        value = execute_query(query2, cursor, params=(date_object,))
      
        # Close the cursor and commit changes   
       
        
        #print(f"Results for {date}: {value}")
        
        return value;  
                
    except pyodbc.Error as err:     
        print("Error connecting to SQL Server:", err)   

    
import requests 
from bs4 import BeautifulSoup

def crawl_number(date):
    def get_data(url):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            result_div = soup.find('div', {'id': 'rs_0_0'})
            if result_div:
                result = result_div.get('data-sofar')
                return result
        return None
    url = f'https://ketqua8.net/xo-so-truyen-thong.php?ngay={date}'
    result = get_data(url)
    if result:
       #print(f"Result from ketqua8.net: {result}")
        return result;
    else:
        return "NULL";
        #print("Data not found")      
          
date = "24-01-2023"        
value1 = get_value_from_db(date)
value2 = crawl_number(date)



# Get today's date
today = datetime.today()

# Loop through the next 7 days
for i in range(8):
    current_date = today - timedelta(days=i)
    date_str = current_date.strftime('%d-%m-%Y')

    # Crawl data and insert it into the database
    value = crawl_number(date_str)
    if value != "NULL":
        update_data_to_db(date_str, value, cursor)
    else:
        print(f"Data not found for date {date_str}")

# Close the cursor and the connection
cursor.close()
conn.close()