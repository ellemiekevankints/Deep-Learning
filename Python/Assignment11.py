import xlrd
import MySQLdb

# Open the workbook and define the worksheet
book = xlrd.open_workbook("Assignment11.xlsx")
sheet = book.sheet_by_name("Sheet1")

# Establish a MySQL connection
database = MySQLdb.connect (host="localhost", user = "root", passwd = "", db = "Assignment11")

# Get the cursor, which is used to traverse the database, line by line
cursor = database.cursor()

# Create the INSERT INTO sql query
query = """INSERT INTO Stocks (Symbol, DateID, Open, High, Low, Close, Volume, Adj_Close) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""

# Create a For loop to iterate through each row in the XLS file, starting at row 2 to skip the headers
for r in range(2, sheet.nrows):
		Symbol		= sheet.cell(r,1).value
		DateID	    = sheet.cell(r,2).value
		Open		= sheet.cell(r,3).value
		High		= sheet.cell(r,4).value
		Low		    = sheet.cell(r,5).value
		Close	    = sheet.cell(r,6).value
		Volume		= sheet.cell(r,7).value
		Adj_Close	= sheet.cell(r,8).value

        # Assign values from each row
		values = (Symbol, DateID, Open, High, Low, Close, Volume, Adj_Close)

		# Execute sql Query
		cursor.execute(query, values)

# Close the cursor
cursor.close()

# Commit the transaction
database.commit()

# Close the database connection
database.close()

# Print results
print ("")
print ("All Done! Bye, for now.")
print ("")
columns = str(sheet.ncols)
rows = str(sheet.nrows)








